"""Training utilities for the improved AMNTDDA pipeline.

This module centralises:
    - FocalLoss (optional replacement for CrossEntropyLoss)
    - ModelEMA (exponential moving average of weights for evaluation)
    - ranking_loss (BPR-style margin loss with hard-negative mining)
    - get_adamw_param_groups (decoupled weight decay)
    - warmup_lr_factor (linear warmup schedule)

All components are additive and respect an explicit on/off switch from CLI flags
in ``train_final.py``; defaults preserve sensible behaviour when disabled.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Focal loss with optional class weighting + label smoothing
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss.

    loss_i = -alpha_{y_i} * (1 - p_{y_i})^gamma * log p_{y_i}

    When ``gamma == 0`` this degenerates to weighted cross-entropy, so a single
    code path covers both the focal and non-focal cases.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else torch.empty(0), persistent=False)
        self.label_smoothing = float(label_smoothing)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_prob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal = (1.0 - target_prob).clamp_min(1e-8).pow(self.gamma)

        if self.weight is not None and self.weight.numel() > 0:
            alpha = self.weight.gather(0, targets)
            focal = focal * alpha

        loss = -focal * target_log_prob

        if self.label_smoothing > 0.0:
            # Blend with per-class uniform log-likelihood as in nn.CrossEntropyLoss.
            smooth_loss = -log_probs.mean(dim=-1)
            if self.weight is not None and self.weight.numel() > 0:
                alpha = self.weight.gather(0, targets)
                smooth_loss = smooth_loss * alpha
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss.mean()


def compute_focal_gamma(epoch: int, warmup_epochs: int, gamma_start: float, gamma_end: float) -> float:
    """Linearly anneal focal ``gamma`` from ``gamma_start`` to ``gamma_end`` over ``warmup_epochs``."""
    if warmup_epochs <= 0:
        return gamma_end
    if epoch >= warmup_epochs:
        return gamma_end
    alpha = epoch / float(warmup_epochs)
    return gamma_start * (1.0 - alpha) + gamma_end * alpha


# -----------------------------------------------------------------------------
# Exponential Moving Average of model parameters
# -----------------------------------------------------------------------------
class ModelEMA:
    """Keep a shadow copy of the model parameters for evaluation.

    Call ``update(model)`` after each optimizer step, ``apply_to(model)`` before
    evaluation, and ``restore(model)`` after evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, ema_param in self.shadow.items():
            new_val = state[name].detach()
            if new_val.dtype != ema_param.dtype:
                new_val = new_val.to(dtype=ema_param.dtype)
            ema_param.mul_(self.decay).add_(new_val, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        self._backup = {name: state[name].detach().clone() for name in self.shadow}
        for name, ema_param in self.shadow.items():
            state[name].copy_(ema_param)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        state = model.state_dict()
        for name, orig in self._backup.items():
            state[name].copy_(orig)
        self._backup = None


# -----------------------------------------------------------------------------
# Ranking loss (BPR-style margin) with optional hard-negative mining
# -----------------------------------------------------------------------------
def ranking_loss(
    train_score: torch.Tensor,
    y_train: torch.Tensor,
    margin: float = 0.2,
    num_samples: int = 2048,
    hard_weight: float = 1.2,
    hard_ratio: float = 0.3,
) -> torch.Tensor:
    """Pairwise margin ranking loss on the positive class logit.

    ``train_score`` is the raw pair-head output of shape ``[N, 2]``. We
    convert it to a scalar ``logit_diff = score[:, 1] - score[:, 0]`` per
    pair, sample pos/neg pools from ``y_train``, and enforce::

        pos_score >= neg_score + margin

    A fraction ``hard_ratio`` of negatives are chosen from the top of the
    (detached) score distribution to act as hard negatives, and their
    contribution is up-weighted by ``hard_weight``.
    """
    if train_score.numel() == 0:
        return train_score.new_tensor(0.0)

    logit_diff = train_score[:, 1] - train_score[:, 0]
    pos = logit_diff[y_train == 1]
    neg = logit_diff[y_train == 0]

    if pos.numel() == 0 or neg.numel() == 0:
        return train_score.new_tensor(0.0)

    n = int(min(num_samples, pos.numel(), neg.numel()))
    if n <= 0:
        return train_score.new_tensor(0.0)

    pos_perm = torch.randperm(pos.numel(), device=pos.device)[:n]
    pos_sample = pos[pos_perm]

    n_hard = int(max(0, min(n, round(hard_ratio * n))))
    n_rand = n - n_hard

    loss_terms: List[torch.Tensor] = []
    weights: List[float] = []

    if n_hard > 0:
        k = int(min(n_hard, neg.numel()))
        hard_idx = neg.detach().topk(k).indices
        hard_neg = neg[hard_idx]
        hard_pos = pos_sample[:k]
        hard_loss = F.relu(margin - hard_pos + hard_neg).mean()
        loss_terms.append(hard_loss)
        weights.append(float(hard_weight))

    if n_rand > 0:
        rand_perm = torch.randperm(neg.numel(), device=neg.device)[:n_rand]
        rand_neg = neg[rand_perm]
        rand_pos = pos_sample[n_hard : n_hard + n_rand]
        if rand_pos.numel() < rand_neg.numel():
            # If there are not enough distinct positives, wrap around.
            rand_neg = rand_neg[: rand_pos.numel()]
        rand_loss = F.relu(margin - rand_pos + rand_neg).mean()
        loss_terms.append(rand_loss)
        weights.append(1.0)

    if not loss_terms:
        return train_score.new_tensor(0.0)

    total = sum(w * loss for w, loss in zip(weights, loss_terms))
    return total / float(sum(weights))


# -----------------------------------------------------------------------------
# AdamW param groups with decoupled weight decay (no WD on LN / bias)
# -----------------------------------------------------------------------------
_NO_DECAY_KEYS = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight", "norm.weight")


def get_adamw_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Split params into decayed / non-decayed groups for AdamW."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(key in name for key in _NO_DECAY_KEYS):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------------------
# LR warmup helper
# -----------------------------------------------------------------------------
def warmup_lr_factor(epoch: int, warmup_epochs: int) -> float:
    """Return a scalar in (0, 1] for linear LR warmup."""
    if warmup_epochs <= 0:
        return 1.0
    if epoch >= warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(warmup_epochs)


def apply_warmup_lr(optimizer, base_lrs: Iterable[float], epoch: int, warmup_epochs: int) -> float:
    """Scale optimizer learning rates by the warmup factor for ``epoch``.

    Returns the factor actually applied. After warmup is finished this is a
    no-op (factor == 1.0) so the plateau scheduler can take over.
    """
    factor = warmup_lr_factor(epoch, warmup_epochs)
    if epoch < warmup_epochs:
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = float(base) * factor
    return factor
