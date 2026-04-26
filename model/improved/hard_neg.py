"""Adaptive hard-negative mining for DDA training.

Standard random negative sampling treats all negative drug-disease pairs as
equally informative. In practice the model quickly learns to push almost all
of them below the positives, and the gradient through them goes to zero -
the network spends most of its capacity on signal it has already mastered.

Adaptive hard-neg mining (Schroff et al. 2015 for FaceNet, applied here to
DDA) instead biases the negative pool toward pairs with the **highest**
predicted positive score: those are the negatives the model is currently
most confused about, and so contribute the largest gradient.

The "adaptive" part is a warmup ramp: at the start of training, scores are
random, so any "hardness" signal is noise. We ramp the hard-neg fraction
from 0 (random) at epoch 0 to a target (e.g. 0.5) by epoch ``warmup_epochs``,
then hold it constant. This avoids the well-known instability of starting
with hard negatives on a fresh model.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def hard_neg_ratio_schedule(
    epoch: int,
    warmup_epochs: int,
    target_ratio: float,
) -> float:
    """Linear ramp from 0 to `target_ratio` over `warmup_epochs` epochs."""
    if warmup_epochs <= 0:
        return target_ratio
    progress = min(1.0, max(0.0, epoch / float(warmup_epochs)))
    return progress * target_ratio


@torch.no_grad()
def sample_adaptive_hard_negatives(
    scores: torch.Tensor,
    labels: torch.Tensor,
    hard_ratio: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Build a sample weight vector that emphasizes hard negatives.

    Args:
        scores: ``[N, 2]`` raw logits emitted by the pair head this step.
        labels: ``[N]`` integer labels (0=negative, 1=positive).
        hard_ratio: fraction of total negatives to designate as "hard".
            Hard negatives keep their default weight; easy negatives are
            **down-weighted by 0.5** so they contribute half the gradient.
            Set to 0 for no mining (all negatives equal).
        rng: NumPy RNG (only used as fallback if no negatives are found).

    Returns:
        ``[N]`` float tensor of per-sample weights (positives = 1.0,
        hard negatives = 1.0, easy negatives = 0.5). Apply by multiplying
        into a per-sample loss before reduction.
    """
    if hard_ratio <= 0.0:
        return torch.ones_like(labels, dtype=scores.dtype)

    weights = torch.ones_like(labels, dtype=scores.dtype)
    neg_mask = labels == 0
    n_neg = int(neg_mask.sum().item())
    if n_neg == 0:
        return weights

    # Score of class "positive" - higher means model thinks the (negative)
    # pair is actually positive, i.e. it is a confusing/hard sample.
    pos_score = scores[:, 1]
    neg_pos_scores = pos_score[neg_mask]

    n_hard = max(1, int(n_neg * hard_ratio))
    if n_hard >= n_neg:
        # All negatives are hard - nothing to down-weight.
        return weights

    # Indices of hardest negatives within the *negative* subset.
    hardest_local = torch.topk(neg_pos_scores, k=n_hard, largest=True).indices
    # Map back to global indices.
    neg_global = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
    hardest_global = neg_global[hardest_local]

    # Mark every negative as easy first, then promote the hardest ones.
    weights[neg_mask] = 0.5
    weights[hardest_global] = 1.0
    return weights
