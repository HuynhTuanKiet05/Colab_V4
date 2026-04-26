"""Pair-level Mixup (Zhang et al. 2018) adapted for drug-disease pairs.

Standard Mixup interpolates *features* of two random training samples and
their *labels* with the same coefficient lambda ~ Beta(alpha, alpha). Here
we cannot interpolate node features directly because each pair indexes into
a node-level representation table - instead we interpolate the **scores**
emitted by the pair head and the **one-hot labels** for that pair.

This is mathematically equivalent to classical Mixup whenever the pair head
is the last layer of the network, which holds for both `mul_mlp` and the
new `moe` head used in the improved recipe.

Cost: a permutation + scalar Beta sample per epoch. No extra forward.

Returns:
    A `(mixed_score, label_a, label_b, lam)` tuple. The training loss is
    computed as `lam * loss(mixed_score, label_a) + (1 - lam) * loss(mixed_score, label_b)`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def pair_mixup(
    score: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup at the score / label level.

    Args:
        score: ``[N, num_classes]`` raw logits from the pair head.
        labels: ``[N]`` integer class labels (0/1 for binary DDA).
        alpha: Beta distribution parameter. ``0`` disables mixup (returns
            inputs unchanged with lam=1.0). Typical values: 0.1-0.4.
        rng: NumPy RNG to draw the lambda and permutation.

    Returns:
        ``(mixed_score, labels_a, labels_b, lam)``. The first three tensors
        live on the same device/dtype as the inputs; lam is a Python float
        in [0, 1].
    """
    if alpha <= 0.0 or score.size(0) <= 1:
        return score, labels, labels, 1.0

    lam = float(rng.beta(alpha, alpha))
    # Beta(alpha, alpha) is symmetric around 0.5; clip extreme values that
    # would make the loss degenerate (lam=0 or 1 == no mixup).
    lam = max(min(lam, 1.0 - 1e-3), 1e-3)

    perm_np = rng.permutation(score.size(0))
    perm = torch.as_tensor(perm_np, device=score.device, dtype=torch.long)

    mixed_score = lam * score + (1.0 - lam) * score[perm]
    labels_b = labels[perm]
    return mixed_score, labels, labels_b, lam


def mixup_loss(
    criterion: torch.nn.Module,
    mixed_score: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Convenience wrapper for the standard Mixup loss formula."""
    if lam >= 1.0 - 1e-6:
        return criterion(mixed_score, labels_a)
    return lam * criterion(mixed_score, labels_a) + (1.0 - lam) * criterion(mixed_score, labels_b)
