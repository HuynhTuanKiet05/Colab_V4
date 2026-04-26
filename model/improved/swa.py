"""Stochastic Weight Averaging (Izmailov et al. 2018).

SWA snapshots the live model weights at a fixed cadence in the *late* portion
of training and averages them at the end, producing a checkpoint whose
generalization typically beats the best-by-AUC snapshot. Cost is one extra
state_dict copy in CPU/GPU memory (configurable) and an O(N_params) running
average; **no extra forward passes**, so SWA does not slow training down.

Usage in a training loop:

    swa = StochasticWeightAveraging(model, device=device)
    for epoch in range(num_epochs):
        # ... regular train + eval ...
        if epoch >= int(num_epochs * 0.7) and epoch % 10 == 0:
            swa.update(model)
    # at the very end, before final eval:
    swa.transfer_to(model)             # load averaged weights into model
    final_metrics = evaluate(model)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StochasticWeightAveraging:
    """Running average of model weights for SWA.

    Args:
        model: the model whose weights will be averaged.
        device: device to keep the running average on. Defaults to the model
            device. Pass `torch.device("cpu")` to save GPU memory at a small
            sync cost.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device if device is not None else next(model.parameters()).device
        self._avg_state = {
            k: v.detach().to(self.device).clone()
            for k, v in model.state_dict().items()
        }
        self.n = 1

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Add the current `model` weights to the running average."""
        self.n += 1
        for k, v in model.state_dict().items():
            avg = self._avg_state[k]
            new = v.detach().to(self.device)
            # Online mean: avg += (new - avg) / n
            if avg.dtype.is_floating_point:
                avg.add_(new - avg, alpha=1.0 / self.n)
            else:
                # Buffers like running_mean indices: just copy the latest.
                avg.copy_(new)

    @torch.no_grad()
    def transfer_to(self, model: nn.Module) -> None:
        """Load the averaged weights into `model` in-place."""
        if self.n == 0:
            return
        target_device = next(model.parameters()).device
        model.load_state_dict(
            {k: v.to(target_device) for k, v in self._avg_state.items()},
            strict=True,
        )

    def state_dict(self) -> dict:
        return {"avg_state": self._avg_state, "n": self.n}

    def load_state_dict(self, state: dict) -> None:
        self._avg_state = state["avg_state"]
        self.n = int(state["n"])

    @property
    def num_snapshots(self) -> int:
        return self.n
