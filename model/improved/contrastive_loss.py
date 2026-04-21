import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss with learnable temperature.

    Improvement: log-parameterized learnable temperature adapts during
    training to find the optimal sharpness for the similarity distribution.
    Fixed temperature may be suboptimal — too high blurs positives/negatives,
    too low causes gradient vanishing on hard negatives.
    """

    def __init__(self, temperature=0.5, learnable_temperature=True):
        super().__init__()
        if learnable_temperature:
            # Log-parameterization ensures temperature stays positive
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(temperature)))

    @property
    def temperature(self):
        # Clamp to prevent numerical instability
        return self.log_temperature.exp().clamp(min=0.01, max=5.0)

    def forward(self, view1, view2):
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, labels)


class MultiViewContrastiveLoss(nn.Module):
    """
    Contrastive loss across all view pairs with shared learnable temperature.

    For 3 views (sim, assoc, topo), computes InfoNCE for all 3 pairs
    and averages. The shared learnable temperature lets the model
    auto-tune alignment sharpness during training.
    """

    def __init__(self, temperature=0.5, learnable_temperature=True):
        super().__init__()
        self.cl = ContrastiveLoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

    def forward(self, sim_view, assoc_view, topo_view):
        loss_sa = self.cl(sim_view, assoc_view)
        loss_st = self.cl(sim_view, topo_view)
        loss_at = self.cl(assoc_view, topo_view)
        return (loss_sa + loss_st + loss_at) / 3.0
