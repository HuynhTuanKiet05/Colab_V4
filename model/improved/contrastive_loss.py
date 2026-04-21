import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, view1, view2):
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, labels)


class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.cl = ContrastiveLoss(temperature=temperature)

    def forward(self, sim_view, assoc_view, topo_view):
        loss_sa = self.cl(sim_view, assoc_view)
        loss_st = self.cl(sim_view, topo_view)
        loss_at = self.cl(assoc_view, topo_view)
        return (loss_sa + loss_st + loss_at) / 3.0
