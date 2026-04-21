import torch
import torch.nn as nn


class MultiViewAggregator(nn.Module):
    def __init__(self, view_dim=200, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=view_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, sim_view, assoc_view, topo_view):
        views = torch.stack([sim_view, assoc_view, topo_view], dim=1)
        fused = self.transformer(views)
        return fused.reshape(fused.size(0), -1)
