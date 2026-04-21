import torch.nn as nn


class TopologyEncoder(nn.Module):
    def __init__(self, topo_feat_dim=7, hidden_dim=128, out_dim=200, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, topo_features):
        return self.encoder(topo_features)
