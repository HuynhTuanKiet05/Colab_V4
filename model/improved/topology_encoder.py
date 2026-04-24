import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyEncoder(nn.Module):
    """
    Enhanced topology encoder with GELU activation and residual connection.

    Improvements over baseline:
      - GELU activation (smoother gradient flow than ReLU for small-dim inputs)
      - Residual connection in hidden layer (prevents topology signal washout)
      - Preserves the 7 → hidden → hidden → out_dim architecture
    """

    def __init__(self, topo_feat_dim=7, hidden_dim=128, out_dim=200, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(topo_feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.norm_out = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, topo_features):
        # First layer: project 7-dim → hidden
        h = self.fc_in(topo_features)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Second layer with residual skip connection
        residual = h
        h = self.fc_hidden(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = h + residual  # Residual: prevents topology signal washout

        # Output projection
        h = self.fc_out(h)
        h = self.norm_out(h)
        return h
