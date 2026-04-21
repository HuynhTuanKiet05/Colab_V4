import torch
import torch.nn as nn


def _resolve_heads(view_dim, requested_heads):
    heads = max(1, min(requested_heads, view_dim))
    while heads > 1 and view_dim % heads != 0:
        heads -= 1
    return heads


class SimilarityViewFusion(nn.Module):
    def __init__(self, view_dim, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        effective_heads = _resolve_heads(view_dim, nhead)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=view_dim,
            nhead=effective_heads,
            dim_feedforward=view_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        hidden_dim = max(32, view_dim // 2)
        self.view_scorer = nn.Sequential(
            nn.LayerNorm(view_dim),
            nn.Linear(view_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(view_dim)

    def forward(self, view_tensors, return_weights=False):
        if not view_tensors:
            raise ValueError("SimilarityViewFusion requires at least one view tensor.")

        if len(view_tensors) == 1:
            fused = self.output_norm(view_tensors[0])
            if return_weights:
                weights = torch.ones(fused.size(0), 1, device=fused.device, dtype=fused.dtype)
                return fused, weights
            return fused

        views = torch.stack(view_tensors, dim=1)
        encoded = self.transformer(views)
        scores = self.view_scorer(encoded)
        weights = torch.softmax(scores, dim=1)
        fused = self.output_norm((weights * encoded).sum(dim=1))
        if return_weights:
            return fused, weights.squeeze(-1)
        return fused
