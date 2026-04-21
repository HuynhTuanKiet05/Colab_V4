import torch
import torch.nn as nn


class FuzzyGate(nn.Module):
    def __init__(
        self,
        base_dim,
        topo_dim,
        dropout=0.1,
        gate_mode="scalar",
        gate_bias_init=-2.0,
    ):
        super().__init__()
        if gate_mode not in {"scalar", "vector"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        gate_out_dim = 1 if gate_mode == "scalar" else base_dim
        self.gate_mode = gate_mode

        self.topo_proj = nn.Sequential(
            nn.Linear(topo_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(base_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, gate_out_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.topo_norm = nn.LayerNorm(base_dim)
        nn.init.constant_(self.gate[-2].bias, gate_bias_init)

    def forward(self, x_base, x_topo, return_details=False):
        topo_proj = self.topo_norm(self.dropout(self.topo_proj(x_topo)))
        gate = self.gate(torch.cat([x_base, topo_proj], dim=-1))
        residual = gate * topo_proj
        enhanced = x_base + residual

        if return_details:
            return enhanced, {
                "gate": gate,
                "topo_proj": topo_proj,
                "residual": residual,
            }
        return enhanced
