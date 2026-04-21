import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HGTConv


class RelationAwareLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_node_types, num_rel_types, dropout=0.2):
        super().__init__()
        head_dim = max(1, out_dim // num_heads)
        self.hgt_base = HGTConv(in_dim, head_dim, num_heads, num_node_types, num_rel_types)
        self.residual_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, g, h, ntype, etype):
        h_out = self.hgt_base(g, h, ntype, etype, presorted=True)
        residual = self.residual_proj(h)
        return self.norm(residual + self.dropout(h_out))


class LayerAggregator(nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, layer_outputs):
        weight = F.softmax(self.weights, dim=0)
        mixed = 0.0
        for idx, tensor in enumerate(layer_outputs):
            mixed = mixed + weight[idx] * tensor
        return self.norm(mixed)


class RLGHGT(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        num_heads,
        num_layers,
        canonical_etypes,
        node_types,
        dropout=0.2,
        use_relation_attention=True,
        use_metapath=True,
        use_global=True,
        use_topological=True,
    ):
        super().__init__()
        del use_relation_attention, use_metapath, use_global, use_topological

        self.node_types = list(node_types)
        self.num_node_types = len(self.node_types)
        self.num_rel_types = len(canonical_etypes)
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_dim = hidden_dim if layer_idx > 0 else hidden_dim
            self.layers.append(
                RelationAwareLayer(
                    in_dim=layer_in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    num_node_types=self.num_node_types,
                    num_rel_types=self.num_rel_types,
                    dropout=dropout,
                )
            )
        self.layer_agg = LayerAggregator(num_layers, out_dim)

    def forward(self, g, feature_dict):
        with g.local_scope():
            g.ndata["h"] = feature_dict
            homo_graph = dgl.to_homogeneous(g, ndata="h")
            h = homo_graph.ndata["h"]
            ntype = homo_graph.ndata["_TYPE"]
            etype = homo_graph.edata["_TYPE"]

            layer_outputs = []
            for layer in self.layers:
                h = layer(homo_graph, h, ntype, etype)
                layer_outputs.append(h)

            mixed = self.layer_agg(layer_outputs)
            out = {}
            for node_type in self.node_types:
                type_id = g.get_ntype_id(node_type)
                out[node_type] = mixed[ntype == type_id]
            return out
