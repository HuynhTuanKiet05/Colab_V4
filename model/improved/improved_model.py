import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from AMDGT_original.model import gt_net_drug, gt_net_disease
from .rlg_hgt import RLGHGT


def _valid_num_heads(dim, preferred):
    for heads in range(min(preferred, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class MultiViewFusion(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        hidden = max(dim // 2, 64)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, views):
        weights = torch.softmax(self.score(views).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * views).sum(dim=1)
        return self.out_norm(fused), weights


class TopologyGate(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, sim_repr, hgt_repr):
        topo_repr = self.project(torch.cat([sim_repr, hgt_repr], dim=-1))
        gate = self.net(torch.cat([sim_repr, hgt_repr, topo_repr], dim=-1))
        fused = sim_repr + gate * (hgt_repr + topo_repr - sim_repr)
        return self.out_norm(fused), gate, topo_repr


class PairScorer(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        pair_dim = dim * 6 + 4
        hidden = max(dim * 3, 256)
        compact = max(dim * 2, 192)

        self.bilinear = nn.Bilinear(dim, dim, 2)
        self.main = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, compact),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compact, 2),
        )
        self.skip = nn.Linear(pair_dim, 2)
        self.topology_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, drug_repr, disease_repr, topology_score=None):
        pair_mul = drug_repr * disease_repr
        pair_diff = torch.abs(drug_repr - disease_repr)
        pair_sum = drug_repr + disease_repr
        pair_sqdiff = (drug_repr - disease_repr) ** 2
        pair_dot = (drug_repr * disease_repr).sum(dim=-1, keepdim=True)
        pair_cos = F.cosine_similarity(drug_repr, disease_repr, dim=-1).unsqueeze(-1)
        if topology_score is None:
            topology_score = torch.zeros_like(pair_dot)
        topology_score = self.topology_scale * topology_score
        fuzzy_close = torch.exp(-pair_diff.mean(dim=-1, keepdim=True))

        full_features = torch.cat([
            drug_repr, disease_repr, pair_mul, pair_diff, pair_sum, pair_sqdiff, pair_dot, pair_cos, topology_score, fuzzy_close
        ], dim=-1)
        local_features = torch.cat([pair_mul, pair_diff, pair_sum, pair_cos, pair_dot, topology_score, fuzzy_close], dim=-1)

        score_main = self.main(full_features)
        score_bilinear = self.bilinear(drug_repr, disease_repr)
        return 0.6 * score_main + 0.4 * score_bilinear + self.skip(full_features)


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.runtime_device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.drug_feature_dim = 300
        self.disease_feature_dim = 64
        self.protein_feature_dim = 320
        self.contrastive_temperature = getattr(args, 'contrastive_temperature', 0.2)
        self.cached_aux = {}

        self.drug_linear = nn.Linear(self.drug_feature_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(self.disease_feature_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(self.protein_feature_dim, args.hgt_in_dim)
        self.drug_norm = nn.LayerNorm(args.hgt_in_dim)
        self.disease_norm = nn.LayerNorm(args.hgt_in_dim)
        self.protein_norm = nn.LayerNorm(args.hgt_in_dim)
        self.input_dropout = nn.Dropout(args.dropout)

        self.drug_view_encoders = nn.ModuleDict({
            'fingerprint': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'gip': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'consensus': gt_net_drug.GraphTransformer(self.runtime_device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
        })
        self.disease_view_encoders = nn.ModuleDict({
            'phenotype': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'gip': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
            'consensus': gt_net_disease.GraphTransformer(self.runtime_device, args.gt_layer, args.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout),
        })

        canonical_etypes = [
            ('drug', 'association', 'disease'),
            ('disease', 'association_rev', 'drug'),
            ('drug', 'association', 'protein'),
            ('protein', 'association_rev', 'drug'),
            ('disease', 'association', 'protein'),
            ('protein', 'association_rev', 'disease'),
        ]
        node_types = ['drug', 'disease', 'protein']
        self.hgt = RLGHGT(
            hidden_dim=args.hgt_in_dim,
            out_dim=args.hgt_in_dim,
            num_heads=args.hgt_head,
            num_layers=args.hgt_layer,
            canonical_etypes=canonical_etypes,
            node_types=node_types,
            dropout=args.dropout,
            use_relation_attention=getattr(args, 'use_relation_attention', True),
            use_metapath=getattr(args, 'use_metapath', True),
            use_global=getattr(args, 'use_global_hgt', True),
            use_topological=getattr(args, 'use_topological', True),
        )

        self.hgt_drug_out = nn.Sequential(nn.Linear(args.hgt_in_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))
        self.hgt_disease_out = nn.Sequential(nn.Linear(args.hgt_in_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))
        self.drug_view_fusion = MultiViewFusion(args.gt_out_dim, args.dropout)
        self.disease_view_fusion = MultiViewFusion(args.gt_out_dim, args.dropout)
        self.drug_topology_gate = TopologyGate(args.gt_out_dim, args.dropout)
        self.disease_topology_gate = TopologyGate(args.gt_out_dim, args.dropout)

        align_dim = min(max(args.gt_out_dim // 2, 64), 256)
        self.drug_align_sim = nn.Linear(args.gt_out_dim, align_dim)
        self.drug_align_hgt = nn.Linear(args.gt_out_dim, align_dim)
        self.disease_align_sim = nn.Linear(args.gt_out_dim, align_dim)
        self.disease_align_hgt = nn.Linear(args.gt_out_dim, align_dim)

        self.topology_scorer = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, args.gt_out_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gt_out_dim, 1),
        )
        self.topology_scale = nn.Parameter(torch.tensor(0.25))
        self.pair_scorer = PairScorer(args.gt_out_dim, args.dropout)

    def _prepare_graph_dict(self, graph_input, graph_names):
        if isinstance(graph_input, dict):
            return graph_input
        return {name: graph_input for name in graph_names}

    def _encode_similarity_views(self, graph_input, encoders):
        prepared = self._prepare_graph_dict(graph_input, encoders.keys())
        fallback_graph = prepared.get('consensus', next(iter(prepared.values())))
        fallback = encoders['consensus'](fallback_graph)
        outputs = {'consensus': fallback}
        for name, encoder in encoders.items():
            graph = prepared.get(name)
            outputs[name] = encoder(graph) if graph is not None else fallback
        return outputs

    def _contrastive_loss(self, lhs, rhs):
        lhs = F.normalize(lhs, dim=-1)
        rhs = F.normalize(rhs, dim=-1)
        logits = lhs @ rhs.t() / self.contrastive_temperature
        labels = torch.arange(lhs.shape[0], device=lhs.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample, edge_stats=None, return_aux=False):
        drug_views = self._encode_similarity_views(drdr_graph, self.drug_view_encoders)
        disease_views = self._encode_similarity_views(didi_graph, self.disease_view_encoders)

        drug_view_stack = torch.stack([drug_views['fingerprint'], drug_views['gip'], drug_views['consensus']], dim=1)
        disease_view_stack = torch.stack([disease_views['phenotype'], disease_views['gip'], disease_views['consensus']], dim=1)

        drug_view_fused, drug_view_weights = self.drug_view_fusion(drug_view_stack)
        disease_view_fused, disease_view_weights = self.disease_view_fusion(disease_view_stack)

        hgt_drug_feature = self.input_dropout(self.drug_norm(self.drug_linear(drug_feature)))
        hgt_disease_feature = self.input_dropout(self.disease_norm(self.disease_linear(disease_feature)))
        hgt_protein_feature = self.input_dropout(self.protein_norm(self.protein_linear(protein_feature)))
        feature_dict = {
            'drug': hgt_drug_feature,
            'disease': hgt_disease_feature,
            'protein': hgt_protein_feature,
        }

        if drdipr_graph.device != hgt_drug_feature.device:
            drdipr_graph = drdipr_graph.to(hgt_drug_feature.device)
        hgt_out = self.hgt(drdipr_graph, feature_dict)
        drug_hgt = self.hgt_drug_out(hgt_out['drug'])
        disease_hgt = self.hgt_disease_out(hgt_out['disease'])

        drug_topo, drug_gate, drug_topo_repr = self.drug_topology_gate(drug_view_fused, drug_hgt)
        disease_topo, disease_gate, disease_topo_repr = self.disease_topology_gate(disease_view_fused, disease_hgt)

        pair_drug = drug_topo[sample[:, 0]]
        pair_disease = disease_topo[sample[:, 1]]
        topology_score = self.topology_scale * torch.tanh(self.topology_scorer(torch.cat([pair_drug, pair_disease], dim=-1)))
        edge_bias = torch.zeros_like(topology_score)
        if edge_stats is not None:
            edge_bias = edge_stats.get('pair_bias', edge_bias)
        pair_topology = topology_score + edge_bias
        output = self.pair_scorer(pair_drug, pair_disease, topology_score=pair_topology)

        topology_align = self._contrastive_loss(self.drug_align_sim(drug_view_fused), self.drug_align_hgt(drug_topo_repr))
        topology_align = topology_align + self._contrastive_loss(self.disease_align_sim(disease_view_fused), self.disease_align_hgt(disease_topo_repr))

        self.cached_aux = {
            'contrastive': self._contrastive_loss(self.drug_align_sim(drug_view_fused), self.drug_align_hgt(drug_hgt))
            + self._contrastive_loss(self.disease_align_sim(disease_view_fused), self.disease_align_hgt(disease_hgt))
            + 0.5 * topology_align,
            'drug_view_weights': drug_view_weights.detach(),
            'disease_view_weights': disease_view_weights.detach(),
            'drug_gate_mean': float(drug_gate.mean().item()),
            'disease_gate_mean': float(disease_gate.mean().item()),
            'topology_score': pair_topology.detach(),
            'edge_bias': edge_bias.detach(),
            'drug_repr_norm': torch.norm(drug_topo, dim=-1).mean().detach(),
            'disease_repr_norm': torch.norm(disease_topo, dim=-1).mean().detach(),
            'pair_topology_mean': pair_topology.mean().detach(),
        }

        if return_aux:
            return drug_topo, output, self.cached_aux
        return drug_topo, output
