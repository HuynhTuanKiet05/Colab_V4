import os

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from AMDGT_original.model import gt_net_disease, gt_net_drug

from .contrastive_loss import MultiViewContrastiveLoss
from .fuzzy_attention import FuzzyGate
from .multi_view_aggregator import MultiViewAggregator
from .rlg_hgt import RLGHGT
from .topology_encoder import TopologyEncoder


class PairInteractionHead(nn.Module):
    def __init__(self, node_dim, dropout=0.3, mode="interaction"):
        super().__init__()
        self.mode = mode

        if mode == "mul_mlp":
            in_dim = node_dim
            self.main = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2),
            )
            self.skip = nn.Linear(in_dim, 2)
            self.bilinear = None
        elif mode == "interaction":
            in_dim = node_dim * 4 + 2
            self.main = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )
            self.skip = nn.Linear(in_dim, 2)
            self.bilinear = nn.Bilinear(node_dim, node_dim, 1)
        else:
            raise ValueError(f"Unsupported pair interaction mode: {mode}")

    def forward(self, drug_repr, disease_repr):
        if self.mode == "mul_mlp":
            features = drug_repr * disease_repr
        else:
            pair_mul = drug_repr * disease_repr
            pair_abs = torch.abs(drug_repr - disease_repr)
            pair_cos = F.cosine_similarity(drug_repr, disease_repr, dim=-1).unsqueeze(-1)
            pair_bilinear = self.bilinear(drug_repr, disease_repr)
            features = torch.cat([drug_repr, disease_repr, pair_mul, pair_abs, pair_cos, pair_bilinear], dim=-1)
        return self.main(features) + self.skip(features)


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.runtime_device = torch.device(os.environ.get("AMDGT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        self.assoc_backbone = getattr(args, "assoc_backbone", "rlghgt")
        self.fusion_mode = getattr(args, "fusion_mode", "mva_fuzzy")
        self.pair_mode = getattr(args, "pair_mode", "interaction")
        self.gate_mode = getattr(args, "gate_mode", "vector")
        self.gate_bias_init = getattr(args, "gate_bias_init", -2.0)

        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        self.gt_drug = gt_net_drug.GraphTransformer(
            self.runtime_device,
            args.gt_layer,
            args.drug_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout,
        )
        self.gt_disease = gt_net_disease.GraphTransformer(
            self.runtime_device,
            args.gt_layer,
            args.disease_number,
            args.gt_out_dim,
            args.gt_out_dim,
            args.gt_head,
            args.dropout,
        )

        if not hasattr(args, "hgt_head_dim") or args.hgt_head_dim is None:
            args.hgt_head_dim = max(1, args.gt_out_dim // args.hgt_head)

        self.canonical_etypes = [
            ("drug", "association", "disease"),
            ("disease", "association_rev", "drug"),
            ("drug", "association", "protein"),
            ("protein", "association_rev", "drug"),
            ("disease", "association", "protein"),
            ("protein", "association_rev", "disease"),
        ]
        self.node_types = ["drug", "disease", "protein"]

        if self.assoc_backbone == "vanilla_hgt":
            self.num_node_types = len(self.node_types)
            self.num_edge_types = len(self.canonical_etypes)
            hgt_layers = []
            for layer_idx in range(args.hgt_layer):
                if layer_idx == args.hgt_layer - 1:
                    out_head_dim = args.hgt_head_dim
                else:
                    out_head_dim = max(1, args.hgt_in_dim // args.hgt_head)
                hgt_layers.append(
                    dgl.nn.pytorch.conv.HGTConv(
                        args.hgt_in_dim,
                        out_head_dim,
                        args.hgt_head,
                        self.num_node_types,
                        self.num_edge_types,
                        args.dropout,
                    )
                )
            self.hgt_layers = nn.ModuleList(hgt_layers)
            assoc_dim = args.hgt_head_dim * args.hgt_head
            self.rlg_hgt = None
        elif self.assoc_backbone == "rlghgt":
            self.hgt_layers = None
            self.rlg_hgt = RLGHGT(
                hidden_dim=args.hgt_in_dim,
                out_dim=args.hgt_in_dim,
                num_heads=args.hgt_head,
                num_layers=args.hgt_layer,
                canonical_etypes=self.canonical_etypes,
                node_types=self.node_types,
                dropout=args.dropout,
                use_relation_attention=getattr(args, "use_relation_attention", True),
                use_metapath=getattr(args, "use_metapath", True),
                use_global=getattr(args, "use_global_hgt", True),
                use_topological=getattr(args, "use_topological", True),
            )
            assoc_dim = args.hgt_in_dim
        else:
            raise ValueError(f"Unsupported assoc_backbone: {self.assoc_backbone}")

        self.drug_assoc_proj = nn.Sequential(nn.Linear(assoc_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))
        self.disease_assoc_proj = nn.Sequential(nn.Linear(assoc_dim, args.gt_out_dim), nn.LayerNorm(args.gt_out_dim))

        topo_feat_dim = getattr(args, "topo_feat_dim", 7)
        topo_hidden = getattr(args, "topo_hidden", 128)
        temperature = getattr(args, "temperature", getattr(args, "contrastive_temperature", 0.5))
        node_repr_dim = args.gt_out_dim * 3

        self.drug_topology_encoder = TopologyEncoder(
            topo_feat_dim=topo_feat_dim,
            hidden_dim=topo_hidden,
            out_dim=args.gt_out_dim,
            dropout=args.dropout,
        )
        self.disease_topology_encoder = TopologyEncoder(
            topo_feat_dim=topo_feat_dim,
            hidden_dim=topo_hidden,
            out_dim=args.gt_out_dim,
            dropout=args.dropout,
        )

        self.contrastive_loss = MultiViewContrastiveLoss(temperature=temperature)
        self.drug_multi_view = MultiViewAggregator(
            view_dim=args.gt_out_dim,
            nhead=args.tr_head,
            num_layers=args.tr_layer,
            dropout=args.dropout,
        )
        self.disease_multi_view = MultiViewAggregator(
            view_dim=args.gt_out_dim,
            nhead=args.tr_head,
            num_layers=args.tr_layer,
            dropout=args.dropout,
        )

        if self.fusion_mode == "mva_fuzzy":
            self.drug_fuzzy_gate = FuzzyGate(
                base_dim=node_repr_dim,
                topo_dim=args.gt_out_dim,
                dropout=args.dropout,
                gate_mode=self.gate_mode,
                gate_bias_init=self.gate_bias_init,
            )
            self.disease_fuzzy_gate = FuzzyGate(
                base_dim=node_repr_dim,
                topo_dim=args.gt_out_dim,
                dropout=args.dropout,
                gate_mode=self.gate_mode,
                gate_bias_init=self.gate_bias_init,
            )
        elif self.fusion_mode == "mva":
            self.drug_fuzzy_gate = None
            self.disease_fuzzy_gate = None
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        self.pair_head = PairInteractionHead(node_repr_dim, dropout=args.dropout, mode=self.pair_mode)

    def _compute_vanilla_assoc_views(self, drdipr_graph, drug_feature, disease_feature, protein_feature):
        if drdipr_graph.device != drug_feature.device:
            drdipr_graph = drdipr_graph.to(drug_feature.device)

        feature_dict = {
            "drug": self.drug_linear(drug_feature),
            "disease": self.disease_linear(disease_feature),
            "protein": self.protein_linear(protein_feature),
        }

        with drdipr_graph.local_scope():
            drdipr_graph.ndata["h"] = feature_dict
            homo_graph = dgl.to_homogeneous(drdipr_graph, ndata="h")
            feature = homo_graph.ndata["h"]
            for layer in self.hgt_layers:
                feature = layer(homo_graph, feature, homo_graph.ndata["_TYPE"], homo_graph.edata["_TYPE"], presorted=True)

        drug_count = self.args.drug_number
        disease_count = self.args.disease_number
        drug_assoc = feature[:drug_count, :]
        disease_assoc = feature[drug_count:drug_count + disease_count, :]
        return drug_assoc, disease_assoc

    def _compute_rlghgt_assoc_views(self, drdipr_graph, drug_feature, disease_feature, protein_feature):
        if drdipr_graph.device != drug_feature.device:
            drdipr_graph = drdipr_graph.to(drug_feature.device)

        feature_dict = {
            "drug": self.drug_linear(drug_feature),
            "disease": self.disease_linear(disease_feature),
            "protein": self.protein_linear(protein_feature),
        }
        assoc_out = self.rlg_hgt(drdipr_graph, feature_dict)
        return assoc_out["drug"], assoc_out["disease"]

    def _compute_assoc_views(self, drdipr_graph, drug_feature, disease_feature, protein_feature):
        if self.assoc_backbone == "vanilla_hgt":
            drug_assoc, disease_assoc = self._compute_vanilla_assoc_views(drdipr_graph, drug_feature, disease_feature, protein_feature)
        else:
            drug_assoc, disease_assoc = self._compute_rlghgt_assoc_views(drdipr_graph, drug_feature, disease_feature, protein_feature)

        drug_assoc = self.drug_assoc_proj(drug_assoc)
        disease_assoc = self.disease_assoc_proj(disease_assoc)
        return drug_assoc, disease_assoc

    def _fuse_entity(self, sim_view, assoc_view, topo_view, aggregator, fuzzy_gate):
        base_repr = aggregator(sim_view, assoc_view, topo_view)
        if fuzzy_gate is None:
            gate_details = None
            final_repr = base_repr
        else:
            final_repr, gate_details = fuzzy_gate(base_repr, topo_view, return_details=True)
        return base_repr, final_repr, gate_details

    @staticmethod
    def _gate_stats(prefix, gate_details):
        if gate_details is None:
            return {
                f"{prefix}_gate_mean": 0.0,
                f"{prefix}_gate_std": 0.0,
                f"{prefix}_residual_norm": 0.0,
            }
        gate = gate_details["gate"]
        residual = gate_details["residual"]
        return {
            f"{prefix}_gate_mean": float(gate.mean().item()),
            f"{prefix}_gate_std": float(gate.std(unbiased=False).item()),
            f"{prefix}_residual_norm": float(torch.norm(residual, dim=-1).mean().item()),
        }

    def forward(
        self,
        drdr_graph,
        didi_graph,
        drdipr_graph,
        drug_feature,
        disease_feature,
        protein_feature,
        sample,
        drug_topo_feat=None,
        disease_topo_feat=None,
        edge_stats=None,
        return_aux=False,
        return_diagnostics=False,
    ):
        del edge_stats

        drug_sim = self.gt_drug(drdr_graph)
        disease_sim = self.gt_disease(didi_graph)
        drug_assoc, disease_assoc = self._compute_assoc_views(drdipr_graph, drug_feature, disease_feature, protein_feature)

        if drug_topo_feat is None:
            drug_topo = torch.zeros_like(drug_sim)
        else:
            drug_topo = self.drug_topology_encoder(drug_topo_feat)

        if disease_topo_feat is None:
            disease_topo = torch.zeros_like(disease_sim)
        else:
            disease_topo = self.disease_topology_encoder(disease_topo_feat)

        contrastive_drug = self.contrastive_loss(drug_sim, drug_assoc, drug_topo)
        contrastive_disease = self.contrastive_loss(disease_sim, disease_assoc, disease_topo)
        contrastive = 0.5 * (contrastive_drug + contrastive_disease)

        drug_base, drug_repr, drug_gate_details = self._fuse_entity(
            drug_sim,
            drug_assoc,
            drug_topo,
            self.drug_multi_view,
            self.drug_fuzzy_gate,
        )
        disease_base, disease_repr, disease_gate_details = self._fuse_entity(
            disease_sim,
            disease_assoc,
            disease_topo,
            self.disease_multi_view,
            self.disease_fuzzy_gate,
        )

        pair_drug = drug_repr[sample[:, 0].long()]
        pair_disease = disease_repr[sample[:, 1].long()]
        output = self.pair_head(pair_drug, pair_disease)

        aux_losses = {
            "contrastive": contrastive,
        }
        diagnostics = {
            "assoc_backbone": self.assoc_backbone,
            "fusion_mode": self.fusion_mode,
            "pair_mode": self.pair_mode,
            "drug_sim_assoc_cos": float(F.cosine_similarity(drug_sim, drug_assoc, dim=-1).mean().item()),
            "drug_sim_topo_cos": float(F.cosine_similarity(drug_sim, drug_topo, dim=-1).mean().item()),
            "disease_sim_assoc_cos": float(F.cosine_similarity(disease_sim, disease_assoc, dim=-1).mean().item()),
            "disease_sim_topo_cos": float(F.cosine_similarity(disease_sim, disease_topo, dim=-1).mean().item()),
            "drug_assoc_norm": float(torch.norm(drug_assoc, dim=-1).mean().item()),
            "disease_assoc_norm": float(torch.norm(disease_assoc, dim=-1).mean().item()),
            "drug_base_norm": float(torch.norm(drug_base, dim=-1).mean().item()),
            "disease_base_norm": float(torch.norm(disease_base, dim=-1).mean().item()),
            "contrastive": float(contrastive.item()),
        }
        diagnostics.update(self._gate_stats("drug", drug_gate_details))
        diagnostics.update(self._gate_stats("disease", disease_gate_details))

        if return_aux and return_diagnostics:
            return drug_repr, output, aux_losses, diagnostics
        if return_aux:
            return drug_repr, output, aux_losses
        if return_diagnostics:
            return drug_repr, output, diagnostics
        return drug_repr, output
