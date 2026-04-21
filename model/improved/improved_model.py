import os

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from AMDGT_original.model import gt_net_disease, gt_net_drug

from .contrastive_loss import MultiViewContrastiveLoss
from .multi_view_aggregator import MultiViewAggregator
from .topology_encoder import TopologyEncoder


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.runtime_device = torch.device(os.environ.get("AMDGT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

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

        self.num_node_types = 3
        self.num_edge_types = 6

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

        topo_feat_dim = getattr(args, "topo_feat_dim", 7)
        topo_hidden = getattr(args, "topo_hidden", 128)
        temperature = getattr(args, "temperature", getattr(args, "contrastive_temperature", 0.5))

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

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def _compute_hgt_views(self, drdipr_graph, drug_feature, disease_feature, protein_feature):
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
        drug_hgt = feature[:drug_count, :]
        disease_hgt = feature[drug_count:drug_count + disease_count, :]
        return drug_hgt, disease_hgt

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
        drug_hgt, disease_hgt = self._compute_hgt_views(drdipr_graph, drug_feature, disease_feature, protein_feature)

        if drug_topo_feat is None:
            drug_topo = torch.zeros_like(drug_sim)
        else:
            drug_topo = self.drug_topology_encoder(drug_topo_feat)

        if disease_topo_feat is None:
            disease_topo = torch.zeros_like(disease_sim)
        else:
            disease_topo = self.disease_topology_encoder(disease_topo_feat)

        contrastive_drug = self.contrastive_loss(drug_sim, drug_hgt, drug_topo)
        contrastive_disease = self.contrastive_loss(disease_sim, disease_hgt, disease_topo)
        contrastive = 0.5 * (contrastive_drug + contrastive_disease)

        drug_repr = self.drug_multi_view(drug_sim, drug_hgt, drug_topo)
        disease_repr = self.disease_multi_view(disease_sim, disease_hgt, disease_topo)

        pair_embedding = drug_repr[sample[:, 0].long()] * disease_repr[sample[:, 1].long()]
        output = self.mlp(pair_embedding)

        aux_losses = {
            "contrastive": contrastive,
        }
        diagnostics = {
            "drug_sim_hgt_cos": float(F.cosine_similarity(drug_sim, drug_hgt, dim=-1).mean().item()),
            "drug_sim_topo_cos": float(F.cosine_similarity(drug_sim, drug_topo, dim=-1).mean().item()),
            "disease_sim_hgt_cos": float(F.cosine_similarity(disease_sim, disease_hgt, dim=-1).mean().item()),
            "disease_sim_topo_cos": float(F.cosine_similarity(disease_sim, disease_topo, dim=-1).mean().item()),
            "contrastive": float(contrastive.item()),
        }

        if return_aux and return_diagnostics:
            return drug_repr, output, aux_losses, diagnostics
        if return_aux:
            return drug_repr, output, aux_losses
        if return_diagnostics:
            return drug_repr, output, diagnostics
        return drug_repr, output
