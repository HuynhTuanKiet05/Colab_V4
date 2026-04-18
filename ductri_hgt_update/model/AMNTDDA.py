import os
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease
from model.rlg_hgt import RLGHGT

device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.hgt_drug_out = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        self.hgt_disease_out = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head, dropout=args.dropout, batch_first=True)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

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

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )


    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        disease_feature = self.disease_linear(disease_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        if drdipr_graph.device != drug_feature.device:
            drdipr_graph = drdipr_graph.to(drug_feature.device)
        hgt_out = self.hgt(drdipr_graph, feature_dict)

        dr_hgt = self.hgt_drug_out(hgt_out['drug'])
        di_hgt = self.hgt_disease_out(hgt_out['disease'])

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.reshape(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.reshape(self.args.disease_number, 2 * self.args.gt_out_dim)

        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        return dr, output

