from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from model.AMNTDDA import AMNTDDA
from model.improved.improved_model import AMNTDDA as ImprovedAMNTDDA
from AMDGT_original.model.AMNTDDA import AMNTDDA as OriginalAMNTDDA
from AMDGT_original import data_preprocess as original_preprocess
import data_preprocess_improved as improved_preprocess
from topology_features import extract_topology_features

if os.environ.get('HGT_MODEL_VERSION', 'improved') == 'improved':
    from data_preprocess_improved import dgl_similarity_graph, dgl_heterograph
else:
    from AMDGT_original.data_preprocess import dgl_similarity_graph, dgl_heterograph

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='HGT Drug-Disease Prediction API', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def _global_exc_handler(request: Request, exc: Exception):
    tb = _traceback.format_exc()
    print(f"[ERROR] Unhandled on {request.url.path}:\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})


device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")


class PredictRequest(BaseModel):
    query_type: str = Field(pattern='^(drug_to_disease|disease_to_drug)$')
    input_text: str
    top_k: int = 10
    dataset: str = 'C-dataset'


class ComparePredictRequest(BaseModel):
    dataset: str = 'C-dataset'
    drugs: List[str] = Field(default_factory=list, max_length=5)
    diseases: List[str] = Field(default_factory=list, max_length=5)
    top_k: int = Field(default=5, ge=1, le=20)


DATASET_PRESETS = {
    'B-dataset': {
        'neighbor': 3,
        'gt_out_dim': 512,
        'hgt_in_dim': 512,
        'hgt_layer': 2,
        'hgt_head': 8,
        'hgt_head_dim': 64,
        'topo_hidden': 192,
    },
    'C-dataset': {
        'neighbor': 5,
        'gt_out_dim': 256,
        'hgt_in_dim': 256,
        'hgt_layer': 2,
        'hgt_head': 8,
        'hgt_head_dim': 32,
        'topo_hidden': 128,
    },
    'F-dataset': {
        'neighbor': 5,
        'gt_out_dim': 256,
        'hgt_in_dim': 256,
        'hgt_layer': 2,
        'hgt_head': 8,
        'hgt_head_dim': 32,
        'topo_hidden': 128,
    },
}


class InferenceManager:
    def __init__(self):
        self.cached_models: Dict[str, torch.nn.Module] = {}
        self.cached_data: Dict[str, Dict[str, object]] = {}

    def get_dataset_paths(self, dataset_name: str):
        data_dir = PROJECT_ROOT / 'AMDGT_original' / 'data' / dataset_name
        model_path = self.resolve_model_path(dataset_name)
        return data_dir, model_path

    def resolve_model_path(self, dataset_name: str) -> Path | None:
        dataset_variants = {dataset_name, dataset_name.replace('-', '_'), dataset_name.replace('_', '-')}
        candidate_roots = []
        for variant in dataset_variants:
            candidate_roots.extend([
                PROJECT_ROOT / 'Result' / 'improved' / variant,
                PROJECT_ROOT / 'Result' / variant,
            ])

        candidates: List[Path] = []
        for root in candidate_roots:
            if not root.exists():
                continue
            candidates.extend(root.rglob('best_model*.pth'))

        if not candidates:
            legacy_path = PROJECT_ROOT / 'Result' / dataset_name / 'AMNTDDA' / 'best_model.pth'
            return legacy_path if legacy_path.exists() else None

        candidates.sort(key=lambda path: ('fold' in path.stem.lower(), -path.stat().st_mtime))
        return candidates[0]

    def resolve_checkpoint_path(self, dataset_name: str, model_version: str) -> Path | None:
        if model_version not in {'original', 'improved'}:
            raise ValueError(f"Unsupported model version: {model_version}")

        dataset_variants = {dataset_name, dataset_name.replace('-', '_'), dataset_name.replace('_', '-')}
        candidate_roots = []
        for variant in dataset_variants:
            if model_version == 'original':
                candidate_roots.extend([
                    PROJECT_ROOT / 'Result' / 'original' / variant,
                    PROJECT_ROOT / 'Result' / variant / 'AMNTDDA',
                    PROJECT_ROOT / 'AMDGT_original' / 'Result' / variant / 'AMNTDDA',
                    PROJECT_ROOT / 'Result' / variant / 'RLGHGT_v2',
                ])
            else:
                candidate_roots.extend([
                    PROJECT_ROOT / 'Result' / 'improved' / variant,
                    PROJECT_ROOT / 'Result' / variant / 'improved',
                    PROJECT_ROOT / 'Result' / variant / 'AMNTDDA_improved',
                ])

        candidates: List[Path] = []
        for root in candidate_roots:
            if root.exists():
                candidates.extend(root.rglob('best_model*.pth'))
                candidates.extend(root.rglob('*.pth'))

        seen: Dict[str, Path] = {}
        for path in candidates:
            seen[str(path.resolve())] = path
        candidates = list(seen.values())
        if not candidates:
            return None

        def sort_key(path: Path):
            stem = path.stem.lower()
            fold_penalty = 1 if 'fold' in stem else 0
            return fold_penalty, -path.stat().st_mtime

        candidates.sort(key=sort_key)
        return candidates[0]

    @staticmethod
    def load_csv(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open('r', encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def load_node_ids(data_dir: Path) -> List[str]:
        for filename in ('AllNode.csv', 'Allnode.csv'):
            path = data_dir / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path, header=None)
            values = frame.iloc[:, -1].astype(str).str.strip()
            return [value for value in values if value and value.lower() not in {'id', 'nan'}]
        return []

    def load_drug_info(self, data_dir: Path, fallback_count: int) -> List[Dict[str, str]]:
        rows = self.load_csv(data_dir / 'DrugInformation.csv')
        if rows:
            return [
                {
                    'id': str(row.get('id') or row.get('drug_id') or index),
                    'name': str(row.get('name') or row.get('drug_name') or row.get('id') or index),
                    'smiles': str(row.get('smiles') or ''),
                }
                for index, row in enumerate(rows)
            ]
        return [{'id': f'DRUG_{index}', 'name': f'DRUG_{index}'} for index in range(fallback_count)]

    def load_disease_info(self, data_dir: Path, drug_count: int, disease_count: int) -> List[Dict[str, str]]:
        rows = self.load_csv(data_dir / 'DiseaseInformation.csv')
        if rows:
            return [
                {
                    'id': str(row.get('id') or row.get('disease_id') or index),
                    'name': str(row.get('name') or row.get('disease_name') or row.get('id') or index),
                }
                for index, row in enumerate(rows)
            ]

        node_ids = self.load_node_ids(data_dir)
        disease_ids = node_ids[drug_count:drug_count + disease_count]
        if len(disease_ids) < disease_count:
            disease_ids.extend(f'DISEASE_{index}' for index in range(len(disease_ids), disease_count))
        return [{'id': disease_id, 'name': disease_id} for disease_id in disease_ids[:disease_count]]

    def load_protein_info(self, data_dir: Path, protein_count: int) -> List[Dict[str, str]]:
        rows = self.load_csv(data_dir / 'ProteinInformation.csv')
        if rows:
            return [
                {
                    'id': str(row.get('id') or row.get('protein_id') or index),
                    'name': str(row.get('protein_name') or row.get('name') or row.get('id') or index),
                    'protein_name': str(row.get('protein_name') or row.get('name') or row.get('id') or index),
                    'sequence': str(row.get('sequence') or ''),
                }
                for index, row in enumerate(rows)
            ]
        return [{'id': f'PROTEIN_{index}', 'name': f'PROTEIN_{index}', 'protein_name': f'PROTEIN_{index}'} for index in range(protein_count)]

    @staticmethod
    def move_tensor_dict(values):
        if values is None:
            return None
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }

    @staticmethod
    def load_checkpoint(path: Path):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    @staticmethod
    def build_args(dataset_name: str, data_dir: Path, model_version: str = 'improved'):
        if model_version == 'original':
            preset = {
                'neighbor': 20,
                'gt_out_dim': 200,
                'hgt_in_dim': 64,
                'hgt_layer': 2,
                'hgt_head': 8,
                'hgt_head_dim': 25,
                'topo_hidden': 128,
            }
        else:
            preset = DATASET_PRESETS.get(dataset_name, DATASET_PRESETS['C-dataset'])

        class MockArgs:
            def __init__(self, dataset: str, root: Path, cfg: Dict[str, int]):
                self.dataset = dataset
                self.data_dir = str(root)
                self.neighbor = cfg['neighbor']
                self.hgt_layer = cfg['hgt_layer']
                self.hgt_head = cfg['hgt_head']
                self.hgt_head_dim = cfg['hgt_head_dim']
                self.hgt_in_dim = cfg['hgt_in_dim']
                self.hgt_out_dim = cfg['gt_out_dim']
                self.gt_layer = 2
                self.gt_head = cfg.get('gt_head', 2) # Cả hai bản đều dùng 2
                self.gt_out_dim = cfg['gt_out_dim']
                # Cả hai bản đều dùng 2 layers transformer
                self.tr_layer = cfg.get('tr_layer', 2)
                self.tr_head = cfg.get('tr_head', 4)
                self.dropout = 0.2
                self.topo_hidden = cfg.get('topo_hidden', 128)
                self.assoc_backbone = 'vanilla_hgt'
                self.fusion_mode = 'mva'
                self.pair_mode = 'mul_mlp'
                self.gate_mode = 'vector'
                self.gate_bias_init = -2.0
                self.temperature = 0.5
                self.topo_hidden = cfg['topo_hidden']
                self.topo_feat_dim = 7
                self.device = str(device)

        return MockArgs(dataset_name, data_dir, preset)

    def get_related_proteins(self, ctx: Dict[str, object], source_type: str, source_idx: int, limit: int = 5) -> List[Dict[str, str]]:
        protein_indices: List[int] = []
        if source_type == 'drug':
            for row in ctx['drpr_edges']:
                if int(row[0]) == source_idx:
                    protein_indices.append(int(row[1]))
        else:
            for row in ctx['dipr_edges']:
                if int(row[0]) == source_idx:
                    protein_indices.append(int(row[1]))

        seen = set()
        related = []
        for protein_idx in protein_indices:
            if protein_idx in seen or protein_idx >= len(ctx['protein_info']):
                continue
            seen.add(protein_idx)
            related.append(ctx['protein_info'][protein_idx])
            if len(related) >= limit:
                break
        return related

    def load_context(self, dataset_name: str):
        if dataset_name in self.cached_data:
            return self.cached_data[dataset_name]

        data_dir, model_path = self.get_dataset_paths(dataset_name)
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_name}")

        args = self.build_args(dataset_name, data_dir)

        drf = pd.read_csv(data_dir / 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
        drg = pd.read_csv(data_dir / 'DrugGIP.csv').iloc[:, 1:].to_numpy()
        dip = pd.read_csv(data_dir / 'DiseasePS.csv').iloc[:, 1:].to_numpy()
        dig = pd.read_csv(data_dir / 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

        drs = np.where(drf == 0, drg, (drf + drg) / 2)
        dis = np.where(dip == 0, dig, (dip + dig) / 2)

        data = {
            'drs': drs,
            'dis': dis,
            'drdi': pd.read_csv(data_dir / 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy(),
            'drpr': pd.read_csv(data_dir / 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy(),
            'dipr': pd.read_csv(data_dir / 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy(),
            'drugfeature': pd.read_csv(data_dir / 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy(),
            'diseasefeature': pd.read_csv(data_dir / 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy(),
            'proteinfeature': pd.read_csv(data_dir / 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy(),
        }

        args.drug_number = data['drugfeature'].shape[0]
        args.disease_number = data['diseasefeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]

        drugs_info = self.load_drug_info(data_dir, args.drug_number)
        disease_info = self.load_disease_info(data_dir, args.drug_number, args.disease_number)
        protein_info = self.load_protein_info(data_dir, args.protein_number)

        drdr_graph, didi_graph, _ = dgl_similarity_graph(data, args)
        drug_topo_feat, disease_topo_feat = extract_topology_features(data, args)

        empty_drdi = np.empty((0, 2), dtype=int)
        heterograph_result = dgl_heterograph(data, empty_drdi, args)
        drdipr_graph = heterograph_result[0]
        data = heterograph_result[1]
        edge_stats = heterograph_result[2] if len(heterograph_result) > 2 else None

        drdr_graph = drdr_graph.to(device)
        didi_graph = didi_graph.to(device)
        drdipr_graph = drdipr_graph.to(device)
        drug_feature = torch.tensor(data['drugfeature'], dtype=torch.float32).to(device)
        disease_feature = torch.tensor(data['diseasefeature'], dtype=torch.float32).to(device)
        protein_feature = torch.tensor(data['proteinfeature'], dtype=torch.float32).to(device)
        drug_topo_feat = drug_topo_feat.to(device)
        disease_topo_feat = disease_topo_feat.to(device)

        context = {
            'args': args,
            'model_path': model_path,
            'drdr_graph': drdr_graph,
            'didi_graph': didi_graph,
            'drdipr_graph': drdipr_graph,
            'drug_feature': drug_feature,
            'disease_feature': disease_feature,
            'protein_feature': protein_feature,
            'drug_topo_feat': drug_topo_feat,
            'disease_topo_feat': disease_topo_feat,
            'edge_stats': self.move_tensor_dict(edge_stats),
            'drugs_info': drugs_info,
            'disease_info': disease_info,
            'protein_info': protein_info,
            'drpr_edges': data['drpr'],
            'dipr_edges': data['dipr'],
        }
        self.cached_data[dataset_name] = context
        return context

    def get_model(self, dataset_name: str):
        if dataset_name in self.cached_models:
            return self.cached_models[dataset_name]

        _, model_path = self.get_dataset_paths(dataset_name)
        if model_path is None or not model_path.exists():
            return None

        ctx = self.load_context(dataset_name)
        model = AMNTDDA(ctx['args']).to(device)

        try:
            checkpoint = self.load_checkpoint(model_path)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            self.cached_models[dataset_name] = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_context_for_version(self, dataset_name: str, model_version: str):
        cache_key = f"{model_version}:{dataset_name}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]

        data_dir = PROJECT_ROOT / 'AMDGT_original' / 'data' / dataset_name
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_name}")

        args = self.build_args(dataset_name, data_dir, model_version)
        preprocess = improved_preprocess if model_version == 'improved' else original_preprocess

        drf = pd.read_csv(data_dir / 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
        drg = pd.read_csv(data_dir / 'DrugGIP.csv').iloc[:, 1:].to_numpy()
        dip = pd.read_csv(data_dir / 'DiseasePS.csv').iloc[:, 1:].to_numpy()
        dig = pd.read_csv(data_dir / 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

        drs = np.where(drf == 0, drg, (drf + drg) / 2)
        dis = np.where(dip == 0, dig, (dip + dig) / 2)

        data = {
            'drs': drs,
            'dis': dis,
            'drdi': pd.read_csv(data_dir / 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy(),
            'drpr': pd.read_csv(data_dir / 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy(),
            'dipr': pd.read_csv(data_dir / 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy(),
            'drugfeature': pd.read_csv(data_dir / 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy(),
            'diseasefeature': pd.read_csv(data_dir / 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy(),
            'proteinfeature': pd.read_csv(data_dir / 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy(),
        }

        args.drug_number = data['drugfeature'].shape[0]
        args.disease_number = data['diseasefeature'].shape[0]
        args.protein_number = data['proteinfeature'].shape[0]

        drugs_info = self.load_drug_info(data_dir, args.drug_number)
        disease_info = self.load_disease_info(data_dir, args.drug_number, args.disease_number)
        protein_info = self.load_protein_info(data_dir, args.protein_number)

        drdr_graph, didi_graph, _ = preprocess.dgl_similarity_graph(data, args)
        heterograph_result = preprocess.dgl_heterograph(data, np.empty((0, 2), dtype=int), args)
        drdipr_graph = heterograph_result[0]
        data = heterograph_result[1]
        edge_stats = heterograph_result[2] if len(heterograph_result) > 2 else None

        drug_topo_feat, disease_topo_feat = extract_topology_features(data, args)

        context = {
            'args': args,
            'drdr_graph': drdr_graph.to(device),
            'didi_graph': didi_graph.to(device),
            'drdipr_graph': drdipr_graph.to(device),
            'drug_feature': torch.tensor(data['drugfeature'], dtype=torch.float32).to(device),
            'disease_feature': torch.tensor(data['diseasefeature'], dtype=torch.float32).to(device),
            'protein_feature': torch.tensor(data['proteinfeature'], dtype=torch.float32).to(device),
            'drug_topo_feat': drug_topo_feat.to(device),
            'disease_topo_feat': disease_topo_feat.to(device),
            'edge_stats': self.move_tensor_dict(edge_stats),
            'drugs_info': drugs_info,
            'disease_info': disease_info,
            'protein_info': protein_info,
        }
        self.cached_data[cache_key] = context
        return context

    def get_model_for_version(self, dataset_name: str, model_version: str):
        checkpoint_path = self.resolve_checkpoint_path(dataset_name, model_version)
        if checkpoint_path is None or not checkpoint_path.exists():
            message = 'Không tìm thấy checkpoint model gốc' if model_version == 'original' else 'Không tìm thấy checkpoint model cải tiến'
            raise HTTPException(status_code=404, detail=f"{message} cho {dataset_name}.")

        cache_key = f"{model_version}:{dataset_name}:{checkpoint_path.resolve()}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key], checkpoint_path

        ctx = self.load_context_for_version(dataset_name, model_version)
        model_cls = OriginalAMNTDDA if model_version == 'original' else ImprovedAMNTDDA
        model = model_cls(ctx['args']).to(device)

        try:
            checkpoint = self.load_checkpoint(checkpoint_path)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                stripped = {
                    key.replace('module.', '', 1) if key.startswith('module.') else key: value
                    for key, value in state_dict.items()
                }
                model.load_state_dict(stripped)
            model.eval()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Không load được checkpoint {model_version}: {checkpoint_path}. Lỗi: {exc}")

        print(f"[compare_predict] Loaded {model_version} checkpoint: {checkpoint_path}")
        self.cached_models[cache_key] = model
        return model, checkpoint_path

    def score_pairs(self, dataset_name: str, model_version: str, pair_indices: List[Tuple[int, int]]) -> np.ndarray:
        model, _ = self.get_model_for_version(dataset_name, model_version)
        ctx = self.load_context_for_version(dataset_name, model_version)
        x_pairs = torch.tensor(pair_indices, dtype=torch.long, device=device)

        with torch.no_grad():
            if model_version == 'improved':
                _, scores = model(
                    ctx['drdr_graph'], ctx['didi_graph'], ctx['drdipr_graph'],
                    ctx['drug_feature'], ctx['disease_feature'], ctx['protein_feature'],
                    x_pairs,
                    drug_topo_feat=ctx['drug_topo_feat'],
                    disease_topo_feat=ctx['disease_topo_feat'],
                    edge_stats=ctx['edge_stats'],
                )
            else:
                _, scores = model(
                    ctx['drdr_graph'], ctx['didi_graph'], ctx['drdipr_graph'],
                    ctx['drug_feature'], ctx['disease_feature'], ctx['protein_feature'],
                    x_pairs,
                )

        return fn.softmax(scores, dim=-1)[:, 1].cpu().numpy()

manager = InferenceManager()


def fuzzy_match(items: List[Dict[str, str]], text: str, name_key: str = 'name'):
    text_lower = text.strip().lower()
    for item in items:
        if (item.get(name_key) or '').lower() == text_lower or item.get('id', '').lower() == text_lower:
            return item
    for item in items:
        if text_lower in (item.get(name_key) or '').lower() or text_lower in item.get('id', '').lower():
            return item
    return None


def normalize_requested_entities(values: List[str], label: str) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for raw_value in values:
        for part in str(raw_value).replace('\n', ',').split(','):
            value = part.strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            cleaned.append(value)
            seen.add(key)

    if not cleaned:
        raise HTTPException(status_code=422, detail=f"Vui lòng nhập ít nhất 1 {label}.")
    if len(cleaned) > 5:
        raise HTTPException(status_code=422, detail=f"Chỉ được nhập tối đa 5 {label}.")
    return cleaned


def resolve_requested_items(items: List[Dict[str, str]], values: List[str], label: str):
    resolved = []
    used_ids = set()
    for value in values:
        match = fuzzy_match(items, value, 'name')
        if not match:
            raise HTTPException(status_code=422, detail=f"Không tìm thấy {label}: {value}")
        if match['id'] in used_ids:
            continue
        resolved.append(match)
        used_ids.add(match['id'])
    return resolved


def top_model_results(rows: List[Dict[str, object]], score_key: str, top_k: int) -> List[Dict[str, object]]:
    ranked = sorted(rows, key=lambda item: float(item[score_key]), reverse=True)[:top_k]
    return [
        {
            'drug_id': row['drug_id'],
            'drug_name': row['drug_name'],
            'disease_id': row['disease_id'],
            'disease_name': row['disease_name'],
            'score': row[score_key],
        }
        for row in ranked
    ]


@app.post('/predict')
async def predict(payload: PredictRequest):
    dataset = payload.dataset
    ctx = manager.load_context(dataset)
    model = manager.get_model(dataset)
    is_real = model is not None

    if is_real and ctx['model_path'] is not None:
        note = f"Đang sử dụng checkpoint {Path(ctx['model_path']).name} trên {dataset}."
    else:
        note = f"Đang chạy Demo cho {dataset} (chưa tìm thấy checkpoint phù hợp)."

    source = None

    if payload.query_type == 'drug_to_disease':
        source = fuzzy_match(ctx['drugs_info'], payload.input_text, 'name')
        if not source:
            alt_source = fuzzy_match(ctx['disease_info'], payload.input_text, 'name')
            if alt_source:
                note = f"⚠️ Thông báo: '{payload.input_text}' là một Bệnh."
            else:
                note = f"ℹ️ Không tìm thấy thực thể '{payload.input_text}' trong hệ thống dữ liệu."
            return {'status': 'success', 'results': [], 'note': note, 'graph': {'nodes': [], 'links': []}}
        source_idx = ctx['drugs_info'].index(source)
        num_diseases = len(ctx['disease_info'])
        x_infer = torch.tensor([[source_idx, i] for i in range(num_diseases)], dtype=torch.long, device=device)
        target_info = ctx['disease_info']
        target_type = 'disease'
    else:
        source = fuzzy_match(ctx['disease_info'], payload.input_text, 'name')
        if not source:
            alt_source = fuzzy_match(ctx['drugs_info'], payload.input_text, 'name')
            if alt_source:
                note = f"⚠️ Thông báo: '{payload.input_text}' là một Thuốc."
            else:
                note = f"ℹ️ Không tìm thấy thực thể '{payload.input_text}' trong hệ thống dữ liệu."
            return {'status': 'success', 'results': [], 'note': note, 'graph': {'nodes': [], 'links': []}}

        source_idx = ctx['disease_info'].index(source)
        num_drugs = len(ctx['drugs_info'])
        x_infer = torch.tensor([[i, source_idx] for i in range(num_drugs)], dtype=torch.long, device=device)
        target_info = ctx['drugs_info']
        target_type = 'drug'

    if is_real:
        with torch.no_grad():
            _, scores = model(
                ctx['drdr_graph'], ctx['didi_graph'], ctx['drdipr_graph'],
                ctx['drug_feature'], ctx['disease_feature'], ctx['protein_feature'],
                x_infer,
                drug_topo_feat=ctx['drug_topo_feat'],
                disease_topo_feat=ctx['disease_topo_feat'],
                edge_stats=ctx['edge_stats'],
            )
            probs = fn.softmax(scores, dim=-1)[:, 1].cpu().numpy()
    else:
        probs = np.random.uniform(0.1, 0.9, len(target_info))

    results = []
    indices = np.argsort(-probs)[:payload.top_k]
    for idx in indices:
        item = target_info[idx]
        results.append({
            'id': item['id'],
            'name': item.get('name', item['id']),
            'score': float(round(probs[idx], 4)),
            'type': target_type
        })

    graph_nodes = [{'id': f"{payload.query_type.split('_')[0]}:{source['id']}", 'actual_id': source['id'], 'label': source.get('name', source['id']), 'type': payload.query_type.split('_')[0], 'color': '#2563eb' if payload.query_type.startswith('drug') else '#dc2626'}]
    graph_links = []

    for p in manager.get_related_proteins(ctx, payload.query_type.split('_')[0], source_idx, limit=5):
        p_id = f"protein:{p['id']}"
        label = p.get('protein_name', p.get('name', p['id']))
        graph_nodes.append({'id': p_id, 'actual_id': p['id'], 'label': label, 'type': 'protein', 'color': '#f59e0b'})
        graph_links.append({'source': graph_nodes[0]['id'], 'target': p_id, 'score': 0.8})

    for res in results[:5]:
        res_id = f"{res['type']}:{res['id']}"
        graph_nodes.append({'id': res_id, 'actual_id': res['id'], 'label': res['name'], 'type': res['type'], 'color': '#dc2626' if res['type'] == 'disease' else '#2563eb'})
        graph_links.append({'source': graph_nodes[0]['id'], 'target': res_id, 'score': res['score']})

    return {
        'status': 'success',
        'matched_input': {'id': source['id'], 'name': source.get('name', source['id']), 'type': payload.query_type.split('_')[0]},
        'results': results,
        'graph': {'nodes': graph_nodes, 'links': graph_links},
        'note': note,
    }


@app.post('/compare_predict')
async def compare_predict(payload: ComparePredictRequest):
    dataset = payload.dataset
    if dataset not in DATASET_PRESETS:
        raise HTTPException(status_code=422, detail=f"Dataset không hợp lệ: {dataset}")

    requested_drugs = normalize_requested_entities(payload.drugs, 'thuốc')
    requested_diseases = normalize_requested_entities(payload.diseases, 'bệnh')

    original_checkpoint = manager.resolve_checkpoint_path(dataset, 'original')
    if original_checkpoint is None or not original_checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy checkpoint model gốc cho {dataset}.")

    improved_checkpoint = manager.resolve_checkpoint_path(dataset, 'improved')
    if improved_checkpoint is None or not improved_checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy checkpoint model cải tiến cho {dataset}.")

    ctx = manager.load_context_for_version(dataset, 'improved')
    selected_drugs = resolve_requested_items(ctx['drugs_info'], requested_drugs, 'thuốc')
    selected_diseases = resolve_requested_items(ctx['disease_info'], requested_diseases, 'bệnh')

    drug_indices = [ctx['drugs_info'].index(item) for item in selected_drugs]
    disease_indices = [ctx['disease_info'].index(item) for item in selected_diseases]
    pair_indices = [(drug_idx, disease_idx) for drug_idx in drug_indices for disease_idx in disease_indices]

    original_scores = manager.score_pairs(dataset, 'original', pair_indices)
    improved_scores = manager.score_pairs(dataset, 'improved', pair_indices)

    comparison: List[Dict[str, object]] = []
    pair_offset = 0
    for drug in selected_drugs:
        for disease in selected_diseases:
            original_score = float(round(float(original_scores[pair_offset]), 4))
            improved_score = float(round(float(improved_scores[pair_offset]), 4))
            delta = float(round(improved_score - original_score, 4))
            if delta > 0:
                winner = 'improved'
            elif delta < 0:
                winner = 'original'
            else:
                winner = 'tie'

            comparison.append({
                'drug_id': drug['id'],
                'drug_name': drug.get('name', drug['id']),
                'disease_id': disease['id'],
                'disease_name': disease.get('name', disease['id']),
                'original_score': original_score,
                'improved_score': improved_score,
                'delta': delta,
                'winner': winner,
            })
            pair_offset += 1

    comparison.sort(key=lambda item: float(item['improved_score']), reverse=True)
    graph_links = []
    for row in comparison[:min(max(payload.top_k, 1), 20)]:
        graph_links.append({
            'source': f"drug:{row['drug_id']}",
            'target': f"disease:{row['disease_id']}",
            'original_score': row['original_score'],
            'improved_score': row['improved_score'],
            'delta': row['delta'],
            'winner': row['winner'],
        })

    graph_nodes = [
        {
            'id': f"drug:{drug['id']}",
            'actual_id': drug['id'],
            'label': drug.get('name', drug['id']),
            'type': 'drug',
        }
        for drug in selected_drugs
    ] + [
        {
            'id': f"disease:{disease['id']}",
            'actual_id': disease['id'],
            'label': disease.get('name', disease['id']),
            'type': 'disease',
        }
        for disease in selected_diseases
    ]

    return {
        'status': 'success',
        'dataset': dataset,
        'input': {
            'drugs': [{'id': item['id'], 'name': item.get('name', item['id'])} for item in selected_drugs],
            'diseases': [{'id': item['id'], 'name': item.get('name', item['id'])} for item in selected_diseases],
            'top_k': payload.top_k,
        },
        'models': {
            'original': {
                'checkpoint': str(original_checkpoint),
                'results': top_model_results(comparison, 'original_score', payload.top_k),
            },
            'improved': {
                'checkpoint': str(improved_checkpoint),
                'results': top_model_results(comparison, 'improved_score', payload.top_k),
            },
        },
        'comparison': comparison,
        'graph2d': {
            'nodes': graph_nodes,
            'links': graph_links,
        },
    }


@app.get('/entities')
def list_entities(dataset: str = 'C-dataset'):
    if dataset not in DATASET_PRESETS:
        raise HTTPException(status_code=422, detail=f"Dataset không hợp lệ: {dataset}")
    data_dir = PROJECT_ROOT / 'AMDGT_original' / 'data' / dataset
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
    drugs = manager.load_drug_info(data_dir, 0)
    try:
        dc = pd.read_csv(data_dir / 'DiseaseFeature.csv', header=None).shape[0]
    except Exception:
        dc = 0
    diseases = manager.load_disease_info(data_dir, len(drugs), dc)
    return {
        'dataset': dataset,
        'drugs': [{'id': d['id'], 'name': d.get('name', d['id'])} for d in drugs],
        'diseases': [{'id': d['id'], 'name': d.get('name', d['id'])} for d in diseases],
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'device': str(device)}
