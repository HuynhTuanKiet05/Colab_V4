from __future__ import annotations

from typing import Any, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import csv
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'ductri_hgt_update', 'data', 'C-dataset')

app = FastAPI(title='Drug Disease AI API', version='1.0.0')


class PredictRequest(BaseModel):
    query_type: str = Field(pattern='^(drug_to_disease|disease_to_drug)$')
    input_text: str
    top_k: int = 10


class SearchItem(BaseModel):
    id: str
    name: str
    score: float
    type: str


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    color: str


class GraphLink(BaseModel):
    source: str
    target: str
    score: float


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_drugs() -> List[Dict[str, str]]:
    return load_csv_rows(os.path.join(DATASET_DIR, 'DrugInformation.csv'))


def load_proteins() -> List[Dict[str, str]]:
    return load_csv_rows(os.path.join(DATASET_DIR, 'ProteinInformation.csv'))


def load_diseases() -> List[Dict[str, str]]:
    rows = []
    disease_feature_path = os.path.join(DATASET_DIR, 'DiseaseFeature.csv')
    if not os.path.exists(disease_feature_path):
        return rows
    with open(disease_feature_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts and parts[0]:
                code = parts[0]
                rows.append({'id': code, 'name': code})
    return rows


def fuzzy_match(items: List[Dict[str, str]], text: str, name_key: str = 'name') -> List[Dict[str, str]]:
    text_lower = text.strip().lower()
    starts = []
    contains = []
    for item in items:
        value = (item.get(name_key) or item.get('id') or '').lower()
        if not value:
            continue
        if value.startswith(text_lower):
            starts.append(item)
        elif text_lower in value:
            contains.append(item)
    if starts:
        return starts
    if contains:
        return contains
    return items[:20]


def build_mock_results(query_type: str, input_text: str, top_k: int) -> Dict[str, Any]:
    drugs = load_drugs()
    diseases = load_diseases()
    proteins = load_proteins()

    rng = random.Random(input_text.lower())

    if query_type == 'drug_to_disease':
        source_items = fuzzy_match(drugs, input_text, 'name')
        source = source_items[0] if source_items else {'id': 'drug_unknown', 'name': input_text}
        candidates = diseases[:]
        rng.shuffle(candidates)
        results = []
        for idx, item in enumerate(candidates[:top_k], start=1):
            results.append({
                'id': item['id'],
                'name': item.get('name', item['id']),
                'score': round(max(0.5, 0.95 - idx * 0.03), 4),
                'type': 'disease'
            })

        graph_nodes = [
            {'id': f"drug:{source['id']}", 'label': source['name'], 'type': 'drug', 'color': '#2563eb'}
        ]
        graph_links = []

        chosen_proteins = proteins[: min(8, len(proteins))]
        for protein in chosen_proteins:
            node_id = f"protein:{protein['id']}"
            graph_nodes.append({'id': node_id, 'label': protein['id'], 'type': 'protein', 'color': '#f59e0b'})
            graph_links.append({'source': f"drug:{source['id']}", 'target': node_id, 'score': round(rng.uniform(0.55, 0.88), 4)})

        for result in results[: min(8, len(results))]:
            disease_node_id = f"disease:{result['id']}"
            graph_nodes.append({'id': disease_node_id, 'label': result['name'], 'type': 'disease', 'color': '#dc2626'})
            graph_links.append({'source': f"drug:{source['id']}", 'target': disease_node_id, 'score': result['score']})
            if chosen_proteins:
                protein = chosen_proteins[rng.randrange(len(chosen_proteins))]
                graph_links.append({'source': f"protein:{protein['id']}", 'target': disease_node_id, 'score': round(rng.uniform(0.52, 0.85), 4)})

        return {
            'matched_input': {'id': source['id'], 'name': source['name'], 'type': 'drug'},
            'results': results,
            'graph': {'nodes': graph_nodes, 'links': graph_links},
            'note': 'Đây là API demo tích hợp dữ liệu sẵn có. Có thể thay bằng suy luận model thật sau.'
        }

    source_items = fuzzy_match(diseases, input_text, 'name')
    source = source_items[0] if source_items else {'id': 'disease_unknown', 'name': input_text}
    candidates = drugs[:]
    rng.shuffle(candidates)
    results = []
    for idx, item in enumerate(candidates[:top_k], start=1):
        results.append({
            'id': item['id'],
            'name': item.get('name', item['id']),
            'score': round(max(0.5, 0.95 - idx * 0.03), 4),
            'type': 'drug'
        })

    graph_nodes = [
        {'id': f"disease:{source['id']}", 'label': source['name'], 'type': 'disease', 'color': '#dc2626'}
    ]
    graph_links = []

    chosen_proteins = proteins[: min(8, len(proteins))]
    for protein in chosen_proteins:
        node_id = f"protein:{protein['id']}"
        graph_nodes.append({'id': node_id, 'label': protein['id'], 'type': 'protein', 'color': '#f59e0b'})
        graph_links.append({'source': f"disease:{source['id']}", 'target': node_id, 'score': round(rng.uniform(0.55, 0.88), 4)})

    for result in results[: min(8, len(results))]:
        drug_node_id = f"drug:{result['id']}"
        graph_nodes.append({'id': drug_node_id, 'label': result['name'], 'type': 'drug', 'color': '#2563eb'})
        graph_links.append({'source': f"disease:{source['id']}", 'target': drug_node_id, 'score': result['score']})
        if chosen_proteins:
            protein = chosen_proteins[rng.randrange(len(chosen_proteins))]
            graph_links.append({'source': f"protein:{protein['id']}", 'target': drug_node_id, 'score': round(rng.uniform(0.52, 0.85), 4)})

    return {
        'matched_input': {'id': source['id'], 'name': source['name'], 'type': 'disease'},
        'results': results,
        'graph': {'nodes': graph_nodes, 'links': graph_links},
        'note': 'Đây là API demo tích hợp dữ liệu sẵn có. Có thể thay bằng suy luận model thật sau.'
    }


@app.get('/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: PredictRequest) -> Dict[str, Any]:
    top_k = max(1, min(payload.top_k, 20))
    return build_mock_results(payload.query_type, payload.input_text, top_k)
