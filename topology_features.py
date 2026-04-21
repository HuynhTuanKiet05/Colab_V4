import os

import networkx as nx
import numpy as np
import torch


def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _compute_graph_topology(adj_matrix):
    graph = nx.from_numpy_array(adj_matrix)
    node_count = adj_matrix.shape[0]

    degrees = np.array([graph.degree(i) for i in range(node_count)], dtype=np.float32)
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    degrees_norm = degrees / max_deg

    weighted_degrees = np.array([graph.degree(i, weight="weight") for i in range(node_count)], dtype=np.float32)
    max_weighted_deg = weighted_degrees.max() if weighted_degrees.max() > 0 else 1.0
    weighted_degrees_norm = weighted_degrees / max_weighted_deg

    clustering = nx.clustering(graph, weight="weight")
    clustering_arr = np.array([clustering[i] for i in range(node_count)], dtype=np.float32)

    try:
        pagerank = nx.pagerank(graph, weight="weight", max_iter=100)
        pagerank_arr = np.array([pagerank[i] for i in range(node_count)], dtype=np.float32)
    except nx.PowerIterationFailedConvergence:
        pagerank_arr = np.ones(node_count, dtype=np.float32) / max(node_count, 1)
    pagerank_norm = pagerank_arr / max(pagerank_arr.max(), 1.0)

    avg_neighbor_deg = nx.average_neighbor_degree(graph, weight="weight")
    avg_neighbor_arr = np.array([avg_neighbor_deg.get(i, 0.0) for i in range(node_count)], dtype=np.float32)
    avg_neighbor_norm = avg_neighbor_arr / max(avg_neighbor_arr.max(), 1.0)

    return np.stack(
        [degrees_norm, weighted_degrees_norm, clustering_arr, pagerank_norm, avg_neighbor_norm],
        axis=1,
    )


def _compute_association_degrees(associations, num_entities, entity_col):
    degrees = np.zeros(num_entities, dtype=np.float32)
    for row in associations:
        entity_idx = int(row[entity_col])
        if 0 <= entity_idx < num_entities:
            degrees[entity_idx] += 1.0
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    return degrees / max_deg


def extract_topology_features(data, args, force_recompute=False):
    cache_dir = os.path.join(args.data_dir, "topology_cache")
    os.makedirs(cache_dir, exist_ok=True)

    drug_cache = os.path.join(cache_dir, f"drug_topo_k{args.neighbor}_n{args.drug_number}.pt")
    disease_cache = os.path.join(cache_dir, f"disease_topo_k{args.neighbor}_n{args.disease_number}.pt")

    if not force_recompute and os.path.exists(drug_cache) and os.path.exists(disease_cache):
        drug_topo = _safe_torch_load(drug_cache)
        disease_topo = _safe_torch_load(disease_cache)
        if drug_topo.shape[0] == args.drug_number and disease_topo.shape[0] == args.disease_number:
            return drug_topo, disease_topo

    from data_preprocess_improved import k_matrix

    drug_knn = k_matrix(data["drs"], args.neighbor)
    disease_knn = k_matrix(data["dis"], args.neighbor)

    drug_graph_feats = _compute_graph_topology(drug_knn)
    disease_graph_feats = _compute_graph_topology(disease_knn)

    drug_disease_deg = _compute_association_degrees(data["drdi"], args.drug_number, entity_col=0)
    drug_protein_deg = _compute_association_degrees(data["drpr"], args.drug_number, entity_col=0)
    disease_drug_deg = _compute_association_degrees(data["drdi"], args.disease_number, entity_col=1)
    disease_protein_deg = _compute_association_degrees(data["dipr"], args.disease_number, entity_col=0)

    drug_topo = np.concatenate(
        [drug_graph_feats, drug_disease_deg[:, None], drug_protein_deg[:, None]],
        axis=1,
    )
    disease_topo = np.concatenate(
        [disease_graph_feats, disease_drug_deg[:, None], disease_protein_deg[:, None]],
        axis=1,
    )

    drug_topo = torch.tensor(drug_topo, dtype=torch.float32)
    disease_topo = torch.tensor(disease_topo, dtype=torch.float32)
    torch.save(drug_topo, drug_cache)
    torch.save(disease_topo, disease_cache)
    return drug_topo, disease_topo
