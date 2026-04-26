"""DropEdge (Rong et al. 2020) for DGL similarity / heterograph subgraphs.

DropEdge randomly removes a fraction `p` of edges from a graph at every
training step. On dense kNN similarity graphs (drug-drug, disease-disease)
this acts like a strong regularizer: the model can no longer rely on any
single similarity edge and is forced to learn distributed representations.

The implementation here:
- supports both single homogeneous DGLGraph and dict-of-views (multi-view
  similarity), to match the two `similarity_view_mode` options in the model;
- never drops edges at eval time;
- is a pure subgraph operation (no edge feature copy needed).

Cost: O(num_edges) on CPU per epoch — negligible vs. forward time.
"""
from __future__ import annotations

from typing import Mapping, Union

import dgl
import torch


GraphInput = Union[dgl.DGLGraph, Mapping[str, dgl.DGLGraph]]


def _drop_one(graph: dgl.DGLGraph, p: float, generator: torch.Generator) -> dgl.DGLGraph:
    if p <= 0.0:
        return graph
    num_edges = graph.num_edges()
    if num_edges == 0:
        return graph
    keep_mask = torch.rand(num_edges, generator=generator, device=torch.device("cpu")) >= p
    keep_eids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).to(graph.device)
    sub = dgl.edge_subgraph(graph, keep_eids, relabel_nodes=False)
    return sub


def dropedge_graph(
    graph: GraphInput,
    p: float,
    generator: torch.Generator,
) -> GraphInput:
    """Return a copy of `graph` with a fraction `p` of edges randomly dropped.

    Args:
        graph: a single DGLGraph, or a dict mapping view-name -> DGLGraph.
        p: probability of dropping each edge. ``0.0`` returns the graph
            unchanged; ``1.0`` would drop every edge (avoid in practice).
        generator: a CPU `torch.Generator` to make the drop reproducible
            across folds.

    Returns:
        Same container type as the input (DGLGraph or dict), with edges
        sub-sampled. Node ids are preserved (`relabel_nodes=False`).
    """
    if p <= 0.0:
        return graph
    if isinstance(graph, dict):
        return {name: _drop_one(g, p, generator) for name, g in graph.items()}
    return _drop_one(graph, p, generator)
