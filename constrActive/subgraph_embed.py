#!/usr/bin/env python
"""Provides wrapper for optimizer."""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import minmax_scale
from scipy.stats import rankdata
from toolz import curry
from collections import defaultdict
import random

import logging
logger = logging.getLogger(__name__)


# TODO: for each graph, consider the case same target a) or different b):
# the prob of an edge existing between two nodes in a) is proportional to
# how similar the two targets are
# if they are identical then the prob is 1, if not then the prob is computed
# on the empirical distribution of target values.sample_fraction.
# the probability is computed as 1-p in the case of b).
# The ratio is that instances with same class should cluster together and
# different classes should be separated.

def sample(
        data,
        sample_fraction=0.5,
        feature_fraction=0.5,
        sample_p=None,
        feature_p=None):
    """Sample a data matrix by row and col."""
    r, c = data.shape
    rows_sample_size = int(r * sample_fraction)
    row_ids = np.random.choice(
        range(r),
        size=rows_sample_size,
        p=sample_p,
        replace=False)
    row_ids = np.sort(row_ids)
    sample_row_data = data[row_ids, :]

    cols_sample_size = int(c * feature_fraction)
    cols_sample_size = max(2, cols_sample_size)
    col_ids = np.random.choice(
        range(c),
        size=cols_sample_size,
        p=feature_p,
        replace=False)
    col_ids = np.sort(col_ids)
    sample_col_data = sample_row_data[:, col_ids]
    return sample_col_data, row_ids, col_ids


def step_func(x, alpha=0, gamma=1, beta=30):
    """Step function."""
    return min(beta, x * gamma + alpha)


def update(count_dict, lengths_dict, graph):
    """Update edge weights."""
    for i, j in graph.edges():
        count_dict[i][j] += 1
        lengths_dict[i][j].append(graph.edge[i][j]['len'])


def make_knn_graph(count_dict, lengths_dict,
                   size=1, target=None, confidence=1.0):
    """Make a graph given the count matrix."""
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    for i in count_dict:
        neighbors = [j for j in count_dict[i]]
        if neighbors is not None:
            w_norm = max([count_dict[i][j] for j in neighbors])
            for j in neighbors:
                length = random.choice(lengths_dict[i][j])
                if target is not None and target[i] == target[j]:
                    effective_confidence = 1
                else:
                    effective_confidence = confidence
                length = length * effective_confidence
                w = count_dict[i][j]
                if random.random() < w / float(w_norm):
                    graph.add_edge(i, j, len=length, weight=w)
    return graph


def _graph_layout(graph):
    pos = nx.graphviz_layout(graph, prog='neato')
    # pos = nx.spring_layout(graph, weight='len', iterations=50, pos=pos)
    x2d = np.vstack([pos[u] for u in sorted(pos)])
    return x2d


def make_mst_tree(distance_mtx, ids):
    """make_mst_tree."""
    graph = nx.Graph()
    for u in ids:
        graph.add_node(u)
    for i, u in enumerate(ids):
        for j, v in enumerate(ids):
            graph.add_edge(u, v, weight=distance_mtx[i, j])
    graph = nx.minimum_spanning_tree(graph)
    return graph


def make_qks_tree(distance_mtx, ids):
    """make_qks_tree."""
    graph = nx.Graph()
    for u in ids:
        graph.add_node(u)
    # compute instance density as average pairwise similarity
    centrality = np.mean(distance_mtx, 1)
    knn_ids = np.argsort(distance_mtx, 1)
    r, c = distance_mtx.shape
    parents = []
    for i in range(r):
        parent = i
        for j in knn_ids[i]:
            if centrality[j] < centrality[i]:
                parent = j
                break
        parents.append(parent)
    for i, j in enumerate(parents):
        if i != j:
            u, v = ids[i], ids[j]
            graph.add_edge(u, v)
    return graph


def add_edge_length(graph, rank_mtx, sample_id_inv_map, func=None):
    """add_edge_length."""
    for u, v in graph.edges():
        i, j = sample_id_inv_map[u], sample_id_inv_map[v]
        nn_order = max(rank_mtx[i, j], rank_mtx[j, i])
        graph.edge[u][v]['len'] = func(nn_order)


def make_id_map(sample_ids):
    """make_id_map."""
    sample_id_inv_map = dict([(u, i) for i, u in enumerate(sample_ids)])
    return sample_id_inv_map


def compute_ranks(distance_mtx):
    """compute_ranks."""
    rank_mtx = np.vstack([rankdata(row, method='min')
                          for row in distance_mtx])
    return rank_mtx


def graph_embed(
        data,
        target=None,
        confidence=1,
        n_iter=20,
        sample_fraction=.7,
        sample_p=None,
        feature_fraction=1,
        feature_p=None,
        alpha=0,
        gamma=1,
        beta=30):
    """Provide 2D embedding of high dimensional data."""
    # set parameters in all functions
    if sample_p is not None:
        sample_p = sample_p / np.sum(sample_p)
        sample_p = sample_p + 0.01
        sample_p = sample_p / np.sum(sample_p)
    if feature_p is not None:
        feature_p = feature_p / np.sum(feature_p)
        feature_p = feature_p + 0.01
        feature_p = feature_p / np.sum(feature_p)
    _sample = curry(sample)(sample_fraction=sample_fraction,
                            sample_p=sample_p,
                            feature_fraction=feature_fraction,
                            feature_p=feature_p)
    _step_func = curry(step_func)(
        alpha=alpha,
        gamma=gamma,
        beta=beta)
    _add_edge_length = curry(add_edge_length)(func=_step_func)
    _make_knn_graph = curry(make_knn_graph)(size=len(data),
                                            target=target,
                                            confidence=confidence)

    def make_graph(make_tree):
        count_dict = defaultdict(lambda: defaultdict(int))
        lengths_dict = defaultdict(lambda: defaultdict(list))
        for i in range(n_iter):
            sample_data, sample_ids, sample_feature_ids = _sample(data)
            sample_id_inv_map = make_id_map(sample_ids)
            distance_mtx = euclidean_distances(sample_data)
            rank_mtx = compute_ranks(distance_mtx)
            tree = make_tree(distance_mtx, sample_ids)
            _add_edge_length(tree, rank_mtx, sample_id_inv_map)
            update(count_dict, lengths_dict, tree)
        graph = _make_knn_graph(count_dict, lengths_dict)
        return graph

    mst_graph = make_graph(make_mst_tree)
    qks_graph = make_graph(make_qks_tree)
    graph = nx.compose(mst_graph, qks_graph)
    return graph


def embed(
        data,
        target=None,
        confidence=1,
        n_iter=100,
        sample_fraction=.3,
        sample_p=None,
        feature_fraction=1,
        feature_p=None,
        alpha=0,
        gamma=1,
        beta=30):
    """Provide 2D embedding of high dimensional data."""
    graph = graph_embed(
        data,
        target,
        confidence,
        n_iter,
        sample_fraction,
        sample_p,
        feature_fraction,
        feature_p,
        alpha,
        gamma,
        beta)
    x2d = _graph_layout(graph)
    x2d = minmax_scale(x2d, feature_range=(0, 1))
    return x2d, graph


def centrality(data,
               n_iter=20,
               sample_fraction=.7,
               sample_p=None,
               feature_fraction=1,
               feature_p=None,
               len_quant=50,
               alpha=0,
               gamma=1,
               beta=30):
    """centrality."""
    graph = graph_embed(
        data,
        n_iter=n_iter,
        sample_fraction=sample_fraction,
        sample_p=sample_p,
        feature_fraction=feature_fraction,
        feature_p=feature_p,
        len_quant=len_quant,
        alpha=alpha,
        gamma=gamma,
        beta=beta)
    node_dict = nx.closeness_centrality(graph, distance='len', normalized=True)
    centralities = [node_dict[u] for u in sorted(node_dict)]
    return centralities
