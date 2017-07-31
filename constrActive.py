#!/usr/bin/env python
"""Provides sampling of graphs."""

from toolz.curried import pipe, concat
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async
from pareto_graph_optimizer import optimize_desired_distances
from pareto_graph_optimizer import GrammarWrapper
from pareto_graph_optimizer import MultiObjectiveCostEstimator
from pareto_graph_optimizer import SimVolPredStdSizeMultiObjectiveCostEstimator
from pareto_graph_optimizer import get_pareto_set


import logging
logger = logging.getLogger(__name__)


def optimize_single(reference_graph,
                    all_graphs,
                    all_vecs,
                    vectorizer,
                    grammar,
                    cost_estimator,
                    nearest_neighbors,
                    n_iter=100):
    """optimize_single."""
    reference_vec = vectorizer.transform([reference_graph])
    # find neighbors
    neighbors = nearest_neighbors.kneighbors(reference_vec,
                                             return_distance=False)[0]
    # compute center of mass
    reference_graphs = [all_graphs[i] for i in neighbors]
    reference_vecs = all_vecs[neighbors]
    avg_reference_vec = sp.sparse.csr_matrix.mean(reference_vecs, axis=0)

    reference_vecs = vectorizer.transform(reference_graphs)
    # compute desired distances
    desired_distances = euclidean_distances(avg_reference_vec,
                                            reference_vecs)[0]

    max_neighborhood_order = 1
    max_n_iter = 100
    pareto_set_graphs = optimize_desired_distances(reference_graph,
                                                   desired_distances,
                                                   reference_graphs,
                                                   vectorizer,
                                                   grammar,
                                                   cost_estimator,
                                                   max_neighborhood_order,
                                                   max_n_iter)
    return pareto_set_graphs


def optimize_parallel(reference_graphs,
                      all_graphs,
                      all_vecs,
                      vectorizer,
                      grammar,
                      cost_estimator,
                      nearest_neighbors,
                      n_iter=100):
    """optimize_parallel."""
    pool = multiprocessing.Pool()
    res = [apply_async(
        pool, optimize_single, args=(g,
                                     all_graphs,
                                     all_vecs,
                                     vectorizer,
                                     grammar,
                                     cost_estimator,
                                     nearest_neighbors,
                                     n_iter))
           for g in reference_graphs]
    pareto_set_graphs_list = [p.get() for p in res]
    pool.close()
    pool.join()
    return pareto_set_graphs_list


def optimize_seeds(pos_graphs,
                   neg_graphs,
                   seed_graphs=None,
                   vectorizer=None,
                   n_neighbors=5,
                   min_count=2,
                   max_neighborhood_size=100,
                   max_n_neighbors=10,
                   n_neigh_steps=2,
                   improve=True):
    """optimize."""
    all_graphs = pos_graphs + neg_graphs
    np, nn = len(pos_graphs), len(neg_graphs)
    logger.info('#positive graphs:%5d   #negative graphs:%5d' % (np, nn))
    logger.info('#seed graphs:%5d' % (len(seed_graphs)))

    grammar = GrammarWrapper(vectorizer,
                             radius_list=[1, 2, 3],
                             thickness_list=[2],
                             min_cip_count=min_count,
                             min_interface_count=min_count,
                             max_n_neighbors=max_n_neighbors,
                             n_neigh_steps=n_neigh_steps,
                             max_neighborhood_size=max_n_neighbors)
    grammar.fit(all_graphs)
    logger.info('%s' % grammar)

    moce = MultiObjectiveCostEstimator(vectorizer, improve)
    moce.fit(pos_graphs, neg_graphs)
    all_vecs = vectorizer.transform(all_graphs)
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(all_vecs)
    pareto_set_graphs_list = optimize_parallel(seed_graphs,
                                               all_graphs,
                                               all_vecs,
                                               vectorizer,
                                               grammar,
                                               moce,
                                               nearest_neighbors)

    tot_size = sum(len(graphs) for graphs in pareto_set_graphs_list)
    msg = 'pareto set sizes [%d]: ' % tot_size
    for graphs in pareto_set_graphs_list:
        msg += '[%d]' % len(graphs)
    logger.info(msg)
    return pipe(pareto_set_graphs_list, concat, list)


def optimize(pos_graphs,
             neg_graphs,
             n_neighbors=5,
             min_count=2,
             max_neighborhood_size=None,
             max_n_neighbors=None,
             n_neigh_steps=1,
             class_discretizer=2,
             class_std_discretizer=1,
             similarity_discretizer=10,
             size_discretizer=1,
             volume_discretizer=10,
             improve=True):
    """optimize."""
    vec = Vectorizer(r=3, d=3,
                     normalization=True, inner_normalization=True,
                     n_jobs=1)
    moce = SimVolPredStdSizeMultiObjectiveCostEstimator(
        vec,
        class_discretizer=class_discretizer,
        class_std_discretizer=class_std_discretizer,
        similarity_discretizer=similarity_discretizer,
        size_discretizer=size_discretizer,
        volume_discretizer=volume_discretizer,
        improve=improve)
    moce.fit(pos_graphs, neg_graphs)
    costs = moce.compute(pos_graphs + neg_graphs)
    seed_pareto_set_graphs = get_pareto_set(pos_graphs + neg_graphs, costs)
    vec_nn = Vectorizer(r=3, d=3,
                        normalization=False, inner_normalization=False,
                        n_jobs=1)
    pareto_set_graphs = optimize_seeds(
        pos_graphs,
        neg_graphs,
        seed_graphs=seed_pareto_set_graphs,
        vectorizer=vec_nn,
        n_neighbors=n_neighbors,
        min_count=min_count,
        max_neighborhood_size=max_neighborhood_size,
        max_n_neighbors=max_n_neighbors,
        n_neigh_steps=n_neigh_steps,
        improve=improve)
    pareto_set_costs = moce.compute(pareto_set_graphs)
    sel_pareto_set_graphs = get_pareto_set(pareto_set_graphs, pareto_set_costs)
    logger.info('#constructed graphs:%5d' % (len(sel_pareto_set_graphs)))
    return sel_pareto_set_graphs
