#!/usr/bin/env python
"""Provides sampling of graphs."""

from toolz.curried import pipe, concat
import scipy as sp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async
from pareto_graph_optimizer import optimize_desired_distances
from pareto_graph_optimizer import GrammarWrapper
from pareto_graph_optimizer import MultiObjectiveCostEstimator
# from pareto_graph_optimizer import diversity_pareto_selection
from pareto_graph_optimizer import SimVolPredStdSizeMultiObjectiveCostEstimator
from pareto_graph_optimizer import get_pareto_set


import logging
logger = logging.getLogger(__name__)


def min_similarity_selection(matrix, scores=None, max_num=None):
    """Select the max_num most dissimilar instances.

    Given a similarity matrix and a score associate to each instance,
    iteratively find the most similar pair and remove the one with the
    smallest score until only max_num remain.
    """
    similarity_matrix = matrix.copy()
    size = similarity_matrix.shape[0]
    # remove diagonal elements, so that sim(i,i)=0
    # and the pair i,i is not selected
    similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
    # iterate size - k times, i.e. until only k instances are left
    for t in range(size - max_num):
        # find pairs with largest similarity
        (i, j) = np.unravel_index(
            np.argmax(similarity_matrix), similarity_matrix.shape)
        # choose instance with smallest score as the one to be removed
        if scores[i] < scores[j]:
            id = i
        else:
            id = j
        # remove instance with lower score by setting all its
        # pairwise similarities to 0
        similarity_matrix[id, :] = 0
        similarity_matrix[:, id] = 0
    # extract surviving elements, i.e. element that have a row
    # that is not only 0s
    select_ids = [ind for ind, row in enumerate(similarity_matrix)
                  if np.sum(row) > 0]
    return select_ids


def _outliers(graphs, k=3):
    vec = Vectorizer(r=3, d=3,
                     normalization=False, inner_normalization=False, n_jobs=1)
    x = vec.transform(graphs)
    knn = NearestNeighbors()
    knn.fit(x)
    neigbhbors = knn.kneighbors(x, n_neighbors=k, return_distance=False)
    outlier_list = []
    non_outlier_list = []
    for i, ns in enumerate(neigbhbors):
        not_outlier = False
        for n in ns[1:]:
            if i in list(neigbhbors[n, :]):
                not_outlier = True
                break
        if not_outlier is False:
            outlier_list.append(i)
        else:
            non_outlier_list.append(i)
    return outlier_list, non_outlier_list


def _select_non_outliers(graphs, k=3):
    outlier_list, non_outlier_list = _outliers(graphs, k)
    graphs = [graphs[i] for i in non_outlier_list]
    logging.info('outlier removal:%d' % len(graphs))
    return graphs


def _remove_similar_pairs(graphs):
    vec = Vectorizer(r=3, d=3,
                     normalization=False, inner_normalization=False, n_jobs=1)
    x = vec.transform(graphs)
    matrix = cosine_similarity(x)
    scores = np.array([1] * len(graphs))
    ids = min_similarity_selection(matrix,
                                   scores=scores,
                                   max_num=len(graphs) / 2)
    graphs = [graphs[i] for i in ids]
    logging.info('similar pairs removal:%d' % len(graphs))
    return graphs


def _size_filter(graphs, fraction_to_remove=.1):
    frac = 1.0 - fraction_to_remove / 2
    size = len(graphs)
    graphs = sorted(graphs, key=lambda g: len(g))[:int(size * frac)]
    graphs = sorted(graphs, key=lambda g: len(g), reverse=True)
    graphs = graphs[:int(size * frac)]
    logging.info('size filter:%d' % len(graphs))
    return graphs


def _random_sample(graphs, max_size):
    if len(graphs) > max_size:
        graphs = random.sample(graphs, max_size)
    logging.info('random sample:%d' % len(graphs))
    return graphs


def pre_process(graphs,
                fraction_to_remove=.1,
                n_neighbors_for_outliers=3,
                remove_similar=True,
                max_size=500):
    """pre_process."""
    logging.info('original size:%d' % len(graphs))
    graphs = _random_sample(graphs, 3000)
    graphs = _size_filter(graphs, fraction_to_remove)
    graphs = _select_non_outliers(graphs, k=n_neighbors_for_outliers)
    if remove_similar:
        graphs = _remove_similar_pairs(graphs)
    graphs = _random_sample(graphs, max_size)
    return graphs

# ---------------------------------------------------------------------------


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
