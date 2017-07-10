#!/usr/bin/env python
"""Provides sampling of graphs."""


import scipy as sp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async
from eden.graph import Vectorizer
from pareto_graph_optimizer import optimize_desired_distances
from pareto_graph_optimizer import GrammarWrapper
from pareto_graph_optimizer import MultiObjectiveCostEstimator
from pareto_graph_optimizer import diversity_pareto_selection

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


def compute_score(x, y, k=5):
    # induce classification model and compute cross validated confidence
    # compute std estimate of the score for each instance
    y_preds = []
    n_iterations = 5
    for rnd in range(n_iterations):
        sgd = SGDClassifier(average=True,
                            class_weight='balanced',
                            shuffle=True,
                            n_jobs=-1,
                            random_state=rnd)
        cv = StratifiedKFold(n_splits=5, random_state=rnd)
        y_pred = cross_val_predict(sgd, x, y,
                                   method='decision_function', cv=cv)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).T
    mean_y_preds = np.mean(y_preds, axis=1)
    std_y_preds = np.std(y_preds, axis=1)
    # predictive score is the mean+std
    preds = mean_y_preds + std_y_preds
    # compute score averaging k neighbors
    nn = NearestNeighbors()
    nn.fit(x)
    distances, neighbors = nn.kneighbors(x, k)
    cum_dists = np.sum(distances, axis=1)
    avg_scores = np.multiply(cum_dists[neighbors], preds[neighbors])
    vol_scores = np.sum(avg_scores, axis=1)
    # the idea is to find the largest volume, i.e. the most diverse set
    # that is at the same time high scoring
    # for each vertex we compute the sum of the distances to other neighbors
    # we weight each cumulative distance by the score of each neighbor
    # we consider as score the cumulative score in the neighborhood
    return vol_scores


def make_data(pos_graphs, neg_graphs, vec):
    # prepare x,y
    y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
    all_graphs = pos_graphs + neg_graphs
    x = vec.transform(all_graphs)
    return all_graphs, x, y


def select_diverse_representative_set(all_graphs, x, scores,
                                      size=18, top_perc=0.1):
    # sort ids from the most uncertain average prediction
    ids = np.argsort(-scores)
    # select a perc of the num of positives
    k_top_scoring = int(len(all_graphs) * top_perc)
    sel_ids = ids[:k_top_scoring]
    sel_vecs = x[sel_ids]
    sel_scores = scores[sel_ids]
    # compute the similarity matrix as cos similarity
    similarity_matrix = cosine_similarity(sel_vecs)
    # select 'size' most different instances
    rel_seed_ids = min_similarity_selection(similarity_matrix,
                                            scores=sel_scores,
                                            max_num=size)
    seed_ids = [sel_ids[i] for i in rel_seed_ids]
    sel_seed_graphs = [all_graphs[gid] for gid in seed_ids]
    return seed_ids, sel_seed_graphs


def select_seeds(pos_graphs, neg_graphs, vec,
                 size=18, top_perc=0.1, k=5, improve=True):
    all_graphs, x, y = make_data(pos_graphs, neg_graphs, vec)
    # compute score as mean confidence in k neighbors
    scores = compute_score(x, y, k)
    if improve is False:
        # score is high if the prediction is near zero
        scores = 1 / (1 + np.absolute(scores))
    dat = select_diverse_representative_set(all_graphs, x, scores,
                                            size, top_perc)
    seed_ids, sel_seed_graphs = dat
    seed_scores = scores[seed_ids]
    return seed_ids, sel_seed_graphs, seed_scores


def optimize_single(reference_graph,
                    all_graphs,
                    all_vecs,
                    vectorizer,
                    grammar,
                    cost_estimator,
                    nearest_neighbors,
                    n_iter=100):
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

    max_neighborhood_order = 2
    max_neighborhood_size = 100
    max_n_iter = 100
    graphs_tuple = optimize_desired_distances(reference_graph,
                                              desired_distances,
                                              reference_graphs,
                                              vectorizer,
                                              grammar,
                                              cost_estimator,
                                              max_neighborhood_order,
                                              max_neighborhood_size,
                                              max_n_iter)
    return graphs_tuple


def optimize_set(reference_graphs,
                 all_graphs,
                 all_vecs,
                 vectorizer,
                 grammar,
                 cost_estimator,
                 nearest_neighbors,
                 n_iter=100):
    pool = multiprocessing.Pool()
    graphs_tuple_list = [apply_async(
        pool, optimize_single, args=(g,
                                     all_graphs,
                                     all_vecs,
                                     vectorizer,
                                     grammar,
                                     cost_estimator,
                                     nearest_neighbors,
                                     n_iter))
                         for g in reference_graphs]
    graphs_tuple_list = [p.get() for p in graphs_tuple_list]
    pool.close()
    pool.join()
    return graphs_tuple_list


def optimize(pos_graphs,
             neg_graphs,
             n_seeds=3,
             n_neighbors=5,
             min_count=2,
             max_neighborhood_size=100,
             return_references=True,
             improve=True):
    all_graphs = pos_graphs + neg_graphs
    np, nn = len(pos_graphs), len(neg_graphs)
    logger.info('#positive graphs:%5d   #negative graphs:%5d' % (np, nn))

    vectorizer = Vectorizer(r=3, d=3,
                            normalization=False,
                            inner_normalization=False,
                            n_jobs=1)
    grammar = GrammarWrapper(vectorizer,
                             radius_list=[1, 2, 3],
                             thickness_list=[2],
                             min_cip_count=min_count,
                             min_interface_count=min_count)
    grammar.fit(all_graphs)
    logger.info('%s' % grammar)

    cost_estimator = MultiObjectiveCostEstimator(vectorizer, improve).fit(
        pos_graphs, neg_graphs)

    res = select_seeds(pos_graphs, neg_graphs, vectorizer, size=n_seeds,
                       top_perc=0.25, k=5, improve=improve)
    seed_ids, sel_seed_graphs, seed_scores = res

    all_vecs = vectorizer.transform(all_graphs)
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(all_vecs)

    graphs_tuple_list = optimize_set(sel_seed_graphs, all_graphs, all_vecs,
                                     vectorizer, grammar, cost_estimator,
                                     nearest_neighbors)
    pareto_set_graphs, sel_seed_graphs = diversity_pareto_selection(
        sel_seed_graphs,
        graphs_tuple_list,
        pos_graphs,
        neg_graphs,
        vectorizer,
        improve)
    result = []
    result.append(pareto_set_graphs)
    if return_references:
        result.append(sel_seed_graphs)
    return result
