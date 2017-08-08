#!/usr/bin/env python
"""Provides utilities for subsampling."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
import random
from sklearn.metrics.pairwise import cosine_similarity

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
                initial_max_size=3000,
                fraction_to_remove=.1,
                n_neighbors_for_outliers=3,
                remove_similar=True,
                max_size=500):
    """pre_process."""
    logging.info('original size:%d' % len(graphs))
    graphs = _random_sample(graphs, initial_max_size)
    graphs = _size_filter(graphs, fraction_to_remove)
    graphs = _select_non_outliers(graphs, k=n_neighbors_for_outliers)
    if remove_similar:
        graphs = _remove_similar_pairs(graphs)
    graphs = _random_sample(graphs, max_size)
    return graphs
