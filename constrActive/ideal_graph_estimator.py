#!/usr/bin/env python
"""Provides wrapper for estimator."""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Perceptron
from toolz import concat
from collections import defaultdict
from scipy.sparse import vstack
from constrActive.multi_objective_optimizer import NearestNeighborsMeanOptimizer
from eden.graph import Vectorizer

import logging
logger = logging.getLogger(__name__)


class IdealGraphEstimator(object):
    """Build an estimator for graphs."""

    def __init__(
            self,
            min_count=2,
            max_n_neighbors=100,
            r=3,
            d=3,
            n_neighbors=10,
            max_num_solutions=30):
        """construct."""
        self.min_count = min_count
        self.max_n_neighbors = max_n_neighbors
        self.max_num_solutions = max_num_solutions
        self.r = r
        self.d = d
        self.n_neighbors = n_neighbors

        self.clf = Perceptron(n_iter=500)
        self.vec = Vectorizer(r=r, d=d,
                              normalization=True,
                              inner_normalization=True,
                              nbits=16)
        self.gs = [.05, .1, .2, .4, .6, .8, 1, 2, 4, 6]

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        ref_graphs = self.construct(pos_graphs, neg_graphs)
        logger.debug('Working on %d constructed graphs' % len(ref_graphs))
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vec.transform(pos_graphs + neg_graphs)
        z = self.vec.transform(ref_graphs)
        n_features = z.shape[0]
        k = np.hstack([pairwise_kernels(x, z, metric='rbf', gamma=g)
                       for g in self.gs])
        step = len(ref_graphs) / 2
        n_inst, n_feat = k.shape
        txt = 'RFECV on %d instances with %d features with step: %d' % \
            (n_inst, n_feat, step)
        logger.debug(txt)
        selector = RFECV(self.clf, step=step, cv=10)
        selector = selector.fit(k, y)

        ids = list(concat([range(n_features)] * len(self.gs)))
        gs_list = list(concat([[g] * n_features for g in self.gs]))

        feat = defaultdict(list)
        for g, i, s in zip(gs_list, ids, selector.support_):
            if s:
                feat[g].append(i)

        self.mats = dict()
        for g in sorted(feat):
            mat = vstack([z[i] for i in feat[g]])
            self.mats[g] = mat

        sel_ids = set([i for i, s in zip(ids, selector.support_) if s])
        self.ideal_graphs_ = [ref_graphs[i] for i in sel_ids]
        return self

    def transform(self, graphs):
        """transform."""
        x = self.vec.transform(graphs)
        xtr = np.hstack([pairwise_kernels(x,
                                          self.mats[g], metric='rbf', gamma=g)
                         for g in sorted(self.mats)])
        return xtr

    def construct(self, pos_graphs, neg_graphs):
        """construct."""
        args = dict(
            min_count=self.min_count,
            max_n_neighbors=self.max_n_neighbors,
            r=self.r,
            d=self.d,
            n_landmarks=5,
            n_neighbors=self.n_neighbors,
            n_iter=20,
            k_best=5,
            max_num_solutions=self.max_num_solutions)
        self.active_constr = NearestNeighborsMeanOptimizer(
            improve=False, **args)
        self.active_constr.fit(pos_graphs, neg_graphs)
        graphs = pos_graphs + neg_graphs
        active_pareto_set_graphs = self.active_constr.optimize(graphs)

        self.pos_constr = NearestNeighborsMeanOptimizer(
            improve=True, **args)
        self.pos_constr.fit(pos_graphs, neg_graphs)
        pareto_set_graphs = self.pos_constr.optimize(graphs)

        sel_constructed_graphs = pareto_set_graphs + active_pareto_set_graphs
        return sel_constructed_graphs
