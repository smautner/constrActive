#!/usr/bin/env python
"""Provides wrapper for estimator."""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Perceptron
from toolz import concat
from collections import defaultdict
from scipy.sparse import vstack
from constrActive import optimize_seeds
from pareto_graph_optimizer import SimVolPredStdSizeMultiObjectiveCostEstimator
from pareto_graph_optimizer import get_pareto_set
from eden.graph import Vectorizer
from constrActive import optimize

import logging
logger = logging.getLogger(__name__)


class IdealGraphEstimator(object):
    """Build an estimator for graphs."""

    def __init__(self, min_count=3, discretization=50, max_n_neighbors=10):
        """construct."""
        self.min_count = min_count
        self.discretization = discretization
        self.max_n_neighbors = max_n_neighbors

        self.clf = Perceptron(n_iter=500)
        self.vec = Vectorizer(r=3, d=6,
                              normalization=True,
                              inner_normalization=True,
                              n_jobs=1,
                              nbits=16)
        self.gs = [.05, .1, .2, .4, .6, .8, 1, 2, 4, 6]

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        sel_constructed_graphs = self.construct(
            pos_graphs,
            neg_graphs,
            min_count=self.min_count,
            discretization=self.discretization,
            max_n_neighbors=self.max_n_neighbors)
        self._fit(sel_constructed_graphs, pos_graphs, neg_graphs)
        return self

    def _fit(self, ref_graphs, pos_graphs, neg_graphs):
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vec.transform(pos_graphs + neg_graphs)
        z = self.vec.transform(ref_graphs)
        n_features = z.shape[0]
        k = np.hstack([pairwise_kernels(x, z, metric='rbf', gamma=g)
                       for g in self.gs])
        step = len(ref_graphs) / 2
        n_inst, n_feat = k.shape
        txt = 'RFECV on %d instances with %d features with step: %d' % (n_inst, n_feat, step)
        logger.info(txt)
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

    def partial_fit(self, sel_graphs, pos_graphs, neg_graphs):
        """partial_fit."""
        sel_constructed_graphs = self.partial_construct(
            sel_graphs, pos_graphs, neg_graphs,
            min_count=self.min_count,
            discretization=self.discretization,
            max_n_neighbors=self.max_n_neighbors)
        ref_graphs = self.ideal_graphs_ + sel_constructed_graphs
        self._fit(ref_graphs, pos_graphs, neg_graphs)
        return self

    def transform(self, graphs):
        """transform."""
        x = self.vec.transform(graphs)
        xtr = np.hstack([pairwise_kernels(x,
                                          self.mats[g], metric='rbf', gamma=g)
                         for g in sorted(self.mats)])
        return xtr

    def construct(self,
                  pos_graphs,
                  neg_graphs,
                  min_count=2,
                  discretization=20,
                  max_n_neighbors=10):
        """construct."""
        args = dict(n_neighbors=15,
                    min_count=min_count,
                    max_neighborhood_size=max_n_neighbors,
                    max_n_neighbors=max_n_neighbors,
                    n_neigh_steps=1,
                    class_discretizer=1000,
                    class_std_discretizer=discretization,
                    similarity_discretizer=discretization,
                    size_discretizer=1,
                    volume_discretizer=discretization)
        pareto_set_graphs = optimize(pos_graphs, neg_graphs,
                                     improve=True, **args)
        active_pareto_set_graphs = optimize(pos_graphs, neg_graphs,
                                            improve=False, **args)
        sel_constructed_graphs = pareto_set_graphs + active_pareto_set_graphs
        return sel_constructed_graphs

    def partial_construct(self,
                          sel_graphs,
                          pos_graphs,
                          neg_graphs,
                          min_count=2,
                          discretization=20,
                          max_n_neighbors=10):
        """partial_construct."""
        vec = Vectorizer(r=3, d=6,
                         normalization=True,
                         inner_normalization=True,
                         n_jobs=1)
        vecnn = Vectorizer(r=3, d=6,
                           normalization=False,
                           inner_normalization=False,
                           n_jobs=1)
        d = discretization
        logger.info('Active graphs generation')
        moce = SimVolPredStdSizeMultiObjectiveCostEstimator(
            vec,
            class_discretizer=100,
            class_std_discretizer=d,
            similarity_discretizer=d,
            size_discretizer=d,
            volume_discretizer=d,
            improve=False)
        moce.fit(pos_graphs, neg_graphs)
        costs = moce.compute(sel_graphs)
        active_seed_pareto_set_graphs = get_pareto_set(sel_graphs, costs)
        active_pareto_set_graphs = optimize_seeds(
            pos_graphs, neg_graphs, improve=False,
            seed_graphs=active_seed_pareto_set_graphs,
            vectorizer=vecnn,
            n_neighbors=15,
            min_count=min_count,
            max_neighborhood_size=max_n_neighbors,
            max_n_neighbors=max_n_neighbors,
            n_neigh_steps=1)
        costs = moce.compute(active_pareto_set_graphs)
        sel_active_pareto_set_graphs = get_pareto_set(active_pareto_set_graphs,
                                                      costs)
        logger.info('# selected: %5d' % len(sel_active_pareto_set_graphs))
        return sel_active_pareto_set_graphs
