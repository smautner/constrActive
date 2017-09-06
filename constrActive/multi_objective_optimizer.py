#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import random
import numpy as np
import scipy as sp
from scipy.stats import rankdata
from toolz.itertoolz import iterate, last
from toolz.curried import pipe, map, concat, curry
from itertools import islice
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import euclidean_distances

from graphlearn.lsgg import lsgg
from pareto_funcs import get_pareto_set
from eden.graph import Vectorizer
from eden.util import timeit, describe

import logging
logger = logging.getLogger(__name__)

# TODO: make the distance from each reference graph as an individual objective
# to treat as a multi objective case


class InstancesDistanceCostEstimator(object):
    """InstancesDistanceCostEstimator."""

    def __init__(self, vectorizer=Vectorizer()):
        """init."""
        self.desired_distances = None
        self.reference_vecs = None
        self.vectorizer = vectorizer

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        return self

    def _avg_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        dist_diff = (distances - d)
        avg_dist_diff = np.mean(np.absolute(dist_diff))
        return avg_dist_diff

    def decision_function(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        avg_distance_diff = np.array([self._avg_distance_diff(vec)
                                      for vec in x])
        avg_distance_diff = avg_distance_diff.reshape(-1, 1)
        return avg_distance_diff


class InstancesMultiDistanceCostEstimator(object):
    """InstancesMultiDistanceCostEstimator."""

    def __init__(self, vectorizer=Vectorizer()):
        """init."""
        self.desired_distances = None
        self.reference_vecs = None
        self.vectorizer = vectorizer

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        return self

    def _multi_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        dist_diff = (distances - d)
        return np.absolute(dist_diff).reshape(1, -1)

    def decision_function(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        multi_distance_diff = np.vstack([self._multi_distance_diff(vec)
                                         for vec in x])
        return multi_distance_diff


class ClassBiasCostEstimator(object):
    """ClassBiasCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.estimator = SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True,
                                       n_jobs=1)
        self.improve = improve

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vectorizer.transform(graphs)
        self.estimator = self.estimator.fit(x, y)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = - np.absolute(scores)
        else:
            scores = - scores
        scores = scores.reshape(-1, 1)
        return scores


class RankBiasCostEstimator(object):
    """RankBiasCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.estimator = SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True)
        self.improve = improve

    def fit(self, ranked_graphs):
        """fit."""
        # TODO: fare il preference ranking
        x = self.vectorizer.transform(ranked_graphs)
        r, c = x.shape
        pos = []
        neg = []
        for i in range(r - 1):
            for j in range(i + 1, r):
                p = x[i] - x[j]
                n = - p
                pos.append(p)
                neg.append(n)
        y = np.array([1] * len(pos) + [-1] * len(neg))
        pos = sp.sparse.vstack(pos)
        neg = sp.sparse.vstack(neg)
        x_ranks = sp.sparse.vstack([pos, neg])
        logger.debug('fitting: %s' % describe(x_ranks))
        self.estimator = self.estimator.fit(x_ranks, y)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = - np.absolute(scores)
        else:
            scores = - scores
        scores = scores.reshape(-1, 1)
        return scores


class SizeCostEstimator(object):
    """ClassBiasCostEstimator."""

    def __init__(self):
        """init."""
        pass

    def fit(self, graphs):
        """fit."""
        self.reference_size = np.percentile([len(g) for g in graphs], 50)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        sizes = np.array([len(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        size_diffs = size_diffs.reshape(-1, 1)
        return size_diffs


# -----------------------------------------------------------------------------

class MultiObjectiveCostEstimator(object):
    """MultiObjectiveCostEstimator."""

    def __init__(self, estimators=None):
        """Initialize."""
        self.set_params(estimators)

    def set_params(self, estimators):
        """set_params."""
        self.estimators = estimators

    def decision_function(self, graphs):
        """decision_function."""
        cost_vec = [estimator.decision_function(graphs)
                    for estimator in self.estimators]
        costs = np.hstack(cost_vec)
        return costs

    def is_fit(self):
        """is_fit."""
        return self.estimators is not None

    def select(self, graphs, k_best=10):
        """select."""
        costs = self.decision_function(graphs)
        ranks = [rankdata(costs[:, i], method='min')
                 for i in range(costs.shape[1])]
        ranks = np.vstack(ranks).T
        agg_ranks = np.sum(ranks, axis=1)
        ids = np.argsort(agg_ranks)
        k_best_graphs = [graphs[id] for id in ids[:k_best]]
        return k_best_graphs


# -----------------------------------------------------------------------------


class DistanceBiasSizeCostEstimator(MultiObjectiveCostEstimator):
    """DistanceBiasSizeCostEstimator."""

    def __init__(self, estimators=None):
        """Initialize."""
        super(DistanceBiasSizeCostEstimator, self).__init__(estimators)

    @timeit
    def fit(
            self,
            desired_distances,
            reference_graphs,
            ranked_graphs):
        """fit."""
        vec = Vectorizer(
            complexity=3,
            normalization=False,
            inner_normalization=False)

        d_est = InstancesMultiDistanceCostEstimator(vec)
        d_est.fit(desired_distances, reference_graphs)

        b_est = RankBiasCostEstimator(vec, improve=True)
        b_est.fit(ranked_graphs)

        s_est = SizeCostEstimator()
        s_est.fit(reference_graphs)

        self.estimators = [d_est, b_est, s_est]
        return self

# -----------------------------------------------------------------------------


class ParetoGraphOptimizer(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            radius_list=[0, 1, 2, 3],
            thickness_list=[1],
            min_cip_count=2,
            min_interface_count=2,
            max_n_neighbors=None,
            n_iter=100,
            random_state=1):
        """init."""
        decomposition_args = {
            "radius_list": radius_list,
            "thickness_list": thickness_list}
        filter_args = {
            "min_cip_count": min_cip_count,
            "min_interface_count": min_interface_count}
        self.grammar = lsgg(decomposition_args, filter_args)
        self.multiobj_est = MultiObjectiveCostEstimator()
        self.max_nn = max_n_neighbors
        self.n_iter = n_iter
        self.pareto_set = dict()
        self.curr_iter = 0
        self.knn_ref_dist = None
        random.seed(random_state)

    def fit(self, domain_graphs):
        """fit."""
        self.grammar.fit(domain_graphs)
        logger.debug(self.grammar)

    def get_objectives(self, multiobj_est):
        """get_objectives."""
        self.multiobj_est = multiobj_est

    @timeit
    def _init_pareto(self, reference_graphs):
        n = self.max_nn
        if n is None:
            get_neighbors = self.grammar.neighbors
        else:
            get_neighbors = curry(self.grammar.neighbors_sample)(n_neighbors=n)
        start_graphs = pipe(
            reference_graphs,
            map(get_neighbors),
            concat,
            list)
        costs = self.multiobj_est.decision_function(start_graphs)
        self.pareto_set = get_pareto_set(start_graphs, costs)
        self.seed_graph = random.choice(self.pareto_set)
        self._log_init_pareto(reference_graphs, start_graphs, self.pareto_set)
        return self

    def _log_init_pareto(self, reference_graphs, start_graphs, pareto_set):
        ref_size = len(reference_graphs)
        par_size = len(pareto_set)
        n_start_graphs = len(start_graphs)
        txt = 'Init pareto set: '
        txt += 'starting from: %3d references ' % ref_size
        txt += 'expanding in: %3d neighbors ' % n_start_graphs
        txt += 'for a pareto set of size: %3d ' % par_size
        logger.debug(txt)

    def _rank(self, graphs):
        costs = self.multiobj_est.decision_function(graphs)
        costs_graphs = sorted(zip(costs, graphs), key=lambda x: x[0][0])
        costs, graphs = zip(*costs_graphs)
        return graphs

    @timeit
    def optimize(self, reference_graphs):
        """Optimize iteratively."""
        self._init_pareto(reference_graphs)
        # iterate
        last(
            islice(
                iterate(
                    self._update_pareto_set, self.seed_graph), self.n_iter))
        graphs = self.pareto_set
        return self._rank(graphs)

    def _update_pareto_set(self, seed_graph):
        """_update_pareto_set."""
        n = self.max_nn
        if n is None:
            get_neighbors = self.grammar.neighbors
        else:
            get_neighbors = curry(self.grammar.neighbors_sample)(n_neighbors=n)
        n_graphs = list(get_neighbors(seed_graph))
        if n_graphs:
            graphs = n_graphs + self.pareto_set
            costs = self.multiobj_est.decision_function(graphs)
            self.pareto_set = get_pareto_set(graphs, costs)
            self._log_update_pareto_set(costs, self.pareto_set, n_graphs)
        new_seed_graph = random.choice(self.pareto_set)
        return new_seed_graph

    def _log_update_pareto_set(self, costs, pareto_set, n_graphs):
        self.curr_iter += 1
        min_dist = min(costs[:, 0])
        par_size = len(pareto_set)
        med_dist = np.percentile(costs[:, 0], 50)
        txt = 'iter: %3d ' % self.curr_iter
        txt += 'current min dist: %.6f ' % min_dist
        txt += 'median dist: %.6f ' % med_dist
        txt += 'in pareto set of size: %3d ' % par_size
        txt += 'expanding: %3d ' % len(n_graphs)
        logger.debug(txt)


class MultiObjectiveSamplerBKP(object):
    """MultiObjectiveSampler."""

    def __init__(
            self,
            radius_list=[0, 1, 2, 3],
            thickness_list=[1],
            min_cip_count=2,
            min_interface_count=2,
            max_n_neighbors=None,
            n_iter=100,
            random_state=1):
        """init."""
        decomposition_args = {
            "radius_list": radius_list,
            "thickness_list": thickness_list}
        filter_args = {
            "min_cip_count": min_cip_count,
            "min_interface_count": min_interface_count}
        self.grammar = lsgg(decomposition_args, filter_args)
        self.multiobj_est = MultiObjectiveCostEstimator()
        self.max_nn = max_n_neighbors
        self.n_iter = n_iter
        self.pareto_set = dict()
        self.curr_iter = 0
        self.knn_ref_dist = None
        random.seed(random_state)

    def fit(self,
            desired_distances,
            reference_graphs,
            domain_graphs,
            pos_graphs,
            neg_graphs):
        """fit."""
        self._fit_grammar(domain_graphs)
        self.knn_ref_dist = self._compute_1_node_distance(reference_graphs[0])
        estimators = self._fit_multiobjectives(
            desired_distances,
            reference_graphs,
            domain_graphs,
            pos_graphs,
            neg_graphs)
        self.multiobj_est.set_params(estimators)

    @timeit
    def _fit_grammar(self, domain_graphs):
        self.grammar.fit(domain_graphs)
        logger.info(self.grammar)

    @timeit
    def _fit_multiobjectives(
            self,
            desired_distances,
            reference_graphs,
            domain_graphs,
            pos_graphs,
            neg_graphs):
        vec = Vectorizer(
            complexity=3,
            normalization=False,
            inner_normalization=False)
        d_est = InstancesMultiDistanceCostEstimator(vec)
        d_est.fit(desired_distances, reference_graphs)

        # b_est = ClassBiasCostEstimator(vec, improve=True)
        # b_est.fit(pos_graphs, neg_graphs)

        b_est = RankBiasCostEstimator(vec, improve=True)
        b_est.fit(pos_graphs + neg_graphs)

        s_est = SizeCostEstimator()
        s_est.fit(reference_graphs)

        return [d_est, b_est, s_est]

    @timeit
    def _init_pareto(self, reference_graphs):
        n = self.max_nn
        if n is None:
            get_neighbors = self.grammar.neighbors
        else:
            get_neighbors = curry(self.grammar.neighbors_sample)(n_neighbors=n)
        start_graphs = pipe(
            reference_graphs,
            map(get_neighbors),
            concat,
            list)
        costs = self.multiobj_est.decision_function(start_graphs)
        self.pareto_set = get_pareto_set(start_graphs, costs)
        self.seed_graph = random.choice(self.pareto_set)
        self._log_init_pareto(reference_graphs, start_graphs, self.pareto_set)
        return self

    def _log_init_pareto(self, reference_graphs, start_graphs, pareto_set):
        ref_size = len(reference_graphs)
        par_size = len(pareto_set)
        n_start_graphs = len(start_graphs)
        txt = 'Init pareto set: '
        txt += 'starting from: %3d references ' % ref_size
        txt += 'expanding in: %3d neighbors ' % n_start_graphs
        txt += 'for a pareto set of size: %3d ' % par_size
        logger.debug(txt)

    def _rank(self, graphs):
        costs = self.multiobj_est.decision_function(graphs)
        costs_graphs = sorted(zip(costs, graphs), key=lambda x: x[0][0])
        costs, graphs = zip(*costs_graphs)
        return graphs

    @timeit
    def sample(self, reference_graphs):
        """Optimize iteratively."""
        self._init_pareto(reference_graphs)
        # iterate
        last(
            islice(
                iterate(
                    self._update_pareto_set, self.seed_graph), self.n_iter))
        graphs = self.pareto_set
        return self._rank(graphs)

    def _update_pareto_set(self, seed_graph):
        """_update_pareto_set."""
        n = self.max_nn
        if n is None:
            get_neighbors = self.grammar.neighbors
        else:
            get_neighbors = curry(self.grammar.neighbors_sample)(n_neighbors=n)
        n_graphs = list(get_neighbors(seed_graph))
        if n_graphs:
            graphs = n_graphs + self.pareto_set
            costs = self.multiobj_est.decision_function(graphs)
            self.pareto_set = get_pareto_set(graphs, costs)
            self._log_update_pareto_set(costs, self.pareto_set, n_graphs)
        new_seed_graph = random.choice(self.pareto_set)
        return new_seed_graph

    def _log_update_pareto_set(self, costs, pareto_set, n_graphs):
        self.curr_iter += 1
        min_dist = min(costs[:, 0])
        rel_min_dist = min_dist / self.knn_ref_dist
        par_size = len(pareto_set)
        med_dist = np.percentile(costs[:, 0], 50) / self.knn_ref_dist
        txt = 'iter: %3d ' % self.curr_iter
        txt += 'current min dist (1-node-diff units): %.6f ' % rel_min_dist
        txt += 'median dist (1-node-diff units): %.6f ' % med_dist
        txt += 'in pareto set of size: %3d ' % par_size
        txt += 'expanding: %3d ' % len(n_graphs)
        logger.debug(txt)

    def _compute_1_node_distance(self, graph):
        vec = Vectorizer(
            complexity=3,
            normalization=False,
            inner_normalization=False)
        gs = []
        for u in graph.nodes():
            gt = graph.copy()
            gt.node[u]['label'] = '_'
            gs.append(gt)
        ref = vec.transform([graph])
        vecs = vec.transform(gs)
        ds = euclidean_distances(vecs, ref)
        return np.percentile(ds, 50)
