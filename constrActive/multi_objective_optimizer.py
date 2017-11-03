#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import random
import numpy as np
from toolz.itertoolz import iterate, last
from toolz.curried import pipe, map, concat
from itertools import islice

from graphlearn.lsgg import lsgg

from eden.util import timeit
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
import scipy as sp
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async


from pareto_funcs import get_pareto_set
from constrActive.multi_objective_cost_estimator import DistRankSizeCostEstimator
from constrActive.multi_objective_cost_estimator import DistBiasSizeCostEstimator
from constrActive.multi_objective_cost_estimator import VarSimVolCostEstimator

import logging
logger = logging.getLogger(__name__)


class ParetoGraphOptimizer(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            grammar=None,
            multiobj_est=None,
            max_n_neighbors=None,
            n_iter=100,
            random_state=1):
        """init."""
        self.grammar = grammar
        self.multiobj_est = multiobj_est
        self.max_nn = max_n_neighbors
        self.n_iter = n_iter
        self.pareto_set = dict()
        self.curr_iter = 0
        self.knn_ref_dist = None
        random.seed(random_state)

    def set_objectives(self, multiobj_est):
        """set_objectives."""
        self.multiobj_est = multiobj_est

    @timeit
    def optimize(self, reference_graphs):
        """Optimize iteratively."""
        assert(self.grammar.is_fit())
        assert(self.multiobj_est.is_fit())
        seed_graph = self._init_pareto(reference_graphs)
        # iterate
        last(
            islice(
                iterate(
                    self._update_pareto_set, seed_graph), self.n_iter))
        graphs = self.pareto_set
        return self._rank(graphs)

    @timeit
    def _init_pareto(self, reference_graphs):
        start_graphs = pipe(
            reference_graphs,
            map(self._get_neighbors),
            concat,
            list)
        costs = self.multiobj_est.decision_function(start_graphs)
        self.pareto_set = get_pareto_set(start_graphs, costs)
        self._log_init_pareto(reference_graphs, start_graphs, self.pareto_set)
        seed_graph = random.choice(self.pareto_set)
        return seed_graph

    def _get_neighbors(self, graph):
        n = self.max_nn
        if n is None:
            return self.grammar.neighbors(graph)
        else:
            return self.grammar.neighbors_sample(graph, n_neighbors=n)

    def _rank(self, graphs):
        costs = self.multiobj_est.decision_function(graphs)
        costs_graphs = sorted(zip(costs, graphs), key=lambda x: x[0][0])
        costs, graphs = zip(*costs_graphs)
        return graphs

    def _update_pareto_set(self, seed_graph):
        """_update_pareto_set."""
        n_graphs = list(self._get_neighbors(seed_graph))
        if n_graphs:
            graphs = n_graphs + self.pareto_set
            costs = self.multiobj_est.decision_function(graphs)
            self.pareto_set = get_pareto_set(graphs, costs)
            self._log_update_pareto_set(costs, self.pareto_set, n_graphs)
        new_seed_graph = random.choice(self.pareto_set)
        return new_seed_graph

    def _log_init_pareto(self, reference_graphs, start_graphs, pareto_set):
        ref_size = len(reference_graphs)
        par_size = len(pareto_set)
        n_start_graphs = len(start_graphs)
        txt = 'Init pareto set: '
        txt += 'starting from: %3d references ' % ref_size
        txt += 'expanding in: %3d neighbors ' % n_start_graphs
        txt += 'for a pareto set of size: %3d ' % par_size
        logger.debug(txt)

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

# -----------------------------------------------------------------------------


class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            min_count=1,
            max_n_neighbors=None,
            n_iter=20,
            k_best=5):
        """init."""
        self.max_n_neighbors = max_n_neighbors
        self.n_iter = n_iter
        self.k_best = k_best
        decomposition_args = {
            "radius_list": [0, 1, 2, 3],
            "thickness_list": [2]}
        filter_args = {
            "min_cip_count": min_count,
            "min_interface_count": min_count}
        self.grammar = lsgg(decomposition_args, filter_args)
        self.multiobj_est = DistRankSizeCostEstimator(r=r, d=d)

    def fit(self):
        """fit."""
        pass

    def optimize(
            self,
            reference_graphs,
            desired_distances,
            loc_graphs):
        """optimize."""
        self.grammar.fit(loc_graphs)
        logger.debug(self.grammar)

        # fit objectives
        self.multiobj_est.fit(desired_distances, reference_graphs, loc_graphs)

        # setup and run optimizer
        pgo = ParetoGraphOptimizer(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            max_n_neighbors=self.max_n_neighbors,
            n_iter=self.n_iter)
        graphs = pgo.optimize(reference_graphs)

        # output a selection of the Pareto set
        graphs = self.multiobj_est.select(graphs, k_best=self.k_best)
        return graphs


def construct(
        reference_graphs,
        desired_distances,
        loc_graphs,
        r=2,
        d=5,
        min_count=1,
        max_n_neighbors=None,
        n_iter=20,
        k_best=5):
    """construct."""
    ld_opt = LocalLandmarksDistanceOptimizer(
        r=r,
        d=d,
        min_count=min_count,
        max_n_neighbors=max_n_neighbors,
        n_iter=n_iter,
        k_best=k_best)
    graphs = ld_opt.optimize(
        reference_graphs,
        desired_distances,
        loc_graphs)
    return graphs

# -----------------------------------------------------------------------------


class LandmarksDistanceOptimizer(object):
    """LandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            min_count=1,
            max_n_neighbors=None,
            n_iter=20,
            k_best=5,
            improve=True):
        """init."""
        self.r = r
        self.d = d
        self.max_n_neighbors = max_n_neighbors
        self.n_iter = n_iter
        self.k_best = k_best
        self.improve = improve
        decomposition_args = {
            "radius_list": [0, 1, 2, 3],
            "thickness_list": [2]}
        filter_args = {
            "min_cip_count": min_count,
            "min_interface_count": min_count}
        self.grammar = lsgg(decomposition_args, filter_args)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        self.multiobj_est = DistBiasSizeCostEstimator(
            pos_graphs,
            neg_graphs,
            r=self.r,
            d=self.d,
            improve=self.improve)

    def optimize(
            self,
            reference_graphs,
            desired_distances,
            loc_graphs):
        """optimize."""
        self.grammar.fit(loc_graphs)
        logger.debug(self.grammar)

        # fit objectives
        self.multiobj_est.fit(desired_distances, reference_graphs)

        # setup and run optimizer
        pgo = ParetoGraphOptimizer(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            max_n_neighbors=self.max_n_neighbors,
            n_iter=self.n_iter)
        graphs = pgo.optimize(reference_graphs)

        # output a selection of the Pareto set
        graphs = self.multiobj_est.select(graphs, k_best=self.k_best)
        return graphs


# -----------------------------------------------------------------------------


class NearestNeighborsMeanOptimizer(object):
    """NearestNeighborsMeanOptimizer."""

    def __init__(
            self,
            min_count=2,
            max_n_neighbors=None,
            r=3,
            d=3,
            n_landmarks=5,
            n_neighbors=100,
            n_iter=20,
            k_best=5,
            max_num_solutions=30,
            improve=True):
        """init."""
        self.improve = improve
        self.max_num = max_num_solutions
        self.n_landmarks = n_landmarks
        self.n_neighbors = n_neighbors
        self.nn_estimator = NearestNeighbors(n_neighbors=n_neighbors)
        self.non_norm_vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=True,
            inner_normalization=True)
        self.dist_opt = LandmarksDistanceOptimizer(
            r=r,
            d=d,
            min_count=min_count,
            max_n_neighbors=max_n_neighbors,
            n_iter=n_iter,
            k_best=k_best,
            improve=improve)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        self.all_graphs = pos_graphs + neg_graphs
        self.all_vecs = self.vec.transform(self.all_graphs)
        self.nn_estimator.fit(self.all_vecs)
        self.dist_opt.fit(pos_graphs, neg_graphs)
        self.sim_est = VarSimVolCostEstimator(improve=self.improve)
        self.sim_est.fit(pos_graphs, neg_graphs)

    def optimize(self, graphs):
        """optimize."""
        seed_graphs = self.select(graphs, max_num=self.max_num)

        # run optimization in parallel
        pareto_graphs_list = self._optimize_parallel(seed_graphs)
        self._log_result(pareto_graphs_list)

        # join all pareto sets
        pareto_set_graphs = pipe(pareto_graphs_list, concat, list)

        # pareto filter using similarity of the solutions
        sel_graphs = self.select(pareto_set_graphs, max_num=self.max_num)
        logger.debug('#constructed graphs:%5d' % (len(sel_graphs)))
        return sel_graphs

    def select(self, graphs, max_num=30):
        """select."""
        costs = self.sim_est.decision_function(graphs)
        pareto_graphs = get_pareto_set(graphs, costs)
        select_graphs = self.sim_est.select(pareto_graphs, k_best=max_num)
        i, p, s = len(graphs), len(pareto_graphs), len(select_graphs)
        logger.debug('initial:%d  pareto:%d  selected:%d' % (i, p, s))
        return select_graphs

    def _log_result(self, pareto_graphs_list):
        tot_size = sum(len(graphs) for graphs in pareto_graphs_list)
        msg = 'pareto set sizes [%d]: ' % tot_size
        for graphs in pareto_graphs_list:
            msg += '[%d]' % len(graphs)
        logger.debug(msg)

    def _optimize_parallel(self, reference_graphs):
        """optimize_parallel."""
        pool = multiprocessing.Pool()
        res = [apply_async(
            pool, self._optimize, args=(reference_graph,))
            for reference_graph in reference_graphs]
        pareto_set_graphs_list = [p.get() for p in res]
        pool.close()
        pool.join()
        return pareto_set_graphs_list

    def _optimize(self, reference_graph):
        """optimize_single."""
        constraints = self._get_constraints(reference_graph)
        graphs = self.dist_opt.optimize(*constraints)
        return graphs

    def _get_constraints(self, reference_graph):
        reference_vec = self.non_norm_vec.transform([reference_graph])
        # find neighbors
        neighbors = self.nn_estimator.kneighbors(
            reference_vec,
            return_distance=False)
        neighbors = neighbors[0]
        # compute center of mass
        landmarks = neighbors[:self.n_landmarks]
        loc_graphs = [self.all_graphs[i] for i in neighbors]
        reference_graphs = [self.all_graphs[i] for i in landmarks]
        reference_vecs = self.all_vecs[landmarks]
        avg_reference_vec = sp.sparse.csr_matrix.mean(reference_vecs, axis=0)

        reference_vecs = self.non_norm_vec.transform(reference_graphs)
        # compute desired distances
        desired_distances = euclidean_distances(
            avg_reference_vec,
            reference_vecs)
        desired_distances = desired_distances[0]
        return reference_graphs, desired_distances, loc_graphs
