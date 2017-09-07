#!/usr/bin/env python
"""Provides sampling of graphs."""

from toolz.curried import pipe, concat
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async
from pareto_graph_optimizer import MultiObjectiveOptimizer
from pareto_graph_optimizer import GrammarWrapper
from pareto_graph_optimizer import MultiObjectiveCostEstimator
from pareto_graph_optimizer import SimVolPredStdSizeMultiObjectiveCostEstimator
from pareto_graph_optimizer import get_pareto_set

import logging
logger = logging.getLogger(__name__)


class VolumeConstructor(object):
    """VolumeConstructor."""

    def __init__(
            self,
            min_count=2,
            max_n_neighbors=100,
            r=3,
            d=3,
            class_discretizer=2,
            class_std_discretizer=1,
            similarity_discretizer=10,
            size_discretizer=1,
            volume_discretizer=10,
            n_neighbors=10,
            improve=True):
        """init."""
        self.improve = improve
        self.n_neighbors = n_neighbors
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
        self.grammar = GrammarWrapper(
            radius_list=[1, 2, 3],
            thickness_list=[2],
            min_cip_count=min_count,
            min_interface_count=min_count,
            max_n_neighbors=max_n_neighbors,
            n_neigh_steps=1,
            max_neighborhood_size=max_n_neighbors)
        self.sim_cost_estimator = SimVolPredStdSizeMultiObjectiveCostEstimator(
            self.vec,
            class_discretizer=class_discretizer,
            class_std_discretizer=class_std_discretizer,
            similarity_discretizer=similarity_discretizer,
            size_discretizer=size_discretizer,
            volume_discretizer=volume_discretizer,
            improve=improve)
        self.cost_estimator = MultiObjectiveCostEstimator(
            self.non_norm_vec,
            improve)
        self.nn_estimator = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        self.all_graphs = pos_graphs + neg_graphs
        self.all_vecs = self.vec.transform(self.all_graphs)
        self.grammar.fit(self.all_graphs)
        logger.info('%s' % self.grammar)
        self.sim_cost_estimator.fit(pos_graphs, neg_graphs)
        self.cost_estimator.fit(pos_graphs, neg_graphs)
        self.nn_estimator.fit(self.all_vecs)

    def sample(self, sample_graphs):
        """sample."""
        costs = self.sim_cost_estimator.compute(sample_graphs)
        seed_graphs = get_pareto_set(sample_graphs, costs)

        pareto_graphs_list = self._optimize_parallel(seed_graphs)

        tot_size = sum(len(graphs) for graphs in pareto_graphs_list)
        msg = 'pareto set sizes [%d]: ' % tot_size
        for graphs in pareto_graphs_list:
            msg += '[%d]' % len(graphs)
        logger.info(msg)
        pareto_set_graphs = pipe(pareto_graphs_list, concat, list)

        pareto_set_costs = self.sim_cost_estimator.compute(pareto_set_graphs)
        sel_pareto_set_graphs = get_pareto_set(
            pareto_set_graphs,
            pareto_set_costs)
        logger.info('#constructed graphs:%5d' % (len(sel_pareto_set_graphs)))
        return sel_pareto_set_graphs

    def _optimize_parallel(self, reference_graphs):
        """optimize_parallel."""
        pool = multiprocessing.Pool()
        res = [apply_async(
            pool, self._optimize_single, args=(g,))
            for g in reference_graphs]
        pareto_set_graphs_list = [p.get() for p in res]
        pool.close()
        pool.join()
        return pareto_set_graphs_list

    def _optimize_single(self, reference_graph):
        """optimize_single."""
        reference_vec = self.non_norm_vec.transform([reference_graph])
        # find neighbors
        neighbors = self.nn_estimator.kneighbors(
            reference_vec,
            return_distance=False)
        neighbors = neighbors[0]
        # compute center of mass
        reference_graphs = [self.all_graphs[i] for i in neighbors]
        reference_vecs = self.all_vecs[neighbors]
        avg_reference_vec = sp.sparse.csr_matrix.mean(reference_vecs, axis=0)

        reference_vecs = self.non_norm_vec.transform(reference_graphs)
        # compute desired distances
        desired_distances = euclidean_distances(
            avg_reference_vec,
            reference_vecs)
        desired_distances = desired_distances[0]
        moo = MultiObjectiveOptimizer(
            self.vec,
            self.grammar,
            self.cost_estimator,
            max_neighborhood_order=1,
            max_n_iter=100)
        moo.fit(desired_distances, reference_graphs)
        pareto_set_graphs = moo.sample(reference_graphs)

        return pareto_set_graphs

# -----------------------------------------------------------------------------


class DistanceConstructor(object):
    """DistanceConstructor."""

    def __init__(self, min_count=2, max_n_neighbors=100, r=3, d=3):
        """init."""
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)
        self.grammar = GrammarWrapper(
            radius_list=[1, 2, 3],
            thickness_list=[2],
            min_cip_count=min_count,
            min_interface_count=min_count,
            max_n_neighbors=max_n_neighbors,
            n_neigh_steps=1,
            max_neighborhood_size=max_n_neighbors)

    def fit(self, graphs):
        """fit."""
        self.grammar.fit(graphs)
        logger.info('%s' % self.grammar)

    def sample(
            self,
            seed_graph,
            reference_graphs,
            desired_distances,
            loc_pos_graphs,
            loc_neg_graphs):
        """sample."""
        cost_estimator = MultiObjectiveCostEstimator(self.vec, improve=True)
        cost_estimator.fit(loc_pos_graphs, loc_neg_graphs)
        moo = MultiObjectiveOptimizer(
            self.vec,
            self.grammar,
            cost_estimator,
            max_neighborhood_order=1,
            max_n_iter=100)
        moo.fit(desired_distances, reference_graphs)
        pareto_set_graphs = moo.sample(reference_graphs)
        logger.debug('constructed %d candidates' % len(pareto_set_graphs))
        costs = cost_estimator.compute(pareto_set_graphs)
        min_dist, candidate_graph = min(zip(costs[:, 1], pareto_set_graphs))
        return candidate_graph, pareto_set_graphs, costs
