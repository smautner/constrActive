#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import numpy as np
from toolz.itertoolz import iterate, last
from toolz.curried import pipe, map, concat, curry
from itertools import islice
import random
from scipy.stats import rankdata
from graphlearn.lsgg import lsgg
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from pareto_funcs import get_pareto_set
import dill
import os.path

import logging
logger = logging.getLogger(__name__)


class GrammarWrapper(object):
    """GrammarWrapper."""

    def __init__(self,
                 radius_list=[0, 1, 2, 3],
                 thickness_list=[1],
                 min_cip_count=4,
                 min_interface_count=2,
                 max_n_neighbors=None,
                 n_neigh_steps=1,
                 max_neighborhood_size=None):
        """init."""
        self.max_n_neighbors = max_n_neighbors
        self.n_neigh_steps = n_neigh_steps
        self.max_neighborhood_size = max_neighborhood_size
        self.grammar = lsgg(
            decomposition_args=dict(
                radius_list=radius_list,
                thickness_list=thickness_list,
                hash_bitmask=2**16 - 1),
            filter_args=dict(
                min_cip_count=min_cip_count,
                min_interface_count=min_interface_count)
        )

    def __repr__(self):
        """repr."""
        n_interfaces, n_cores, n_cips = self.grammar.size()
        txt = '#interfaces: %5d   ' % n_interfaces
        txt += '#cores: %5d   ' % n_cores
        txt += '#core-interface-pairs: %5d' % n_cips
        return txt

    def fit(self, graphs):
        """fit."""
        self.grammar.fit(graphs)
        return self

    def neighborhood(self, graph):
        """neighborhood."""
        m = self.max_n_neighbors
        if m is None:
            neighbors = list(self.grammar.neighbors(graph))
        else:
            neighbors = list(self.grammar.neighbors_sample(graph, m))
        return neighbors

    def _random_sample(self, graphs):
        m = self.max_neighborhood_size
        if m is None:
            return graphs
        else:
            if len(graphs) > m:
                graphs = random.sample(graphs, self.max_neighborhood_size)
            return graphs

    def set_neighborhood(self, graphs):
        """set_neighborhood."""
        graphs = pipe(graphs,
                      map(self.neighborhood),
                      concat,
                      list,
                      self._random_sample)
        return graphs

    def iterated_neighborhood(self, graph):
        """iterated_neighborhood."""
        n = self.n_neigh_steps
        if n == 1:
            return pipe(graph, self.neighborhood, self._random_sample)
        else:
            return last(islice(iterate(self.set_neighborhood, [graph]), n + 1))

# -----------------------------------------------------------------------------


class MultiObjectiveCostEstimator(object):
    """MultiObjectiveCostEstimator."""

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
        self.y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        self.vecs = self.vectorizer.transform(graphs)
        self._fit_class_objective(self.vecs, self.y)
        return self

    def _fit_class_objective(self, x, y):
        self.estimator = self.estimator.fit(x, y)
        return self

    def _estimate_class_objective(self, x):
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = - np.absolute(scores)
        return scores

    def _fit_distance_objective(self,
                                reference_graphs=None,
                                desired_distances=None):
        self.desired_distances = desired_distances
        x = self.vectorizer.transform(reference_graphs)
        self.reference_scores = self._estimate_class_objective(x)
        self.reference_vecs = x

    def _avg_relative_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        relative_distance_diff = (distances - d) / d
        avg_relative_distance_diff = np.mean(np.square(relative_distance_diff))
        return avg_relative_distance_diff

    def _estimate_distance_objective(self, x):
        """predict_distance."""
        return np.array([self._avg_relative_distance_diff(vec) for vec in x])

    def _fit_size_objective(self, graphs):
        self.reference_size = np.percentile([len(g) for g in graphs], 50)

    def _estimate_size_objective(self, graphs):
        sizes = np.array([len(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        return size_diffs

    def fit_local(self, reference_graphs=None, desired_distances=None):
        """fit_local."""
        self._fit_distance_objective(reference_graphs, desired_distances)
        self._fit_size_objective(reference_graphs)

    def _estimate_relative_class_objective(self, x):
        """predict_quality."""
        scores = self._estimate_class_objective(x)
        qualities = []
        for score in scores:
            to_be_ranked = np.hstack([score, self.reference_scores])
            ranked = rankdata(-to_be_ranked, method='min')
            qualities.append(ranked[0])
        return np.array(qualities)

    def compute(self, graphs):
        """predict."""
        x = self.vectorizer.transform(graphs)
        q = self._estimate_relative_class_objective(x).reshape(-1, 1)
        d = self._estimate_distance_objective(x).reshape(-1, 1)
        s = self._estimate_size_objective(graphs).reshape(-1, 1)
        costs = np.hstack([q, d, s])
        return costs

# -----------------------------------------------------------------------------


class SimVolPredStdSizeMultiObjectiveCostEstimator(object):
    """SimVolPredStdSizeMultiObjectiveCostEstimator."""

    def __init__(self,
                 vectorizer,
                 k=9,
                 n_estimators=20,
                 class_discretizer=1,
                 class_std_discretizer=1,
                 similarity_discretizer=20,
                 size_discretizer=2,
                 volume_discretizer=20,
                 improve=True):
        """init."""
        self.class_discretizer = class_discretizer
        self.class_std_discretizer = class_std_discretizer
        self.similarity_discretizer = similarity_discretizer
        self.size_discretizer = size_discretizer
        self.volume_discretizer = volume_discretizer

        self.vectorizer = vectorizer
        self.improve = improve
        self.k = k
        self.n_estimators = n_estimators
        self.vecs = None
        self.reference_size = None
        self.class_estimator = None
        self.nn_estimator = None

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        graphs = pos_graphs + neg_graphs
        self.y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        self.vecs = self.vectorizer.transform(graphs)

        self._fit_class_objective(self.vecs, self.y)
        self._fit_similarity_objective(self.vecs)
        self._fit_size_objective(graphs)
        self._fit_volume_objective(self.vecs)

        return self

    def _fit_class_objective(self, x, y):
        self.estimators = []
        for i in range(self.n_estimators):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.5, random_state=i)
            sgd = SGDClassifier(average=True,
                                class_weight='balanced',
                                shuffle=True,
                                n_jobs=1)
            self.estimators.append(sgd.fit(x_train, y_train))

    def _estimate_class_objective(self, x):
        scores = [estimator.decision_function(x)
                  for estimator in self.estimators]
        scores = np.vstack(scores).T
        avg_scores = np.mean(scores, axis=1)
        std_scores = np.std(scores, axis=1)
        if self.improve is False:
            avg_scores = - np.absolute(avg_scores)
        avg_scores = (avg_scores * self.class_discretizer).astype(int)
        std_scores = (std_scores * self.class_discretizer).astype(int)
        return avg_scores, std_scores

    def _fit_similarity_objective(self, x):
        pass

    def _estimate_similarity_objective(self, x):
        similarity_matrix = cosine_similarity(x, self.vecs)
        sim = np.mean(similarity_matrix, axis=1)
        sim = (sim * self.similarity_discretizer).astype(int)
        return sim

    def _fit_size_objective(self, graphs):
        self.reference_size = np.percentile([len(g) for g in graphs], 50)

    def _estimate_size_objective(self, graphs):
        sizes = np.array([len(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        size_diffs = (size_diffs * self.size_discretizer).astype(int)
        return size_diffs

    def _fit_volume_objective(self, x):
        self.nn_estimator = NearestNeighbors()
        self.nn_estimator = self.nn_estimator.fit(x)

    def _estimate_volume_objective(self, x):
        distances, neighbors = self.nn_estimator.kneighbors(x, self.k)
        vols = np.mean(distances, axis=1)
        vols = (vols * self.volume_discretizer).astype(int)
        return vols

    def compute(self, graphs):
        """compute."""
        x = self.vectorizer.transform(graphs)
        class_costs, class_variance_costs = self._estimate_class_objective(x)
        class_costs = - class_costs.reshape(-1, 1)
        class_variance_costs = - class_variance_costs.reshape(-1, 1)
        sim_costs = self._estimate_similarity_objective(x).reshape(-1, 1)
        size_costs = self._estimate_size_objective(graphs).reshape(-1, 1)
        volume_costs = - self._estimate_volume_objective(x).reshape(-1, 1)
        costs = np.hstack([class_costs,
                           class_variance_costs,
                           sim_costs,
                           size_costs,
                           volume_costs])
        return costs

# -----------------------------------------------------------------------------


class MultiObjectiveOptimizer(object):
    """DistanceOptimizer."""

    def __init__(
            self,
            vectorizer=None,
            grammar=None,
            cost_estimator=None,
            max_neighborhood_order=1,
            max_n_iter=100):
        """init."""
        self.vec = vectorizer
        self.grammar = grammar
        self.cost_estimator = cost_estimator
        self.max_neighborhood_order = max_neighborhood_order
        self.max_n_iter = max_n_iter

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.cost_estimator.fit_local(reference_graphs, desired_distances)

    def sample(self, reference_graphs):
        """Optimize iteratively."""
        # setup
        #iteration_step_ = curry(self._iteration_step)(
        #    self.grammar, self.cost_estimator, self.max_neighborhood_order)
        iteration_step_ = lambda x: self._iteration_step(self.grammar,self.cost_estimator,
                self.max_neighborhood_order,x)
        state_dict = dict()
        state_dict['visited'] = set()
        n_neigh_steps = 1
        start_graphs = self.grammar.set_neighborhood(reference_graphs)
        costs = self.cost_estimator.compute(start_graphs)
        state_dict['pareto_set'] = get_pareto_set(start_graphs, costs)
        seed_graph = random.choice(state_dict['pareto_set'])
        arg = (seed_graph, state_dict, n_neigh_steps)
        # iterate
        arg = last(islice(iterate(iteration_step_, arg), self.max_n_iter))
        seed_graph, state_dict, n_neigh_steps = arg
        pareto_set_graphs = state_dict['pareto_set']
        return pareto_set_graphs

    def _iteration_step(
            self,
            grammar,
            cost_estimator,
            max_neighborhood_order,
            arg):
        """iteration_step."""
        graph, state_dict, n_neigh_steps = arg
        if graph is None:
            return arg
        state_dict['visited'].add(graph)
        graphs = grammar.iterated_neighborhood(graph)
        if graphs:
            graphs = graphs + state_dict['pareto_set']
            costs = cost_estimator.compute(graphs)
            state_dict['pareto_set'] = get_pareto_set(graphs, costs)
        eligible_graphs = [g for g in state_dict['pareto_set']
                           if g not in state_dict['visited']]
        if len(eligible_graphs) > 0:
            new_graph = random.choice(eligible_graphs)
            n_neigh_steps = max(n_neigh_steps - 1, 1)
        else:
            if n_neigh_steps + 1 > max_neighborhood_order:
                new_graph = None
                n_neigh_steps = 1
            else:
                new_graph = random.choice(state_dict['pareto_set'])
                n_neigh_steps += 1
        arg = (new_graph, state_dict, n_neigh_steps)
        log_iteration_step(arg)
        return arg



def log_iteration_step(arg):
    if logger.level <= 10:
        logfilename= 'pareto_arg_log'
        if not os.path.isfile(logfilename):
            logger.log(10,"logging into %s" % logfilename)
        with open(logfilename, "a") as f:
            f.write(dill.dumps(arg)+"#####")

def showlog():
    import graphlearn01.utils.draw as draw
    with open("pareto_arg_log","r")  as f:
        for arg in [dill.loads(e) for e in f.read().split("#####")[:-1]]:
                draw.graphlearn(arg[1]['pareto_set'])





