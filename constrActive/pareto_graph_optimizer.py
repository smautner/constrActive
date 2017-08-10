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
        self.improve = improve

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        sgd = SGDClassifier(average=True,
                            class_weight='balanced',
                            shuffle=True,
                            n_jobs=1)
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vectorizer.transform(graphs)
        self.estimator = sgd.fit(x, y)
        return self

    def fit_local(self, reference_graphs=None, desired_distances=None):
        """fit_local."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        self.reference_scores = self.score(reference_graphs)
        reference_sizes = [len(g) for g in reference_graphs]
        self.reference_size = np.percentile(reference_sizes, 50)
        return self

    def score(self, graphs):
        """score."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = - np.absolute(scores)
        return scores

    def predict_quality(self, graphs):
        """predict_quality."""
        scores = self.score(graphs)
        qualities = []
        for score in scores:
            to_be_ranked = np.hstack([score, self.reference_scores])
            ranked = rankdata(-to_be_ranked, method='min')
            qualities.append(ranked[0])
        return np.array(qualities)

    def discrepancy(self, vector):
        """discrepancy."""
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        return np.mean(np.square(distances - self.desired_distances))

    def predict_distance(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        return np.array([self.discrepancy(vec) for vec in x])

    def predict_size(self, graphs):
        """predict_size."""
        sizes = np.array([len(g) for g in graphs])
        return np.absolute(sizes - self.reference_size)

    def compute(self, graphs):
        """predict."""
        assert(graphs), 'moce'
        q = self.predict_quality(graphs).reshape(-1, 1)
        d = self.predict_distance(graphs).reshape(-1, 1)
        s = self.predict_size(graphs).reshape(-1, 1)
        costs = np.hstack([q, d, s])
        return costs

# -----------------------------------------------------------------------------


class DiversityMultiObjectiveCostEstimator(object):
    """DiversityMultiObjectiveCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.improve = improve

    def set_params(self, reference_graphs=None, all_graphs=None):
        """set_params."""
        self.all_vecs = self.vectorizer.transform(all_graphs)
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        self.reference_scores = self.score(reference_graphs)
        reference_sizes = [len(g) for g in reference_graphs]
        self.reference_size = np.percentile(reference_sizes, 50)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        sgd = SGDClassifier(average=True,
                            class_weight='balanced',
                            shuffle=True,
                            n_jobs=1)
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vectorizer.transform(graphs)
        self.estimator = sgd.fit(x, y)
        return self

    def score(self, graphs):
        """score."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = - np.absolute(scores)
        return scores

    def predict_quality(self, graphs):
        """predict_quality."""
        scores = self.score(graphs)
        qualities = []
        for score in scores:
            to_be_ranked = np.hstack([score, self.reference_scores])
            ranked = rankdata(-to_be_ranked, method='min')
            qualities.append(ranked[0])
        return np.array(qualities)

    def predict_similarity(self, graphs):
        """predict_similarity."""
        x = self.vectorizer.transform(graphs)
        similarity_matrix = cosine_similarity(x, self.all_vecs)
        return np.mean(similarity_matrix, axis=1)

    def predict_size(self, graphs):
        """predict_size."""
        sizes = np.array([len(g) for g in graphs])
        return np.absolute(sizes - self.reference_size)

    def compute(self, graphs):
        """compute."""
        assert(graphs), 'dmoce'
        q = self.predict_quality(graphs).reshape(-1, 1)
        k = self.predict_similarity(graphs).reshape(-1, 1)
        s = self.predict_size(graphs).reshape(-1, 1)
        costs = np.hstack([q, k, s])
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
                x, y, test_size=0.33, random_state=i)
            sgd = SGDClassifier(average=True,
                                class_weight='balanced',
                                shuffle=True,
                                n_jobs=1)
            self.estimators.append(sgd.fit(x_train, y_train))

    def _compute_class_objective(self, x):
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

    def _compute_similarity_objective(self, x):
        similarity_matrix = cosine_similarity(x, self.vecs)
        sim = np.mean(similarity_matrix, axis=1)
        sim = (sim * self.similarity_discretizer).astype(int)
        return sim

    def _fit_size_objective(self, graphs):
        self.reference_size = np.percentile([len(g) for g in graphs], 50)

    def _compute_size_objective(self, graphs):
        sizes = np.array([len(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        size_diffs = (size_diffs * self.size_discretizer).astype(int)
        return size_diffs

    def _fit_volume_objective(self, x):
        self.nn_estimator = NearestNeighbors()
        self.nn_estimator = self.nn_estimator.fit(x)

    def _compute_volume_objective(self, x):
        distances, neighbors = self.nn_estimator.kneighbors(x, self.k)
        vols = np.mean(distances, axis=1)
        vols = (vols * self.volume_discretizer).astype(int)
        return vols

    def compute(self, graphs):
        """compute."""
        x = self.vectorizer.transform(graphs)
        class_costs, class_variance_costs = self._compute_class_objective(x)
        class_costs = - class_costs.reshape(-1, 1)
        class_variance_costs = - class_variance_costs.reshape(-1, 1)
        similarity_costs = self._compute_similarity_objective(x).reshape(-1, 1)
        size_costs = self._compute_size_objective(graphs).reshape(-1, 1)
        volume_costs = - self._compute_volume_objective(x).reshape(-1, 1)
        costs = np.hstack([class_costs,
                           class_variance_costs,
                           similarity_costs,
                           size_costs,
                           volume_costs])
        return costs

# -----------------------------------------------------------------------------


def get_pareto_set(items, costs, return_costs=False):
    """get_pareto_set."""
    def _remove_duplicates(costs, items):
        dedup_costs = []
        dedup_items = []
        costs = [tuple(c) for c in costs]
        prev_c = None
        for c, g in sorted(zip(costs, items)):
            if prev_c != c:
                dedup_costs.append(c)
                dedup_items.append(g)
                prev_c = c
        return np.array(dedup_costs), dedup_items

    def _is_pareto_efficient(costs):
        is_eff = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                is_eff[i] = False
                # Remove dominated points
                is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1)
                is_eff[i] = True
        return is_eff

    def _pareto_front(costs):
        return [i for i, p in enumerate(_is_pareto_efficient(costs)) if p]

    def _pareto_set(items, costs, return_costs=False):
        ids = _pareto_front(costs)
        select_items = [items[i] for i in ids]
        if return_costs:
            select_costs = np.array([costs[i] for i in ids])
            return select_items, select_costs
        else:
            return select_items

    costs, items = _remove_duplicates(costs, items)
    return _pareto_set(items, costs, return_costs)


def iteration_step(grammar,
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
    return arg


def iterative_optimize(reference_graphs,
                       grammar,
                       cost_estimator,
                       max_neighborhood_order,
                       max_n_iter):
    """iterative_optimize."""
    # setup
    iteration_step_ = curry(iteration_step)(
        grammar, cost_estimator, max_neighborhood_order)
    state_dict = dict()
    state_dict['visited'] = set()
    n_neigh_steps = 1
    start_graphs = grammar.set_neighborhood(reference_graphs)
    costs = cost_estimator.compute(start_graphs)
    state_dict['pareto_set'] = get_pareto_set(start_graphs, costs)
    seed_graph = random.choice(state_dict['pareto_set'])
    arg = (seed_graph, state_dict, n_neigh_steps)
    arg = last(islice(iterate(iteration_step_, arg), max_n_iter))
    seed_graph, state_dict, n_neigh_steps = arg
    return state_dict['pareto_set']


def optimize_desired_distances(start_graph,
                               desired_distances,
                               reference_graphs,
                               vectorizer,
                               grammar,
                               cost_estimator,
                               max_neighborhood_order=1,
                               max_n_iter=50):
    """optimize_desired_distances."""
    cost_estimator.fit_local(reference_graphs, desired_distances)
    pareto_set_graphs = iterative_optimize(reference_graphs,
                                           grammar,
                                           cost_estimator,
                                           max_neighborhood_order,
                                           max_n_iter)
    return pareto_set_graphs
