#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import numpy as np
from toolz.itertoolz import iterate, last
from toolz.curried import pipe, map, concat, curry
from itertools import islice
import random
from scipy.stats import rankdata
from graphlearn.utils.neighbors import graph_neighbors
from graphlearn.graphlearn import Sampler
from graphlearn.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar as lsgg
# from graphlearn.lsgg import lsgg
from sklearn.linear_model import SGDClassifier
from eden.graph import _revert_edge_to_vertex_transform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

import logging
logger = logging.getLogger(__name__)


class NullEstimator():

    def __init__(self):
        pass

    def fit(self, arg, **args):
        pass

    def predict(self, vector):
        pass

# -----------------------------------------------------------------------------


class GrammarWrapper(object):

    def __init__(self,
                 vectorizer,
                 radius_list=[0, 1],
                 thickness_list=[1],
                 min_cip_count=4,
                 min_interface_count=2):
        sampler = Sampler(estimator=NullEstimator(),
                          vectorizer=vectorizer,
                          grammar=lsgg(
            radius_list=radius_list,
            thickness_list=thickness_list,
            min_cip_count=min_cip_count,
            min_interface_count=min_interface_count))
        self.sampler = sampler

    def __repr__(self):
        n, i, c, p = self.sampler.grammar().size()
        return '#instances: %5d   #interfaces: %5d   #cores: %5d   #core-interface-pairs: %5d' % (n, i, c, p)

    def fit(self, graphs):
        self.sampler.fit(graphs)
        return self

    def neighborhood(self, graph):
        grammar = self.sampler.lsgg
        tr = self.sampler.graph_transformer
        dec = self.sampler.decomposer.make_new_decomposer(
            tr.re_transform_single(graph))
        neighbors = [_revert_edge_to_vertex_transform(g._base_graph)
                     for g in graph_neighbors(dec, grammar, tr)]
        return neighbors

# -----------------------------------------------------------------------------


class GrammarWrapper2(object):

    def __init__(self,
                 vectorizer,
                 radius_list=[0, 1],
                 thickness_list=[1],
                 min_cip_count=4,
                 min_interface_count=2):
        self.grammar = lsgg(
            decompositionargs=dict(
                radius_list=radius_list,
                thickness_list=thickness_list,
                hash_bitmask=2**16 - 1),
            filterargs=dict(
                min_cip_count=min_cip_count,
                min_interface_count=min_interface_count)
        )

    def __repr__(self):
        return '0'

    def fit(self, graphs):
        self.grammar.fit(graphs)
        return self

    def neighborhood(self, graph):
        return list(self.grammar.neighbors(graph))
# -----------------------------------------------------------------------------


class MultiObjectiveCostEstimator(object):

    def __init__(self, vectorizer, improve=True):
        self.vectorizer = vectorizer
        self.improve = improve

    def set_params(self, reference_graphs=None, desired_distances=None):
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        self.reference_scores = self.score(reference_graphs)
        reference_sizes = [len(g) for g in reference_graphs]
        self.reference_size = np.percentile(reference_sizes, 50)

    def fit(self, pos_graphs, neg_graphs):
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
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = 1 / (1 + np.absolute(scores))
        return scores

    def predict_quality(self, graphs):
        scores = self.score(graphs)
        qualities = []
        for score in scores:
            to_be_ranked = np.hstack([score, self.reference_scores])
            ranked = rankdata(-to_be_ranked, method='min')
            qualities.append(ranked[0])
        return np.array(qualities)

    def discrepancy(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        return np.mean(np.square(distances - self.desired_distances))

    def predict_distance(self, graphs):
        x = self.vectorizer.transform(graphs)
        return np.array([self.discrepancy(vec) for vec in x])

    def predict_size(self, graphs):
        sizes = np.array([len(g) for g in graphs])
        return np.absolute(sizes - self.reference_size)

    def predict(self, graphs):
        assert(graphs), 'moce'
        q = self.predict_quality(graphs).reshape(-1, 1)
        d = self.predict_distance(graphs).reshape(-1, 1)
        s = self.predict_size(graphs).reshape(-1, 1)
        costs = np.hstack([q, d, s])
        return costs

# -----------------------------------------------------------------------------


class DiversityMultiObjectiveCostEstimator(object):

    def __init__(self, vectorizer, improve=True):
        self.vectorizer = vectorizer
        self.improve = improve

    def set_params(self, reference_graphs=None, all_graphs=None):
        self.all_vecs = self.vectorizer.transform(all_graphs)
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        self.reference_scores = self.score(reference_graphs)
        reference_sizes = [len(g) for g in reference_graphs]
        self.reference_size = np.percentile(reference_sizes, 50)

    def fit(self, pos_graphs, neg_graphs):
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
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = 1 / (1 + np.absolute(scores))
        return scores

    def predict_quality(self, graphs):
        scores = self.score(graphs)
        qualities = []
        for score in scores:
            to_be_ranked = np.hstack([score, self.reference_scores])
            ranked = rankdata(-to_be_ranked, method='min')
            qualities.append(ranked[0])
        return np.array(qualities)

    def predict_similarity(self, graphs):
        x = self.vectorizer.transform(graphs)
        similarity_matrix = cosine_similarity(x, self.all_vecs)
        return np.mean(similarity_matrix, axis=1)

    def predict_size(self, graphs):
        sizes = np.array([len(g) for g in graphs])
        return np.absolute(sizes - self.reference_size)

    def predict(self, graphs):
        assert(graphs), 'dmoce'
        q = self.predict_quality(graphs).reshape(-1, 1)
        k = self.predict_similarity(graphs).reshape(-1, 1)
        s = self.predict_size(graphs).reshape(-1, 1)
        costs = np.hstack([q, k, s])
        return costs

# -----------------------------------------------------------------------------


def get_pareto_set(items, costs):

    def remove_duplicates(costs, items):
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

    def is_pareto_efficient(costs):
        is_eff = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                is_eff[i] = False
                # Remove dominated points
                is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1)
                is_eff[i] = True
        return is_eff

    def pareto_front(costs):
        return [i for i, p in enumerate(is_pareto_efficient(costs)) if p]

    def pareto_set(items, costs):
        ids = pareto_front(costs)
        return [items[i] for i in ids]

    costs, items = remove_duplicates(costs, items)
    return pareto_set(items, costs)


def random_sample(max_size, graphs):
    if len(graphs) > max_size:
        graphs = random.sample(graphs, max_size)
    return graphs


def set_neighborhood(graphs, grammar=None, max_neighborhood_size=1000):
    random_sample_ = curry(random_sample)(max_neighborhood_size)
    graphs = pipe(graphs, map(grammar.neighborhood), concat, list)
    graphs = random_sample_(graphs)
    return graphs


def iterated_neighborhood(graph, grammar,
                          n_neigh_steps=1, max_neighborhood_size=1000):
    set_neighborhood_ = curry(set_neighborhood)(
        grammar=grammar,
        max_neighborhood_size=max_neighborhood_size)
    return last(islice(iterate(set_neighborhood_, [graph]), n_neigh_steps + 1))


def iteration_step(grammar, cost_estimator,
                   max_neighborhood_order, max_neighborhood_size, arg):
    graph, state_dict, n_neigh_steps = arg
    if graph is None:
        return arg
    state_dict['visited'].add(graph)
    graphs = iterated_neighborhood(graph, grammar,
                                   n_neigh_steps, max_neighborhood_size)
    if graphs:
        graphs = graphs + state_dict['pareto_set']
        costs = cost_estimator.predict(graphs)
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
                       max_neighborhood_size,
                       max_n_iter):
    # setup
    iteration_step_ = curry(iteration_step)(
        grammar, cost_estimator, max_neighborhood_order, max_neighborhood_size)
    state_dict = dict()
    state_dict['visited'] = set()
    n_neigh_steps = 1
    start_graphs = set_neighborhood(reference_graphs,
                                    grammar,
                                    max_neighborhood_size)
    costs = cost_estimator.predict(start_graphs)
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
                               max_neighborhood_size=100,
                               max_n_iter=50):
    cost_estimator.set_params(reference_graphs, desired_distances)
    pareto_set_graphs = iterative_optimize(reference_graphs,
                                           grammar,
                                           cost_estimator,
                                           max_neighborhood_order,
                                           max_neighborhood_size,
                                           max_n_iter)
    graphs_tuple = (reference_graphs, pareto_set_graphs)
    return graphs_tuple


def diversity_pareto_selection(sel_seed_graphs,
                               graphs_tuple_list,
                               pos_graphs,
                               neg_graphs,
                               vectorizer,
                               improve):
    union_pareto_sets = []
    msg = 'pareto set sizes: '
    for graphs_tuple in graphs_tuple_list:
        reference_graphs, pareto_set_graphs = graphs_tuple
        msg += str(len(pareto_set_graphs)) + ' '
        union_pareto_sets += pareto_set_graphs
    logger.info(msg)

    diversity_cost_estimator = DiversityMultiObjectiveCostEstimator(
        vectorizer, improve)
    diversity_cost_estimator.fit(pos_graphs, neg_graphs)
    diversity_cost_estimator.set_params(reference_graphs=sel_seed_graphs,
                                        all_graphs=pos_graphs + neg_graphs)
    costs = diversity_cost_estimator.predict(union_pareto_sets)
    pareto_set_graphs = get_pareto_set(union_pareto_sets, costs)
    logger.info('result set size: %d' % len(pareto_set_graphs))
    return pareto_set_graphs, sel_seed_graphs
