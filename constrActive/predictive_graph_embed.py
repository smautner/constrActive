#!/usr/bin/env python
"""Provides embedding for high dimensional data."""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import copy
import random
import pylab as plt
from constrActive.subgraph_embed import embed
from constrActive.mediumd_embedder import MediumEmbedder
from eden.util import serialize_dict
import multiprocessing
from eden.util import timeit

import logging
logger = logging.getLogger(__name__)


@timeit
def optimize(embedder_class,
             data,
             target,
             random_search_n_iter=30,
             line_search_n_iter=2,
             stop_score_threshold=.97):
    """optimize."""
    # compute performance for default configuration
    emb = embedder_class()
    params = emb.get_params()
    score, params = _evaluate(emb, data, target, params=params)
    _log_params(emb, score)
    try:
        if score > stop_score_threshold:
            raise StopIteration()
        # try random search followed by line search
        # find a good solution using random search
        curr_score, curr_params = _random_search(
            emb,
            data,
            target,
            best_score=score,
            best_params=params,
            random_search_n_iter=random_search_n_iter)
        if curr_score > stop_score_threshold:
            emb.set_params(**curr_params)
            raise StopIteration()
        if curr_score == score:
            emb.set_params(**params)
            raise StopIteration()
        if curr_score > score:
            score = curr_score
            params = curr_params
        emb.set_params(**params)
        _log_params(emb, score)
        # refine the previous result using a line search
        curr_score, curr_params = _line_search(
            emb,
            data,
            target,
            best_score=score,
            best_params=params,
            line_search_n_iter=line_search_n_iter,
            stop_score_threshold=stop_score_threshold)
        emb.set_params(**params)
        if curr_score > stop_score_threshold:
            emb.set_params(**curr_params)
            raise StopIteration()
        if curr_score == score:
            emb.set_params(**params)
            raise StopIteration()
        if curr_score > score:
            score = curr_score
            params = curr_params
        emb.set_params(**params)
        _log_params(emb, score)
    except StopIteration:
        pass
    emb.fit(data, target)
    return emb


def _log_params(emb, score):
    pstr = emb._repr_params(emb.get_params())
    txt = 'AUC ROC:%.3f %s' % (score, pstr)
    logger.info(txt)


def _evaluate(emb, data, target, params=None):
    if params is not None:
        emb.set_params(**params)
    score = emb._avg_score(data, target)
    params = emb.get_params()
    return score, params


def _evaluate_parallel2(emb, data, target, params_list):
    best_score, best_params = max([_evaluate(emb, data, target, p)
                                   for p in params_list])
    return best_score, best_params


def _evaluate_parallel(emb, data, target, params_list):
    pool = multiprocessing.Pool()
    res = [pool.apply_async(_evaluate, args=(emb, data, target, p))
           for p in params_list]
    best_score, best_params = max([p.get() for p in res])
    pool.close()
    pool.join()
    return best_score, best_params


def _random_search(
        emb,
        data,
        target,
        best_score=0,
        best_params=None,
        random_search_n_iter=20):
    params_list = [emb._params_random_choice()
                   for i in range(random_search_n_iter)]
    score, params = _evaluate_parallel(emb, data, target, params_list)
    if score > best_score:
        return score, params
    else:
        return best_score, best_params


def _line_search(
        emb,
        data,
        target,
        best_score=0,
        best_params=None,
        line_search_n_iter=2,
        stop_score_threshold=.97):
    curr_params = best_params
    curr_score = best_score
    try:
        for i in range(line_search_n_iter):
            prev_best_score = curr_score
            for param in emb.params_range:
                if len(emb.params_range[param]) > 1:
                    params_list = []
                    params = copy.deepcopy(curr_params)
                    for val in emb.params_range[param]:
                        if params[param] != val:
                            params[param] = val
                            params_list.append(copy.deepcopy(params))
                    loc_best_score, loc_best_params = _evaluate_parallel(
                        emb,
                        data,
                        target,
                        params_list)
                    if loc_best_score > curr_score:
                        curr_params = loc_best_params
                        curr_score = loc_best_score
                        pstr = emb._repr_params(loc_best_params)
                        txt = 'AUC ROC:%.3f %s' % (loc_best_score, pstr)
                        logger.info(txt)
                    if curr_score > stop_score_threshold:
                        raise StopIteration()
            if curr_score > prev_best_score:
                pass
            else:
                raise StopIteration()
    except StopIteration:
        pass
    if curr_score > best_score:
        return curr_score, curr_params
    else:
        return best_score, best_params


class Biased2DAveragedClassifier(object):
    """Provide classification in 2D."""

    def __init__(
            self,
            negative_bias=1,
            n_estimators=5,
            n_neighbors=5,
            weights='distance',
            p=2):
        """init."""
        self.negative_bias = negative_bias
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.estimators = []

    def fit(self, data, target):
        """fit."""
        for i in range(self.n_estimators):
            tr_data, ts_data, tr_target, ts_target = train_test_split(
                data, target, test_size=0.33, random_state=42 + i)
            biased_data, biased_target = self._add_negative_bias(
                tr_data,
                tr_target,
                negative_bias=self.negative_bias)
            estimator = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                p=self.p).fit(biased_data, biased_target)
            self.estimators.append(estimator)
        return self

    def predict_proba(self, data):
        """predict_proba."""
        scores = []
        for estimator in self.estimators:
            y_score = estimator.predict_proba(data)
            if y_score.shape[1] > 1:
                y_score = y_score[:, 1]
            scores.append(y_score.reshape(-1, 1))
        scores = np.hstack(scores)
        score = np.mean(scores, axis=1)
        return score

    def _add_negative_bias(self, data2d, target, negative_bias=1.0):
        if negative_bias == 0:
            return data2d, target
        x_min, x_max = data2d[:, 0].min(), data2d[:, 0].max()
        y_min, y_max = data2d[:, 1].min(), data2d[:, 1].max()
        dim = int(np.sqrt(len(data2d)) * negative_bias)
        x, step_x = np.linspace(x_min, x_max, dim, retstep=True)
        y, step_y = np.linspace(y_min, y_max, dim, retstep=True)

        def jiggle(val, step):
            return val + np.random.rand() * step - step / 2
        biased_negdata2d = np.array([[jiggle(i, step_x), jiggle(j, step_y)]
                                     for i in x for j in y])
        biased_data2d = np.vstack([data2d, biased_negdata2d])
        bias_neg_target = np.zeros((len(biased_negdata2d), 1))
        target = target.reshape(-1, 1)
        biased_target = np.vstack([target, bias_neg_target])
        biased_target = np.ravel(biased_target)
        return biased_data2d, biased_target


class PredictiveGraphEmbedder(object):
    """Provide 2D embedding."""

    def __init__(
            self,
            n_estimators=250,
            medium_dim=100,
            nn_n_estimators=30,
            nn_negative_bias=1,
            nn_k=7,
            nn_p=2,
            emb_iter=10,
            emb_confidence=2,
            emb_sample_fraction=0.6,
            emb_feature_fraction=1,
            emb_len_quant=0.5,
            emb_alpha=1,
            emb_gamma=1,
            emb_beta=30):
        """init."""
        self.set_params(
            n_estimators,
            medium_dim,
            nn_n_estimators,
            nn_negative_bias,
            nn_k,
            nn_p,
            emb_iter,
            emb_confidence,
            emb_sample_fraction,
            emb_feature_fraction,
            emb_len_quant,
            emb_alpha,
            emb_gamma,
            emb_beta)

        self.params_range = dict(
            n_estimators=[250],
            medium_dim=[10, 25, 50, 100, 250, 500],
            nn_n_estimators=[30],
            nn_negative_bias=[1],
            nn_k=[1, 3, 5, 7, 11],
            nn_p=[2],
            emb_iter=[20],
            emb_confidence=[1.5, 2, 3],
            emb_sample_fraction=[.6, .75, 1],
            emb_feature_fraction=[.01, .05, .1, .3, .5, .7, 1],
            emb_len_quant=[50],
            emb_alpha=[0, 1, 3],
            emb_gamma=[1],
            emb_beta=[20, 30, 40])

    def get_params(self):
        """get_params."""
        return dict(
            n_estimators=self.n_estimators,
            medium_dim=self.medium_dim,
            nn_n_estimators=self.nn_n_estimators,
            nn_negative_bias=self.nn_negative_bias,
            nn_k=self.nn_k,
            nn_p=self.nn_p,
            emb_iter=self.emb_iter,
            emb_confidence=self.emb_confidence,
            emb_sample_fraction=self.emb_sample_fraction,
            emb_feature_fraction=self.emb_feature_fraction,
            emb_len_quant=self.emb_len_quant,
            emb_alpha=self.emb_alpha,
            emb_gamma=self.emb_gamma,
            emb_beta=self.emb_beta)

    def set_params(
            self,
            n_estimators=250,
            medium_dim=100,
            nn_n_estimators=30,
            nn_negative_bias=1,
            nn_k=7,
            nn_p=2,
            emb_iter=10,
            emb_confidence=2,
            emb_sample_fraction=0.6,
            emb_feature_fraction=1,
            emb_len_quant=0.5,
            emb_alpha=1,
            emb_gamma=1,
            emb_beta=30):
        """set_params."""
        self.n_estimators = n_estimators
        self.medium_dim = medium_dim
        self.nn_n_estimators = nn_n_estimators
        self.nn_negative_bias = nn_negative_bias
        self.nn_k = nn_k
        self.nn_p = nn_p
        self.emb_iter = emb_iter
        self.emb_confidence = emb_confidence
        self.emb_sample_fraction = emb_sample_fraction
        self.emb_feature_fraction = emb_feature_fraction
        self.emb_len_quant = emb_len_quant
        self.emb_alpha = emb_alpha
        self.emb_gamma = emb_gamma
        self.emb_beta = emb_beta
        # set objects
        self.est_medium_dim = MediumEmbedder(
            dim=self.medium_dim)
        self.regress2d = ExtraTreesRegressor(
            n_estimators=self.n_estimators)
        self.est2d = Biased2DAveragedClassifier(
            negative_bias=self.nn_negative_bias,
            n_estimators=self.nn_n_estimators,
            n_neighbors=self.nn_k,
            weights='distance',
            p=self.nn_p)

    def _repr_params(self, params):
        txt = ''
        for key in sorted(self.params_range):
            if len(self.params_range[key]) > 1:
                txt += '  %s:%s  ' % (key, params[key])
        return txt

    def _params_random_choice(self):
        params = dict([(key, random.choice(self.params_range[key]))
                       for key in self.params_range])
        return params

    def _avg_score(
            self,
            data,
            target,
            n_repetitions=3):
        scores = []
        for i in range(n_repetitions):
            tr_data, ts_data, tr_target, ts_target = train_test_split(
                data, target, test_size=0.33, random_state=421 + i)
            self.fit(tr_data, tr_target)
            score = self.score(ts_data, ts_target)
            scores.append(score)
        score = np.mean(score)
        return score

    def _feature_importance(self, data, target):
        ec = ExtraTreesClassifier(n_estimators=self.n_estimators)
        feature_p = ec.fit(data, target).feature_importances_
        return feature_p

    def fit(self, data, target):
        """fit."""
        tr_data, ts_data, tr_target, ts_target = train_test_split(
            data, target, test_size=0.5, random_state=42)
        self.est_medium_dim.fit(tr_data, tr_target)
        tr_data_medium = self.est_medium_dim.transform(tr_data)
        ts_data_medium = self.est_medium_dim.transform(ts_data)
        feature_p = self._feature_importance(tr_data_medium, tr_target)
        tr_data2d, graph = embed(
            tr_data_medium,
            target=tr_target,
            confidence=self.emb_confidence,
            n_iter=self.emb_iter,
            sample_fraction=self.emb_sample_fraction,
            feature_fraction=self.emb_feature_fraction,
            feature_p=feature_p,
            len_quant=self.emb_len_quant,
            alpha=self.emb_alpha,
            gamma=self.emb_gamma,
            beta=self.emb_beta)
        self.regress2d.fit(tr_data_medium, tr_data2d)
        ts_data2d = self.regress2d.predict(ts_data_medium)
        self.est2d.fit(ts_data2d, ts_target)
        return self

    def transform(self, data):
        """transform."""
        data_medium = self.est_medium_dim.transform(data)
        data_2_dim = self.regress2d.predict(data_medium)
        return data_2_dim

    def predict(self, data):
        """predict."""
        data_medium = self.est_medium_dim.transform(data)
        data_2_dim = self.regress2d.predict(data_medium)
        y_score = self.est2d.predict_proba(data_2_dim)
        return y_score

    def score(self, data, target):
        """score."""
        y_score = self.predict(data)
        auc = metrics.roc_auc_score(target, y_score)
        return auc

    def visualize(self, data, target, title='', region_only=False):
        """visualize."""
        auc = self.score(data, target)
        title += 'roc:%.2f' % (auc)
        title += '\nparams:%s' % serialize_dict(self.get_params())

        x2dim = self.transform(data)

        x_min, x_max = x2dim[:, 0].min(), x2dim[:, 0].max()
        y_min, y_max = x2dim[:, 1].min(), x2dim[:, 1].max()
        b = max((x_max - x_min) / 10, (y_max - y_min) / 10)   # border size
        x_min, x_max = x_min - b, x_max + b
        y_min, y_max = y_min - b, y_max + b
        h = b / 20  # step size in the mesh
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))

        grid2d = np.c_[xx.ravel(), yy.ravel()]
        z = self.est2d.predict_proba(grid2d)
        z = 1 - z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.get_cmap('BrBG'),
                     alpha=.3, levels=[0.05, 0.25, 0.5, 0.75, 0.95],
                     extend='both')
        plt.contour(xx, yy, z, levels=[-1, 0.5, 2], colors='w',
                    linewidths=[.5, 4, .5],
                    linestyles=['solid', 'solid', 'solid'],
                    extend='both')
        plt.contour(xx, yy, z, levels=[-1, 0.5, 2], colors='k',
                    linewidths=[.5, 2, .5],
                    linestyles=['solid', 'solid', 'solid'],
                    extend='both')
        if region_only is False:
            plt.scatter(x2dim[:, 0], x2dim[:, 1],
                        alpha=.8,
                        c=target,
                        s=30,
                        edgecolors='k',
                        cmap=plt.get_cmap('gray'))
        plt.title(title)
        plt.grid(False)
        plt.axis('off')
        return self

    def visualize_data(self, data, target=None,
                       x_min=None, x_max=None, y_min=None, y_max=None):
        """visualize_test."""
        x2dim = self.transform(data)
        if x_min is None or x_max is None or y_min is None or y_max is None:
            x_min, x_max = x2dim[:, 0].min(), x2dim[:, 0].max()
            y_min, y_max = x2dim[:, 1].min(), x2dim[:, 1].max()
        self.visualize_region(x_min, x_max, y_min, y_max)
        if target is None:
            c = 'w'
        else:
            c = target
        plt.scatter(x2dim[:, 0], x2dim[:, 1],
                    alpha=.8,
                    c=c,
                    s=30,
                    edgecolors='k',
                    cmap=plt.get_cmap('gray'))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        return self

    def visualize_region(self, x_min=None, x_max=None, y_min=None, y_max=None):
        """visualize_region."""
        b = max((x_max - x_min) / 10, (y_max - y_min) / 10)   # border size
        x_min, x_max = x_min - b, x_max + b
        y_min, y_max = y_min - b, y_max + b
        h = b / 20  # step size in the mesh
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))

        grid2d = np.c_[xx.ravel(), yy.ravel()]
        z = self.est2d.predict_proba(grid2d)
        z = 1 - z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.get_cmap('BrBG'),
                     alpha=.3, levels=[0.05, 0.25, 0.5, 0.75, 0.95],
                     extend='both')
        plt.contour(xx, yy, z, levels=[-1, 0.5, 2], colors='w',
                    linewidths=[.5, 4, .5],
                    linestyles=['solid', 'solid', 'solid'],
                    extend='both')
        plt.contour(xx, yy, z, levels=[-1, 0.5, 2], colors='k',
                    linewidths=[.5, 2, .5],
                    linestyles=['solid', 'solid', 'solid'],
                    extend='both')
