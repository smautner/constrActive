#!/usr/bin/env python
"""Provides wrapper for optimizer."""

import numpy as np


class FireflyOptimizer(object):
    """FO."""

    def __init__(
            self,
            objective=None,
            size=20,
            n_iter=10,
            gamma=None,
            alpha=None):
        """init."""
        self.objective = objective
        self.size = size
        self.n_iter = n_iter
        self.gamma = gamma
        self.alpha = alpha

    def fit(
            self,
            lower_limits=None,
            upper_limits=None,
            lower_limit=0,
            upper_limit=1,
            n_dimensions=2):
        """fit."""
        if lower_limits is None:
            lower_limits = [lower_limit] * n_dimensions
        if upper_limits is None:
            upper_limits = [upper_limit] * n_dimensions
        lower_limits = np.array(lower_limits).reshape(-1, 1)
        upper_limits = np.array(upper_limits).reshape(-1, 1)
        scale = np.max(upper_limits - lower_limits)
        if self.gamma is None:
            self.gamma = scale
        if self.alpha is None:
            self.alpha = scale / 10000.0
        vals = np.random.rand(len(lower_limits), self.size)
        vals = vals * (upper_limits - lower_limits) + lower_limits
        self.fireflies = vals.T

    def optimize(self):
        """optimize."""
        insts = self._optimize(
            self.fireflies,
            self.objective,
            self.n_iter,
            self.gamma,
            self.alpha)

        ranked_vals_insts = sorted([(self.objective(x), x)
                                    for x in insts], key=lambda z: -z[0])
        vals, ranked_insts = zip(*ranked_vals_insts)
        ranked_insts = np.vstack(ranked_insts)
        return ranked_insts

    def _update_position(self, i, j, fireflies, f, gamma=0.1, alpha=0.1):
        xi = fireflies[i].reshape(-1, 1)
        xj = fireflies[j].reshape(-1, 1)
        fxj = f(xj)
        rand_x = np.random.rand(len(xi), 1) * alpha
        dist_ij = np.square(xj - xi)
        xi = xi + fxj * np.exp(-gamma * dist_ij) * (xj - xi) + rand_x
        fireflies[i] = xi.reshape(1, -1)

    def _optimize(self, fireflies, f, n_iter=10, gamma=0.1, alpha=0.1):
        for it in range(n_iter):
            for i in range(fireflies.shape[0]):
                xi = fireflies[i, :]
                fxi = f(xi)
                for j in range(fireflies.shape[0]):
                    xj = fireflies[j, :]
                    fxj = f(xj)
                    if fxj > fxi:
                        self._update_position(i, j, fireflies, f, gamma, alpha)
        return fireflies
