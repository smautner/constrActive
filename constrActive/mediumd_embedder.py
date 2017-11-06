#!/usr/bin/env python
"""Provides embedding for high dimensional data to medium dim."""

from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from scipy.sparse import issparse


class MediumEmbedder(object):
    """MediumEmbedder."""

    def __init__(self, dim=10):
        """init."""
        estimator = SGDClassifier(
            average=True, class_weight='balanced', shuffle=True)
        self.feature_selector = RFE(
            estimator,
            step=.5,
            n_features_to_select=dim)

    def transform(self, data):
        """transform."""
        x = self.feature_selector.transform(data)
        if issparse(x):
            x = x.toarray()
        return x

    def fit(self, data, target):
        """fit."""
        self.feature_selector.fit(data, target)
        return self
