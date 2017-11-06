#!/usr/bin/env python
"""Provides utilities for loadind datasets."""

import numpy as np
from sklearn import datasets
from eden.util import read
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import collections


def _select_targets(y, min_threshold=10, max_threshold=None):
    """Return the set of targets that are occurring a number of times.

    bounded by min_threshold and max_threshold.
    """
    c = collections.Counter(y)
    y_sel = []
    for y_id in c:
        if c[y_id] > min_threshold:
            if max_threshold:
                if c[y_id] < max_threshold:
                    y_sel.append(y_id)
            else:
                y_sel.append(y_id)
    return y_sel


def _filter_dataset(data_matrix, y, y_sel):
    """Filter data matrix and target vector.

    Selecting only instances that belong to y_sel.
    """
    targets = []
    instances = []
    for target, instance in zip(y, data_matrix):
        if target in y_sel:
            targets.append(target)
            instances.append(instance)
    y = np.array(np.hstack(targets))
    data_matrix = np.array(np.vstack(instances))
    return data_matrix, y


def _load_iris():
    print('iris')
    return datasets.load_iris(return_X_y=True)


def _load_breast_cancer():
    print('breast_cancer')
    return datasets.load_breast_cancer(return_X_y=True)


def _load_boston():
    print('boston')
    return datasets.load_boston(return_X_y=True)


def _load_diabetes():
    print('diabetes')
    return datasets.load_diabetes(return_X_y=True)


def _load_wine():
    print('wine')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    M = []
    labels = []
    for line in read(uri):
        line = line.strip()
        if line:
            items = line.split(',')
            label = int(items[0])
            labels.append(label)
            data = [float(x) for x in items[1:]]
            M.append(data)

    X = scale(np.array(M))
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)
    return X, y


def _load_ionosphere():
    print('ionosphere')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
    n_max = 700

    M = []
    labels = []
    counter = 0
    for line in read(uri):
        counter += 1
        if counter > n_max:
            break
        line = line.strip()
        if line:
            items = line.split(',')
            label = hash(items[-1])
            labels.append(label)
            data = [float(x) for x in items[:-1]]
            M.append(data)

    X = (np.array(M))
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)
    return X, y


def _load_abalone():
    print('abalone')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    n_max = 700

    M = []
    labels = []
    counter = 0
    for line in read(uri):
        counter += 1
        if counter > n_max:
            break
        line = line.strip()
        if line:
            items = line.split(',')
            label = int(items[-1]) // 7
            labels.append(label)
            data = [float(x) for x in items[1:-1]]
            M.append(data)

    X = np.array(M)
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)

    y_sel = _select_targets(y, min_threshold=5)
    X, y = _filter_dataset(X, y, y_sel)

    return X, y


def _load_pima():
    print('pima')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    n_max = 700

    M = []
    labels = []
    counter = 0
    for line in read(uri):
        counter += 1
        if counter > n_max:
            break
        line = line.strip()
        if line:
            items = line.split(',')
            label = hash(items[-1])
            labels.append(label)
            data = [float(x) for x in items[:-1]]
            M.append(data)

    X = (np.array(M))
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)
    return X, y


def _load_biodeg():
    print('biodeg')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
    n_max = 700

    M = []
    labels = []
    counter = 0
    for line in read(uri):
        counter += 1
        if counter > n_max:
            break
        line = line.strip()
        if line:
            items = line.split(';')
            label = hash(items[-1])
            labels.append(label)
            data = [float(x) for x in items[:-1]]
            M.append(data)

    X = (np.array(M))
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)
    return X, y


def _load_vehicle():
    print('vehicle')
    n_max = 1500

    def _load_data(uri):
        M = []
        labels = []
        counter = 0
        for line in read(uri):
            counter += 1
            if counter > n_max:
                break
            line = line.strip()
            if line:
                items = line.split(' ')
                label = hash(items[-1]) & 13
                labels.append(label)
                data = [float(x) for x in items[:-1]]
                M.append(data)
        X = np.array(M)
        y = np.array(labels)
        return X, y

    for i, c in enumerate('abcdefghi'):
        uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xa%s.dat' % c
        X_, y_ = _load_data(uri)
        if i == 0:
            X = X_
            y = y_
        else:
            X = np.vstack((X, X_))
            y = np.hstack((y, y_))

    y_sel = _select_targets(y, min_threshold=5)
    X, y = _filter_dataset(X, y, y_sel)
    return X, y


def _load_wdbc():
    print('breast-cancer-wisconsin')
    uri = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    from eden.util import read
    M = []
    labels = []
    for line in read(uri):
        line = line.strip()
        if line:
            items = line.split(',')
            label = str(items[1])
            labels.append(label)
            data = [float(x) for x in items[2:]]
            M.append(data)

    import numpy as np
    from sklearn.preprocessing import normalize, scale
    X = scale(np.array(M))
    from sklearn.preprocessing import LabelEncoder
    targets = LabelEncoder().fit_transform(labels)
    y = np.array(targets)
    return X, y


def _load_digits():
    print('digits')
    return datasets.load_digits(n_class=7, return_X_y=True)
