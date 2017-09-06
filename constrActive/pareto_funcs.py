#!/usr/bin/env python
"""Provides Pareto optimization functions."""

import numpy as np


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


def get_pareto_set(items, costs, return_costs=False):
    """get_pareto_set."""
    costs, items = _remove_duplicates(costs, items)
    return _pareto_set(items, costs, return_costs)
