"""Ranking metrics. Each takes a ranked list and a set of relevant ids."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    return len(set(ranked[:k]) & relevant) / min(len(relevant), k)


def ndcg_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    gains = [1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal)


def mean_metric(fn, ranked_lists, relevant_sets, k) -> float:
    vals = [fn(r, s, k) for r, s in zip(ranked_lists, relevant_sets)]
    return float(np.nanmean(vals))


def gauc(users: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Impression-weighted mean of per-user AUC, over users with both classes."""
    num = den = 0.0
    order = np.argsort(users, kind="stable")
    u, y, p = users[order], y[order], p[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    for yy, pp in zip(np.split(y, bounds), np.split(p, bounds)):
        if 0 < yy.sum() < len(yy):
            num += roc_auc_score(yy, pp) * len(yy)
            den += len(yy)
    return num / den if den else float("nan")
