"""Evaluation metrics and paired statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


def accuracy(z, y):
    return float((z.argmax(axis=1) == y).mean())


def top_k_accuracy(z, y, k=3):
    top = np.argsort(-z, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def macro_f1(z, y):
    return float(f1_score(y, z.argmax(axis=1), average="macro", labels=np.unique(y)))


def ece(p, y, n_bins=10):
    """Top-label expected calibration error with equal-width bins."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))


def brier(p, y):
    """Multi-class Brier score: mean over cases of sum_k (p_k - 1[y=k])^2."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def reliability_bins(p, y, n_bins=10):
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return rows


def paired_test(a, b):
    """Wilcoxon signed-rank test on paired per-repetition scores.

    Returns (mean difference a - b, p-value). If all differences are zero the
    p-value is 1.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return float(d.mean()), 1.0
    return float(d.mean()), float(wilcoxon(a, b, zero_method="zsplit").pvalue)
