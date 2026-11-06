"""Simulated patient cohorts and imperfect knowledge bases.

The Kaggle table gives, for each disease, the symptoms that belong to it and
how often they occur (0.85-1.0 for every listed symptom). We treat that table
as the "true" presentation pattern and simulate patients who

* report only some of their typical symptoms (reporting rate ``report``), and
* may report unrelated symptoms (Poisson ``noise`` per patient, drawn from
  symptoms outside the disease profile in proportion to how common they are).

The model is not given the true table. It is given a degraded copy in which a
fraction ``gap`` of each disease's symptom associations has been deleted, which
mimics a textbook that misses atypical presentations.
"""

from __future__ import annotations

import numpy as np


class World:
    def __init__(self, Q: np.ndarray, report: float = 0.6, noise: float = 1.0,
                 min_symptoms: int = 2):
        self.Q = np.asarray(Q, dtype=float)
        self.C, self.F = self.Q.shape
        self.report = report
        self.noise = noise
        self.min_symptoms = min_symptoms
        prev = self.Q.mean(axis=0)
        self.prevalence = prev / prev.sum()
        self.profile = self.Q > 0

    # ------------------------------------------------------------------ cases
    def _one_class(self, c: int, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, self.F))
        off = np.where(~self.profile[c])[0]
        p_off = self.prevalence[off] / self.prevalence[off].sum()
        filled = 0
        while filled < n:
            m = (n - filled) * 2
            x = (rng.random((m, self.F)) < self.Q[c] * self.report).astype(float)
            if self.noise > 0:
                k = rng.poisson(self.noise, size=m)
                for i in np.where(k > 0)[0]:
                    pick = rng.choice(off, size=min(k[i], len(off)), replace=False, p=p_off)
                    x[i, pick] = 1.0
            x = x[x.sum(axis=1) >= self.min_symptoms]
            take = min(len(x), n - filled)
            out[filled:filled + take] = x[:take]
            filled += take
        return out

    def sample(self, counts, rng: np.random.Generator):
        """Draw ``counts[c]`` cases for every disease c. Returns (X, y)."""
        counts = np.broadcast_to(np.asarray(counts), (self.C,))
        X = np.vstack([self._one_class(c, int(counts[c]), rng) for c in range(self.C)])
        y = np.repeat(np.arange(self.C), counts.astype(int))
        return X, y

    # -------------------------------------------------------- knowledge base
    def knowledge_base(self, gap: float, rng: np.random.Generator) -> np.ndarray:
        """Copy of Q with a fraction ``gap`` of each disease's associations removed."""
        K = self.Q.copy()
        if gap <= 0:
            return K
        for c in range(self.C):
            prof = np.where(self.profile[c])[0]
            drop = min(int(round(gap * len(prof))), len(prof) - 1)
            if drop > 0:
                K[c, rng.choice(prof, size=drop, replace=False)] = 0.0
        return K


def nested_training_sets(world: World, max_per_class: int, rng: np.random.Generator):
    """Draw ``max_per_class`` cases per disease once; smaller sets are prefixes.

    Returns a function ``take(n_or_counts)`` giving (X, y) with the first n cases
    of every class, so learning curves within one repetition are nested.
    """
    pools = [world._one_class(c, max_per_class, rng) for c in range(world.C)]

    def take(counts):
        counts = np.broadcast_to(np.asarray(counts), (world.C,)).astype(int)
        X = np.vstack([pools[c][:counts[c]] for c in range(world.C)])
        y = np.repeat(np.arange(world.C), counts)
        return X, y

    return take


def long_tail_counts(C: int, head: int, exponent: float, rng: np.random.Generator):
    """Zipf-like training counts: the r-th most common disease gets head * r^-exponent.

    Disease ranks are shuffled so that a different set of diseases is rare in
    every repetition. Every disease keeps at least one case.
    """
    ranks = rng.permutation(C) + 1
    return np.maximum(1, np.round(head * ranks.astype(float) ** (-exponent))).astype(int)
