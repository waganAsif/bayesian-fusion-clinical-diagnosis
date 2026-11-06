"""Diagnostic models.

Every model exposes ``fit(X, y, K)`` and ``logits(X)``. ``X`` is a binary
case-by-symptom matrix, ``y`` holds disease indices 0..C-1 and ``K`` is the
knowledge table (disease x symptom, P(symptom | disease), zeros for missing
associations). Models that ignore the knowledge table simply do not use ``K``.

All Naive Bayes variants use a uniform disease prior, because every evaluation
set in the paper is balanced across diseases.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


def _class_counts(X, y, C):
    return np.vstack([X[y == c].sum(axis=0) for c in range(C)])


def kb_log_likelihood(K: np.ndarray, eps: float) -> np.ndarray:
    """Row-normalised log P_KB(symptom | disease) with floor ``eps``."""
    return log_softmax(np.log(np.maximum(K, eps)), axis=1)


def data_log_likelihood(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Multinomial Naive Bayes estimate with additive smoothing ``alpha``."""
    sm = counts + alpha
    return np.log(sm) - np.log(sm.sum(axis=1, keepdims=True))


class _Base:
    probabilistic = True
    name = "base"

    def logits(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self.logits(X).argmax(axis=1)


class KnowledgeOnlyNB(_Base):
    """Naive Bayes that uses only the knowledge table (lambda = 1)."""
    name = "Knowledge only"

    def __init__(self, eps=1e-2, kb_log=None):
        self.eps, self.kb_log = eps, kb_log

    def fit(self, X, y, K):
        self.L_ = self.kb_log if self.kb_log is not None else kb_log_likelihood(K, self.eps)
        return self

    def logits(self, X):
        return X @ self.L_.T


class RuleLookup(_Base):
    """Deterministic lookup: the disease whose listed symptoms cover most of the
    patient's symptoms wins; ties go to the disease with the shorter list."""
    name = "Rule lookup"
    probabilistic = False

    def fit(self, X, y, K):
        self.M_ = (K > 0).astype(float)
        self.size_ = self.M_.sum(axis=1)
        return self

    def logits(self, X):
        return X[:, :self.M_.shape[1]] @ self.M_.T - 1e-3 * self.size_


class MultinomialNBData(_Base):
    """Naive Bayes estimated from patient data only (lambda = 0)."""
    name = "Data only (NB)"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y, K=None):
        C = K.shape[0] if K is not None else int(y.max()) + 1
        self.L_ = data_log_likelihood(_class_counts(X, y, C), self.alpha)
        return self

    def logits(self, X):
        return X @ self.L_.T


class ComplementNBData(_Base):
    """scikit-learn Complement Naive Bayes .

    scikit-learn stores ``-log`` complement probabilities in ``feature_log_prob_``;
    these are class weights, not log-likelihoods, so they cannot be fused with
    a knowledge table in log space. We keep the model as a baseline only.
    """
    name = "Complement NB"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y, K=None):
        self.C_ = K.shape[0] if K is not None else int(y.max()) + 1
        self.clf_ = ComplementNB(alpha=self.alpha).fit(X, y)
        return self

    def logits(self, X):
        z = np.full((X.shape[0], self.C_), -1e9)
        z[:, self.clf_.classes_] = X @ self.clf_.feature_log_prob_.T
        return z


class LogisticData(_Base):
    """Multinomial logistic regression (L2, class-balanced weights)."""
    name = "Logistic regression"

    def __init__(self, C=1.0):
        self.Creg = C

    def fit(self, X, y, K=None):
        self.C_ = K.shape[0] if K is not None else int(y.max()) + 1
        self.clf_ = LogisticRegression(C=self.Creg, max_iter=3000,
                                       class_weight="balanced").fit(X, y)
        return self

    def logits(self, X):
        z = np.full((X.shape[0], self.C_), -1e9)
        z[:, self.clf_.classes_] = self.clf_.decision_function(X)
        return z


class GradientBoostingData(_Base):
    """Histogram-based gradient boosting (scikit-learn), class-balanced.

    Settings were fixed in advance for small training sets: leaves may hold a
    single case (training sets can have one case per disease), 100 boosting
    rounds of trees with at most 15 leaves, learning rate 0.1, L2 penalty 1.0,
    no early stopping.
    """
    name = "Gradient boosting"

    def fit(self, X, y, K=None):
        self.C_ = K.shape[0] if K is not None else int(y.max()) + 1
        self.clf_ = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=1,
            l2_regularization=1.0, class_weight="balanced", early_stopping=False,
            random_state=0).fit(X, y)
        return self

    def logits(self, X):
        z = np.full((X.shape[0], self.C_), -1e9)
        z[:, self.clf_.classes_] = self.clf_.decision_function(X)
        return z


class PooledNB(_Base):
    """Naive Bayes whose symptom likelihoods pool a knowledge table and patient data.

    Two pooling rules are available, both with the same weights:

    * ``mode="geometric"`` (log-linear pool)
          log P'(s|c) = (1 - w_c) log P_data(s|c) + w_c log P_KB(s|c), renormalised;
    * ``mode="arithmetic"`` (linear pool)
          P'(s|c) = (1 - w_c) P_data(s|c) + w_c P_KB(s|c).

    The weight is either fixed (``lam``) or adaptive, ``w_c = kappa / (kappa + n_c)``
    with n_c the number of training cases of disease c. With an adaptive weight
    the arithmetic pool is the posterior mean of a Dirichlet prior centred on the
    knowledge table, so ``kappa`` reads as "the knowledge base is worth kappa
    patients".
    """

    def __init__(self, mode="geometric", lam=None, kappa=None, alpha=0.5, eps=1e-2, kb_log=None):
        if mode not in ("geometric", "arithmetic"):
            raise ValueError("mode must be 'geometric' or 'arithmetic'")
        if (lam is None) == (kappa is None):
            raise ValueError("give exactly one of lam or kappa")
        self.mode, self.lam, self.kappa = mode, lam, kappa
        self.alpha, self.eps, self.kb_log = alpha, eps, kb_log

    @property
    def name(self):
        rule = "Geometric" if self.mode == "geometric" else "Arithmetic"
        weight = f"fixed {self.lam:g}" if self.lam is not None else "adaptive"
        return f"{rule} pool ({weight})"

    def fit(self, X, y, K):
        C = K.shape[0]
        counts = _class_counts(X, y, C)
        n_c = np.bincount(y, minlength=C).astype(float)
        if self.lam is not None:
            w = np.full((C, 1), float(self.lam))
        else:
            w = (self.kappa / (self.kappa + n_c))[:, None]
        self.w_ = w.ravel()
        Lkb = self.kb_log if self.kb_log is not None else kb_log_likelihood(K, self.eps)
        Ld = data_log_likelihood(counts, self.alpha)
        if self.mode == "geometric":
            self.L_ = log_softmax((1 - w) * Ld + w * Lkb, axis=1)
        else:
            self.L_ = np.log((1 - w) * np.exp(Ld) + w * np.exp(Lkb))
        return self

    def logits(self, X):
        return X @ self.L_.T


class DirichletPriorNB(_Base):
    """Bayesian Naive Bayes with a Dirichlet prior centred on the knowledge table.

    Posterior-mean likelihoods: theta_c = (counts_c + kappa * Lbar * P_KB(.|c) + alpha)
    / normaliser, where Lbar is the average number of symptoms per training case.
    The knowledge table therefore acts like ``kappa`` extra patients per disease,
    and its share of the estimate falls as real cases arrive. This is an
    arithmetic (linear) pool of knowledge and data with a sample-size-adaptive
    weight.
    """
    name = "Arithmetic pool (Dirichlet prior)"

    def __init__(self, kappa=10.0, alpha=0.5, eps=1e-2, kb_log=None):
        self.kappa, self.alpha, self.eps, self.kb_log = kappa, alpha, eps, kb_log

    def fit(self, X, y, K):
        C = K.shape[0]
        counts = _class_counts(X, y, C)
        Lkb = self.kb_log if self.kb_log is not None else kb_log_likelihood(K, self.eps)
        per_case = max(X.sum(axis=1).mean(), 1.0)
        pseudo = self.kappa * per_case * np.exp(Lkb)
        self.L_ = data_log_likelihood(counts + pseudo, self.alpha)
        n_c = np.bincount(y, minlength=C).astype(float)
        self.w_ = self.kappa / (self.kappa + n_c)
        return self

    def logits(self, X):
        return X @ self.L_.T


# ------------------------------------------------------------------ calibration
def nll_from_logits(z, y, beta=1.0):
    zz = beta * z
    return float(np.mean(logsumexp(zz, axis=1) - zz[np.arange(len(y)), y]))


class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., 2017).

    The calibration NLL is convex in the inverse temperature beta = 1/T, so a
    bounded one-dimensional search finds the global optimum.
    """

    def __init__(self, beta_bounds=(1e-3, 50.0)):
        self.beta_bounds = beta_bounds
        self.T_ = 1.0

    def fit(self, z_cal, y_cal):
        z_cal = np.maximum(z_cal, -1e6)
        res = minimize_scalar(lambda b: nll_from_logits(z_cal, y_cal, b),
                              bounds=self.beta_bounds, method="bounded",
                              options={"xatol": 1e-5})
        self.T_ = 1.0 / float(res.x)
        return self

    def transform(self, z):
        return np.maximum(z, -1e6) / self.T_


def softmax_proba(z):
    z = np.maximum(z, -1e6)
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))
