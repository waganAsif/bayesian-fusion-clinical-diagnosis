"""Fast checks of the claims the paper relies on. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kbfusion import data, models as M, simulate, treatment  # noqa: E402

RAW = ROOT / "data" / "raw"


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    C, F = 4, 12
    K = (rng.random((C, F)) < 0.4) * rng.uniform(0.85, 1.0, (C, F))
    K[:, 0] = np.maximum(K[:, 0], 0.9)
    X = (rng.random((40, F)) < 0.3).astype(float)
    X[X.sum(1) == 0, 0] = 1
    y = np.repeat(np.arange(C), 10)
    return K, X, y


def test_dirichlet_prior_equals_adaptive_linear_pool():
    """Proposition 1: the Dirichlet posterior mean is a convex mix of data and knowledge."""
    K, X, y = _toy()
    kappa, alpha, eps = 10.0, 0.5, 1e-2
    model = M.DirichletPriorNB(kappa=kappa, alpha=alpha, eps=eps).fit(X, y, K)
    counts = np.vstack([X[y == c].sum(0) for c in range(K.shape[0])])
    N = counts.sum(1, keepdims=True)
    Lbar = X.sum(1).mean()
    F = K.shape[1]
    omega = kappa * Lbar / (N + kappa * Lbar + alpha * F)
    thetaD = (counts + alpha) / (N + alpha * F)
    thetaK = np.exp(M.kb_log_likelihood(K, eps))
    assert np.allclose(np.exp(model.L_), (1 - omega) * thetaD + omega * thetaK)


def test_geometric_pool_bound():
    """Proposition 2(b): upper bound on how far the geometric pool falls below the data."""
    K, X, y = _toy(1)
    eps, alpha, w = 1e-3, 0.5, 0.5
    model = M.PooledNB("geometric", lam=w, alpha=alpha, eps=eps).fit(X, y, K)
    counts = np.vstack([X[y == c].sum(0) for c in range(K.shape[0])])
    thetaD = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * K.shape[1])
    thetaK = np.exp(M.kb_log_likelihood(K, eps))
    tv = 0.5 * np.abs(thetaD - thetaK).sum(1, keepdims=True)
    lhs = model.L_ - np.log(thetaD)
    rhs = w * (np.log(thetaK) - np.log(thetaD)) - np.log(1 - tv)
    assert np.all(lhs <= rhs + 1e-9)


def test_arithmetic_pool_lower_bound():
    """Proposition 2(a): the arithmetic pool never drops far below the data estimate."""
    K, X, y = _toy(2)
    kappa = 10.0
    model = M.PooledNB("arithmetic", kappa=kappa, eps=1e-6).fit(X, y, K)
    counts = np.vstack([X[y == c].sum(0) for c in range(K.shape[0])])
    thetaD = (counts + 0.5) / (counts.sum(1, keepdims=True) + 0.5 * K.shape[1])
    w = model.w_[:, None]
    assert np.all(model.L_ >= np.log(thetaD) + np.log(1 - w) - 1e-12)


def test_temperature_scaler_recovers_temperature():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 3, (4000, 5))
    p = np.exp(z / 2.0) / np.exp(z / 2.0).sum(1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in p])
    ts = M.TemperatureScaler().fit(z, y)
    assert abs(ts.T_ - 2.0) < 0.15


def test_kaggle_audit_and_symbipredict_duplicate():
    kaggle = data.load_kaggle_table(RAW / "kaggle_disease_symptom_training.csv")
    audit = data.audit_kaggle(kaggle)
    assert audit["records"] == 4920 and audit["diseases"] == 41 and audit["symptoms"] == 131
    assert audit["unique_patterns"] == 304 and audit["associations"] == 321
    dup = data.audit_symbipredict(kaggle, RAW / "symbipredict_2022.csv")
    assert dup["shared_unique_patterns"] == dup["symbipredict_unique_patterns"] == 304


def test_brand_names_are_merged():
    assert treatment.active_ingredient("Malarone") == "atovaquone / proguanil"
    assert treatment.active_ingredient("Atovaquone / proguanil") == "atovaquone / proguanil"
    assert treatment.active_ingredient("Lariam") == treatment.active_ingredient("Mefloquine") == "mefloquine"
    assert "Malaria Prevention" not in treatment.CONDITION_TO_DISEASE


def test_simulation_is_reproducible():
    kaggle = data.load_kaggle_table(RAW / "kaggle_disease_symptom_training.csv")
    Q, _, _ = data.as_arrays(data.knowledge_table(kaggle))
    world = simulate.World(Q)
    a = world.sample(3, np.random.default_rng(7))[0]
    b = world.sample(3, np.random.default_rng(7))[0]
    assert np.array_equal(a, b) and a.sum(1).min() >= 2


def test_baselines_return_one_score_per_disease():
    """Data-driven baselines must score every disease, including ones absent from training."""
    K, X, y = _toy()
    keep = y < 3  # disease 3 has no training cases
    for model in (M.LogisticData(), M.GradientBoostingData()):
        z = model.fit(X[keep], y[keep], K).logits(X)
        assert z.shape == (X.shape[0], K.shape[0])
        assert np.all(np.isfinite(z))
        assert np.all(z[:, 3] < z[:, :3].min(axis=1))
