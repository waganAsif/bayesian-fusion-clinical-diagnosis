"""All experiments reported in the paper.

Every setting below was fixed before the experiments were run:

* patients report each typical symptom with probability 0.6 x its frequency;
* one unrelated symptom per patient on average (Poisson mean 1);
* the knowledge base misses 30% of each disease's associations;
* kappa = 10 (the knowledge base counts as 10 patients), eps = 0.01,
  alpha = 0.5, 10 repetitions, 100 test and 5 calibration cases per disease.

Sensitivity to every one of these choices is reported separately.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import log_softmax

from . import metrics as mt
from . import models as M
from .simulate import World, long_tail_counts, nested_training_sets

MAIN = dict(report=0.6, noise=1.0, gap=0.3, kappa=10.0, eps=1e-2, alpha=0.5,
            reps=10, n_test=100, n_cal=5)
N_LIST = [1, 2, 3, 5, 10, 20, 50, 100]

MODEL_ORDER = [
    "Rule lookup", "Knowledge only", "Data only (NB)", "Complement NB",
    "Logistic regression", "Gradient boosting", "Geometric pool (fixed 0.5)", "Geometric pool (adaptive)",
    "Arithmetic pool (Dirichlet prior)",
]


def model_zoo(kappa=MAIN["kappa"], eps=MAIN["eps"], alpha=MAIN["alpha"]):
    """Fresh instances of every model compared in the main experiment."""
    return [
        M.RuleLookup(),
        M.KnowledgeOnlyNB(eps=eps),
        M.MultinomialNBData(alpha=alpha),
        M.ComplementNBData(alpha=alpha),
        M.LogisticData(),
        M.GradientBoostingData(),
        M.PooledNB("geometric", lam=0.5, alpha=alpha, eps=eps),
        M.PooledNB("geometric", kappa=kappa, alpha=alpha, eps=eps),
        M.DirichletPriorNB(kappa=kappa, alpha=alpha, eps=eps),
    ]


def _evaluate(model, X, y, K, Xte, yte, Xcal, ycal):
    model.fit(X, y, K)
    z = model.logits(Xte)
    row = {"accuracy": mt.accuracy(z, yte), "top3": mt.top_k_accuracy(z, yte, 3),
           "macro_f1": mt.macro_f1(z, yte)}
    if model.probabilistic:
        p_raw = M.softmax_proba(z)
        ts = M.TemperatureScaler().fit(model.logits(Xcal), ycal)
        p = M.softmax_proba(ts.transform(z))
        row.update({"ece_raw": mt.ece(p_raw, yte), "nll_raw": mt.nll(p_raw, yte),
                    "brier_raw": mt.brier(p_raw, yte), "T": ts.T_,
                    "ece": mt.ece(p, yte), "nll": mt.nll(p, yte), "brier": mt.brier(p, yte)})
        return row, p
    return row, None


# ----------------------------------------------------------------------- E1
def learning_curve(Q, n_list=N_LIST, gap=MAIN["gap"], noise=MAIN["noise"],
                   report=MAIN["report"], reps=MAIN["reps"], seed=0, keep_proba_at=(5, 50)):
    """Main experiment: every model at every training size, 10 repetitions."""
    world = World(Q, report=report, noise=noise)
    rows, kept = [], {}
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        K = world.knowledge_base(gap, rng)
        Xte, yte = world.sample(MAIN["n_test"], rng)
        Xcal, ycal = world.sample(MAIN["n_cal"], rng)
        take = nested_training_sets(world, max(n_list), rng)
        for n in n_list:
            X, y = take(n)
            for model in model_zoo():
                row, p = _evaluate(model, X, y, K, Xte, yte, Xcal, ycal)
                rows.append({"rep": r, "n_per_disease": n, "model": model.name, **row})
                if r == 0 and n in keep_proba_at and p is not None:
                    kept[(n, model.name)] = (p, yte)
    return pd.DataFrame(rows), kept


# ----------------------------------------------------------------------- E2
def fidelity_grid(Q, gaps=(0.0, 0.15, 0.3, 0.45), noises=(0.0, 1.0, 2.0), n_list=(3, 30),
                  reps=MAIN["reps"], seed=100):
    """Knowledge-base gaps x symptom noise, for four representative models."""
    rows = []
    for gap, noise in itertools.product(gaps, noises):
        world = World(Q, report=MAIN["report"], noise=noise)
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            K = world.knowledge_base(gap, rng)
            Xte, yte = world.sample(MAIN["n_test"], rng)
            take = nested_training_sets(world, max(n_list), rng)
            for n in n_list:
                X, y = take(n)
                for model in [M.KnowledgeOnlyNB(eps=MAIN["eps"]), M.MultinomialNBData(),
                              M.PooledNB("geometric", kappa=MAIN["kappa"], eps=MAIN["eps"]),
                              M.DirichletPriorNB(kappa=MAIN["kappa"], eps=MAIN["eps"])]:
                    model.fit(X, y, K)
                    rows.append({"gap": gap, "noise": noise, "rep": r, "n_per_disease": n,
                                 "model": model.name, "accuracy": mt.accuracy(model.logits(Xte), yte)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- E3
def long_tail(Q, head=100, exponent=1.2, reps=MAIN["reps"], seed=200):
    """Imbalanced training data: a few common diseases, many rare ones."""
    world = World(Q, report=MAIN["report"], noise=MAIN["noise"])
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        K = world.knowledge_base(MAIN["gap"], rng)
        counts = long_tail_counts(world.C, head, exponent, rng)
        X, y = world.sample(counts, rng)
        Xte, yte = world.sample(MAIN["n_test"], rng)
        Xcal, ycal = world.sample(MAIN["n_cal"], rng)
        group = np.where(counts[yte] <= 3, "rare (1-3 cases)",
                         np.where(counts[yte] < 20, "mid (4-19 cases)", "common (20+ cases)"))
        for model in model_zoo():
            row, _ = _evaluate(model, X, y, K, Xte, yte, Xcal, ycal)
            z = model.logits(Xte)
            hit = z.argmax(axis=1) == yte
            for g in ["rare (1-3 cases)", "mid (4-19 cases)", "common (20+ cases)"]:
                row[f"acc_{g}"] = float(hit[group == g].mean())
            rows.append({"rep": r, "model": model.name, "train_cases": int(counts.sum()),
                         "n_rare": int((counts <= 3).sum()), **row})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- E4
def lambda_sweep(Q, lams=np.round(np.linspace(0, 1, 11), 2), n_list=(1, 3, 10, 30, 100),
                 reps=MAIN["reps"], seed=0):
    """Fixed-weight pools: how the best weight moves with the amount of data.

    Uses exactly the same random draws as ``learning_curve`` (same seed and draw
    order), so lambda = 0 and lambda = 1 reproduce the data-only and knowledge-only
    results of the main experiment.
    """
    world = World(Q, report=MAIN["report"], noise=MAIN["noise"])
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        K = world.knowledge_base(MAIN["gap"], rng)
        Xte, yte = world.sample(MAIN["n_test"], rng)
        world.sample(MAIN["n_cal"], rng)  # calibration draw of the main experiment (unused here)
        take = nested_training_sets(world, max(n_list), rng)
        for n in n_list:
            X, y = take(n)
            for mode in ("geometric", "arithmetic"):
                for lam in lams:
                    model = M.PooledNB(mode, lam=float(lam), eps=MAIN["eps"]).fit(X, y, K)
                    rows.append({"rep": r, "n_per_disease": n, "rule": mode, "lambda": float(lam),
                                 "accuracy": mt.accuracy(model.logits(Xte), yte)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- E5
def sensitivity(Q, kappas=(1, 3, 10, 30, 100), epss=(1e-3, 1e-2, 1e-1),
                n_list=(1, 3, 10, 30), reps=MAIN["reps"], seed=400):
    """Sensitivity of the pooled models to kappa and eps (main world)."""
    world = World(Q, report=MAIN["report"], noise=MAIN["noise"])
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        K = world.knowledge_base(MAIN["gap"], rng)
        Xte, yte = world.sample(MAIN["n_test"], rng)
        take = nested_training_sets(world, max(n_list), rng)
        for n in n_list:
            X, y = take(n)
            for kappa, eps in itertools.product(kappas, epss):
                for model in [M.PooledNB("geometric", kappa=kappa, eps=eps),
                              M.PooledNB("arithmetic", kappa=kappa, eps=eps),
                              M.DirichletPriorNB(kappa=kappa, eps=eps)]:
                    model.fit(X, y, K)
                    rows.append({"rep": r, "n_per_disease": n, "kappa": kappa, "eps": eps,
                                 "model": model.name, "accuracy": mt.accuracy(model.logits(Xte), yte)})
            for eps in epss:
                model = M.KnowledgeOnlyNB(eps=eps).fit(X, y, K)
                rows.append({"rep": r, "n_per_disease": n, "kappa": np.nan, "eps": eps,
                             "model": model.name, "accuracy": mt.accuracy(model.logits(Xte), yte)})
            model = M.MultinomialNBData().fit(X, y, K)
            rows.append({"rep": r, "n_per_disease": n, "kappa": np.nan, "eps": np.nan,
                         "model": model.name, "accuracy": mt.accuracy(model.logits(Xte), yte)})
    return pd.DataFrame(rows)


def world_sensitivity(Q, reports=(0.4, 0.6, 0.8), n_list=(3, 30), reps=MAIN["reps"], seed=500):
    """Does the conclusion depend on how many typical symptoms patients report?"""
    rows = []
    for report in reports:
        world = World(Q, report=report, noise=MAIN["noise"])
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            K = world.knowledge_base(MAIN["gap"], rng)
            Xte, yte = world.sample(MAIN["n_test"], rng)
            take = nested_training_sets(world, max(n_list), rng)
            for n in n_list:
                X, y = take(n)
                for model in [M.KnowledgeOnlyNB(eps=MAIN["eps"]), M.MultinomialNBData(),
                              M.PooledNB("geometric", kappa=MAIN["kappa"], eps=MAIN["eps"]),
                              M.DirichletPriorNB(kappa=MAIN["kappa"], eps=MAIN["eps"])]:
                    model.fit(X, y, K)
                    rows.append({"report": report, "rep": r, "n_per_disease": n, "model": model.name,
                                 "accuracy": mt.accuracy(model.logits(Xte), yte)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- E6
def _pair_features(X, pairs):
    Xs = sparse.csr_matrix(X)
    a, b = pairs[:, 0], pairs[:, 1]
    P = Xs[:, a].multiply(Xs[:, b])
    return sparse.hstack([Xs, P]).tocsr()


def bigram_ablation(Q, n_list=(3, 30), reps=MAIN["reps"], seed=600):
    """Pairwise symptom features with the the geometric-mean bigram prior.."""
    world = World(Q, report=MAIN["report"], noise=MAIN["noise"])
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        K = world.knowledge_base(MAIN["gap"], rng)
        Xte, yte = world.sample(MAIN["n_test"], rng)
        take = nested_training_sets(world, max(n_list), rng)
        prof = K > 0
        kb_pairs = set()
        for c in range(world.C):
            idx = np.where(prof[c])[0]
            kb_pairs.update(itertools.combinations(idx, 2))
        for n in n_list:
            X, y = take(n)
            tr_pairs = set()
            for row in X:
                tr_pairs.update(itertools.combinations(np.where(row > 0)[0], 2))
            pairs = np.array(sorted(kb_pairs | tr_pairs))
            base = np.log(np.maximum(K, MAIN["eps"]))
            kb_log2 = log_softmax(np.hstack([base, 0.5 * (base[:, pairs[:, 0]] + base[:, pairs[:, 1]])]), axis=1)
            X2, Xte2 = _pair_features(X, pairs), _pair_features(Xte, pairs)
            K2 = np.hstack([K, np.zeros((world.C, len(pairs)))])
            for feats, Xa, Xb, kbl, Kx in [("single symptoms", X, Xte, None, K),
                                           ("single + pairs", X2, Xte2, kb_log2, K2)]:
                for model in [M.MultinomialNBData(),
                              M.PooledNB("geometric", kappa=MAIN["kappa"], eps=MAIN["eps"], kb_log=kbl),
                              M.DirichletPriorNB(kappa=MAIN["kappa"], eps=MAIN["eps"], kb_log=kbl)]:
                    model.fit(Xa if not sparse.issparse(Xa) else Xa.toarray(), y, Kx)
                    z = np.asarray(Xb @ model.L_.T)
                    rows.append({"rep": r, "n_per_disease": n, "features": feats, "model": model.name,
                                 "accuracy": mt.accuracy(z, yte)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summaries
def summarise(df, by, value="accuracy"):
    g = df.groupby(by)[value]
    return g.mean().rename("mean").to_frame().join(g.std().rename("sd"))


def paired_tests(df, reference="Arithmetic pool (Dirichlet prior)", value="accuracy",
                 by="n_per_disease"):
    """Wilcoxon signed-rank tests of the reference model against every other model."""
    out = []
    for key, sub in df.groupby(by):
        piv = sub.pivot_table(index="rep", columns="model", values=value)
        if reference not in piv:
            continue
        for other in piv.columns:
            if other == reference:
                continue
            diff, p = mt.paired_test(piv[reference], piv[other])
            out.append({by: key, "other": other, "difference": diff, "p_value": p})
    return pd.DataFrame(out)
