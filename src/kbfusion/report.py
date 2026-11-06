"""Figures, LaTeX tables and the key numbers quoted in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import experiments as E
from . import models as M
from . import treatment as tr
from .simulate import World, nested_training_sets

# ------------------------------------------------------------------ style
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {  # colour follows the model in every figure
    "Arithmetic pool (Dirichlet prior)": "#2a78d6",
    "Geometric pool (adaptive)": "#eb6834",
    "Data only (NB)": "#1baf7a",
    "Logistic regression": "#eda100",
    "Geometric pool (fixed 0.5)": "#e87ba4",
    "Knowledge only": "#008300",
    "Gradient boosting": "#4a3aa7",
}
MARKERS = {
    "Arithmetic pool (Dirichlet prior)": "o", "Geometric pool (adaptive)": "s",
    "Data only (NB)": "^", "Logistic regression": "D",
    "Geometric pool (fixed 0.5)": "v", "Knowledge only": None, "Gradient boosting": "P",
}
SHORT = {
    "Arithmetic pool (Dirichlet prior)": "Arithmetic pool (adaptive)",
    "Geometric pool (adaptive)": "Geometric pool (adaptive)",
    "Geometric pool (fixed 0.5)": r"Geometric pool ($\lambda$ = 0.5)",
    "Data only (NB)": "Data only (Naive Bayes)",
    "Logistic regression": "Logistic regression",
    "Gradient boosting": "Gradient boosting",
    "Knowledge only": "Knowledge only",
    "Complement NB": "Complement NB",
    "Rule lookup": "Rule lookup",
}
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def _style():
    plt.rcParams.update({
        # IEEE Access asks for figure labels at roughly 8-10 pt in the printed
        # figure, and names Times/Helvetica/Arial/Cambria as preferred faces.
        # Liberation Serif is metric-compatible with Times; the list degrades
        # gracefully if it is not installed.
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-compatible, bundled with matplotlib
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
    })


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)  # IEEE raster floor is 300 dpi
    plt.close(fig)


def _line(ax, x, y, model, **kw):
    style = dict(color=COLORS[model], marker=MARKERS[model], markeredgecolor="white",
                 markeredgewidth=0.8, label=SHORT[model], zorder=3)
    if model == "Knowledge only":
        style.update(linestyle=(0, (4, 2)), marker=None)
    style.update(kw)
    ax.plot(x, y, **style)


# ---------------------------------------------------------------- figures
FIG_MODELS = ["Arithmetic pool (Dirichlet prior)", "Geometric pool (adaptive)",
              "Geometric pool (fixed 0.5)", "Data only (NB)", "Logistic regression",
              "Knowledge only", "Gradient boosting"]


def fig_learning_curves(lc: pd.DataFrame, out: Path):
    g = lc.groupby(["model", "n_per_disease"])
    acc, acc_sd, nll = g["accuracy"].mean(), g["accuracy"].std(), g["nll"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for m in FIG_MODELS:
        x = acc[m].index.values
        _line(axes[0], x, acc[m].values, m)
        axes[0].fill_between(x, acc[m] - acc_sd[m], acc[m] + acc_sd[m], color=COLORS[m], alpha=0.10, lw=0)
        _line(axes[1], x, nll[m].values, m)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlabel("Training cases per disease")
    axes[0].set_ylabel("Accuracy (41 diseases)")
    axes[0].set_title("(a) Accuracy", loc="left")
    axes[1].set_yscale("log")
    yt = [t for t in (0.2, 0.3, 0.5, 1.0, 2.0, 3.0) if nll.min() * 0.9 <= t <= nll.max() * 1.1]
    axes[1].set_yticks(yt)
    axes[1].set_yticklabels([f"{t:g}" for t in yt])
    axes[1].yaxis.set_minor_locator(plt.NullLocator())
    axes[1].set_ylabel("NLL after temperature scaling")
    axes[1].set_title("(b) Probability quality (lower is better)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "fig2_learning_curves")


def fig_fidelity(fg: pd.DataFrame, out: Path):
    piv = fg.pivot_table(index=["n_per_disease", "gap", "noise"], columns="model", values="accuracy")
    best_single = piv[["Knowledge only", "Data only (NB)"]].max(axis=1)
    cmap = LinearSegmentedColormap.from_list("div", ["#c43d3c", "#e9a3a0", "#f0efec", "#9ec5f4", "#1c5cab"])
    gaps, noises = sorted(fg["gap"].unique()), sorted(fg["noise"].unique())
    rules = [("Arithmetic pool (Dirichlet prior)", "Arithmetic pool (adaptive)"),
             ("Geometric pool (adaptive)", "Geometric pool (adaptive)")]
    ns = sorted(fg["n_per_disease"].unique())
    lim = 0.12
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, axes = plt.subplots(2, len(ns), figsize=(7.2, 4.8), sharex=True, sharey=True)
    for i, (m, label) in enumerate(rules):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            gain = (piv[m] - best_single).loc[n].unstack("noise").loc[gaps, noises].values * 100
            im = ax.imshow(gain / 100, cmap=cmap, norm=norm, aspect="auto", origin="lower")
            for a in range(len(gaps)):
                for b in range(len(noises)):
                    v = gain[a, b]
                    lab = "0.0" if abs(v) < 0.05 else f"{v:+.1f}"  # avoid "-0.0" / "+0.0"
                    ax.text(b, a, lab, ha="center", va="center", fontsize=8.5,
                            color="white" if abs(v) > 7 else INK)
            ax.set_xticks(range(len(noises)))
            ax.set_xticklabels([f"{x:g}" for x in noises])
            ax.set_yticks(range(len(gaps)))
            ax.set_yticklabels([f"{int(g * 100)}%" for g in gaps])
            ax.tick_params(length=0)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"{n} training cases per disease", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{label.split(' (')[0]}\nknowledge-base gaps")
    fig.supxlabel("Unrelated symptoms per patient (Poisson mean)", fontsize=9, color=INK2, y=0.02)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.03)
    cb.set_label("Gain over the better single source (accuracy points)")
    cb.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    cb.set_ticklabels(["-10", "-5", "0", "+5", "+10"])
    cb.outline.set_visible(False)
    _save(fig, out, "fig3_fidelity_grid")


def fig_lambda(ls: pd.DataFrame, out: Path):
    g = ls.groupby(["rule", "n_per_disease", "lambda"])["accuracy"].mean()
    ns = sorted(ls["n_per_disease"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, rule, title in [(axes[0], "geometric", "(a) Geometric pool, fixed weight"),
                            (axes[1], "arithmetic", "(b) Arithmetic pool, fixed weight")]:
        for k, n in enumerate(ns):
            s = g.loc[rule, n]
            ax.plot(s.index, s.values, color=ORDINAL[k], marker="o", markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=f"{n} case{'s' if n > 1 else ''}/disease")
            best = s.idxmax()
            ax.plot([best], [s.max()], marker="o", markersize=7, color=ORDINAL[k],
                    markeredgecolor=INK, markeredgewidth=0.9, zorder=4)
        ax.set_xlabel(r"Knowledge weight $\lambda$")
        ax.set_title(title, loc="left")
        ax.set_xticks(np.linspace(0, 1, 6))
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "fig4_lambda_sweep")


def fig_treatment(examples, out: Path):
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.2, 1.9 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for i, ex in enumerate(examples):
        for j, (items, xlabel) in enumerate([(ex["posterior"], "Posterior probability"),
                                             (ex["treatments"], "Expected utility")]):
            ax = axes[i, j]
            slots = 4 if j == 0 else 5
            names = [n.replace(" / ", "/") for n, _ in items]
            vals = [v for _, v in items]
            y = slots - 1 - np.arange(len(names))
            ax.barh(y, vals, height=0.55, color=COLORS["Arithmetic pool (Dirichlet prior)"])
            ax.set_ylim(-0.6, slots - 0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8.5, color=INK2)
            xmax = 1.0  # same 0-1 scale for posteriors and expected utilities
            ax.set_xlim(0, xmax)
            for yy, v in zip(y, vals):
                ax.text(v + xmax * 0.015, yy, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
            if j == 0 and len(names) < slots:
                ax.text(0.02, (slots - 1 - len(names)) / 2, "All other diseases: < 0.005",
                        va="center", fontsize=8.5, color=INK2, style="italic")
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            if i == len(examples) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_title(f"({chr(97 + i)}) {ex['title']}", loc="left", fontsize=9)
    fig.tight_layout(h_pad=1.4)
    _save(fig, out, "fig5_treatment_examples")


# ------------------------------------------------------------- treatment demo
EXAMPLE_PATIENTS = [
    ("Patient A: burning micturition, bladder discomfort, foul-smelling urine",
     ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine"]),
    ("Patient B: continuous sneezing, chills, headache",
     ["continuous_sneezing", "chills", "headache"]),
]


def treatment_examples(Q, diseases, symptoms, reviews, seed=0, n_per=10):
    world = World(Q, report=E.MAIN["report"], noise=E.MAIN["noise"])
    rng = np.random.default_rng(seed)
    K = world.knowledge_base(E.MAIN["gap"], rng)
    X, y = nested_training_sets(world, n_per, rng)(n_per)
    Xcal, ycal = world.sample(E.MAIN["n_cal"], rng)
    model = M.DirichletPriorNB(kappa=E.MAIN["kappa"], eps=E.MAIN["eps"]).fit(X, y, K)
    ts = M.TemperatureScaler().fit(model.logits(Xcal), ycal)
    eff = tr.effectiveness_table(reviews)
    U, drugs = tr.utility_matrix(eff, diseases)
    s_idx = {s: i for i, s in enumerate(symptoms)}
    out = []
    for title, syms in EXAMPLE_PATIENTS:
        x = np.zeros((1, len(symptoms)))
        x[0, [s_idx[s] for s in syms]] = 1
        p = M.softmax_proba(ts.transform(model.logits(x)))[0]
        top = [k for k in np.argsort(-p)[:4] if p[k] >= 0.005]
        posterior = [(diseases[k], float(p[k])) for k in top]
        treatments = [(d.capitalize(), u) for d, u in tr.rank_treatments(p, U, drugs, top=5)]
        coverage = float(p[(U > 0).any(axis=1)].sum())
        out.append({"title": title, "symptoms": syms, "posterior": posterior,
                    "treatments": treatments, "coverage": coverage})
    return out


# ------------------------------------------------------------------ tables
def _pm(m, s, d=3):
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def _latex_name(m):
    return SHORT[m].replace("$\\lambda$", "$\\lambda$")


TABLE_MODELS = ["Rule lookup", "Knowledge only", "Data only (NB)", "Complement NB", "Logistic regression",
                "Gradient boosting", "Geometric pool (fixed 0.5)", "Geometric pool (adaptive)",
                "Arithmetic pool (Dirichlet prior)"]


def table_main(lc, out: Path, ns=(1, 3, 10, 100)):
    g = lc.groupby(["model", "n_per_disease"])["accuracy"]
    mean, sd = g.mean(), g.std()
    tests = E.paired_tests(lc)
    tests = tests.set_index(["n_per_disease", "other"])
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(ns) + "@{}}", r"\toprule",
             "Model & " + " & ".join(f"$n={n}$" for n in ns) + r" \\", r"\midrule"]
    csv = []
    for m in TABLE_MODELS:
        cells = []
        for n in ns:
            best = mean.xs(n, level="n_per_disease").max()
            c = _pm(mean[(m, n)], sd[(m, n)])
            if abs(mean[(m, n)] - best) < 5e-4:
                c = f"$\\mathbf{{{mean[(m, n)]:.3f} \\pm {sd[(m, n)]:.3f}}}$"
            if m != "Arithmetic pool (Dirichlet prior)" and (n, m) in tests.index and tests.loc[(n, m), "p_value"] < 0.01:
                c += r"$^{\dagger}$"
            cells.append(c)
            csv.append({"model": m, "n_per_disease": n, "mean": mean[(m, n)], "sd": sd[(m, n)]})
        if m == "Geometric pool (fixed 0.5)":
            lines.append(r"\midrule")
        lines.append(_latex_name(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (out / "accuracy.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv).to_csv(out / "accuracy.csv", index=False)
    E.paired_tests(lc).to_csv(out / "paired_tests.csv", index=False)


def table_calibration(lc, out: Path, ns=(3, 50)):
    prob = [m for m in TABLE_MODELS if m != "Rule lookup"]
    g = lc.groupby(["model", "n_per_disease"])
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "ccc" * len(ns) + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$n={n}$ cases per disease}}" for n in ns) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(ns))),
             "Model & " + " & ".join(["ECE raw $\\to$ scaled", "NLL", "Brier"] * len(ns)) + r" \\", r"\midrule"]
    rows = []
    for m in prob:
        cells = []
        for n in ns:
            s = g.get_group((m, n))
            cells += [f"{s['ece_raw'].mean():.3f} $\\to$ {s['ece'].mean():.3f}",
                      f"{s['nll'].mean():.3f}", f"{s['brier'].mean():.3f}"]
            rows.append({"model": m, "n_per_disease": n, **{k: s[k].mean() for k in
                         ["ece_raw", "ece", "nll_raw", "nll", "brier_raw", "brier", "T"]}})
        lines.append(_latex_name(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (out / "calibration.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)


def table_long_tail(lt, out: Path):
    cols = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
            ("acc_rare (1-3 cases)", "Rare"), ("acc_mid (4-19 cases)", "Mid"),
            ("acc_common (20+ cases)", "Common"), ("nll", "NLL")]
    g = lt.groupby("model")
    mean, sd = g[[c for c, _ in cols]].mean(), g[[c for c, _ in cols]].std()
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Accuracy by training frequency} & \\ \cmidrule(lr){4-6}",
             "Model & " + " & ".join(n for _, n in cols) + r" \\", r"\midrule"]
    for m in TABLE_MODELS:
        cells = []
        for c, _ in cols:
            v = mean.loc[m, c]
            if np.isnan(v):
                cells.append("--")
                continue
            best = mean[c].min() if c == "nll" else mean[c].max()
            s = f"{v:.3f}"
            if abs(v - best) < 5e-4:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(_latex_name(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (out / "long_tail.tex").write_text("\n".join(lines) + "\n")
    mean.join(sd, rsuffix="_sd").to_csv(out / "long_tail.csv")


def table_sensitivity(se, out: Path, n=3):
    # Three blocks: the first two share the weight w_d and differ only in the
    # pooling rule, which isolates the rule from the weighting scheme.
    models = [("Geometric pool (adaptive)", r"Geometric, weight $w_d$"),
              ("Arithmetic pool (adaptive)", r"Arithmetic, \emph{same} weight $w_d$"),
              ("Arithmetic pool (Dirichlet prior)", r"Arithmetic, Dirichlet weight $\omega_d$")]
    s = se[(se["n_per_disease"] == n) & se["model"].isin([m for m, _ in models])]
    piv = s.pivot_table(index="kappa", columns=["model", "eps"], values="accuracy")
    epss = sorted(s["eps"].unique())
    nb, ne = len(models), len(epss)
    lines = [r"\begin{tabular}{@{}c" + "c" * (nb * ne) + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{ne}}}{{c}}{{{lab}}}" for _, lab in models) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + i * ne}-{1 + (i + 1) * ne}}}" for i in range(nb)),
             r"$\kappa$ & " + " & ".join([rf"$\varepsilon={e:g}$" for e in epss] * nb) + r" \\", r"\midrule"]
    for k in piv.index:
        cells = [f"{piv.loc[k, (m, e)]:.3f}" for m, _ in models for e in epss]
        lines.append(f"{k:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "sensitivity.tex").write_text("\n".join(lines) + "\n")
    se.groupby(["model", "eps", "kappa", "n_per_disease"], dropna=False)["accuracy"].mean().to_csv(
        out / "sensitivity_full.csv")
def table_datasets(audit, out: Path):
    lines = [r"\begin{tabularx}{\textwidth}{@{}p{3.4cm}p{3.3cm}>{\raggedright\arraybackslash}X@{}}",
             r"\toprule",
             r"Source & Size & Use in this study \\", r"\midrule",
             rf"Kaggle disease--symptom records & {audit['records']:,} records, {audit['diseases']} diseases, "
             rf"{audit['symptoms']} symptoms & Knowledge table $P(s\mid d)$ and simulation template. Only "
             rf"{audit['unique_patterns']} distinct symptom patterns; {audit['associations']} disease--symptom "
             rf"associations, each with frequency {audit['frequency_min']:.2f}--{audit['frequency_max']:.2f}. \\",
             rf"Drugs.com reviews (UCI) & {audit['drug_reviews_clean']:,} usable reviews & Treatment "
             rf"illustration: {audit['drug_reviews_mapped']:,} reviews mapped to {audit['drug_diseases']} diseases "
             rf"with $\geq$10 reviews per active ingredient ({audit['drug_pairs']} disease--ingredient pairs). \\",
             r"\bottomrule", r"\end{tabularx}"]
    (out / "datasets.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ numbers
def key_numbers(lc, fg, lt, ls, se, ws, bg, examples, audit):
    acc = lc.groupby(["model", "n_per_disease"])["accuracy"].mean()
    num = {"audit": audit}
    num["accuracy"] = {f"{m}|{n}": round(float(v), 4) for (m, n), v in acc.items()}
    for v in ["nll", "ece", "ece_raw", "brier", "T", "top3", "macro_f1"]:
        s = lc.groupby(["model", "n_per_disease"])[v].mean()
        num[v] = {f"{m}|{n}": round(float(x), 4) for (m, n), x in s.items()}
    tests = E.paired_tests(lc)
    num["paired_tests"] = tests.round(5).to_dict(orient="records")
    gp = fg.groupby(["model", "n_per_disease", "gap", "noise"])["accuracy"].mean()
    num["fidelity"] = {f"{m}|{n}|{g}|{z}": round(float(v), 4) for (m, n, g, z), v in gp.items()}
    lt_mean = lt.groupby("model").mean(numeric_only=True)
    num["long_tail"] = {m: {k: round(float(v), 4) for k, v in row.items()} for m, row in lt_mean.iterrows()}
    lsm = ls.groupby(["rule", "n_per_disease", "lambda"])["accuracy"].mean()
    num["lambda_best"] = {f"{r}|{n}": {"lambda": float(lsm.loc[r, n].idxmax()), "accuracy": round(float(lsm.loc[r, n].max()), 4)}
                          for r in ["geometric", "arithmetic"] for n in sorted(ls["n_per_disease"].unique())}
    sem = se.groupby(["model", "eps", "kappa", "n_per_disease"], dropna=False)["accuracy"].mean()
    num["sensitivity"] = {f"{m}|{e}|{k}|{n}": round(float(v), 4) for (m, e, k, n), v in sem.items()}
    wsm = ws.groupby(["model", "report", "n_per_disease"])["accuracy"].mean()
    num["report_rate"] = {f"{m}|{r}|{n}": round(float(v), 4) for (m, r, n), v in wsm.items()}
    bgm = bg.groupby(["model", "features", "n_per_disease"])["accuracy"].mean()
    num["bigram"] = {f"{m}|{f}|{n}": round(float(v), 4) for (m, f, n), v in bgm.items()}
    num["treatment_examples"] = examples
    return num


def make_all(root: Path, Q, diseases, symptoms, reviews):
    root = Path(root)
    _style()
    raw, figs, tabs = root / "results/raw", root / "results/figures", root / "results/tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    lc = pd.read_csv(raw / "learning_curve.csv")
    fg = pd.read_csv(raw / "fidelity_grid.csv")
    lt = pd.read_csv(raw / "long_tail.csv")
    ls = pd.read_csv(raw / "lambda_sweep.csv")
    se = pd.read_csv(raw / "sensitivity.csv")
    ws = pd.read_csv(raw / "world_sensitivity.csv")
    bg = pd.read_csv(raw / "bigram_ablation.csv")
    audit = json.loads((raw / "dataset_audit.json").read_text())

    fig_learning_curves(lc, figs)
    fig_fidelity(fg, figs)
    fig_lambda(ls, figs)
    examples = treatment_examples(Q, diseases, symptoms, reviews)
    fig_treatment(examples, figs)

    table_datasets(audit, tabs)
    table_main(lc, tabs)
    table_calibration(lc, tabs)
    table_long_tail(lt, tabs)
    table_sensitivity(se, tabs)
    nums = key_numbers(lc, fg, lt, ls, se, ws, bg, examples, audit)
    (root / "results" / "key_numbers.json").write_text(json.dumps(nums, indent=1))
    print("[report] figures, tables and key_numbers.json written")
