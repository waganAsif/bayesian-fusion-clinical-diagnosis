# Sample-size-adaptive fusion of expert knowledge and scarce patient data for symptom-based diagnosis

Code and data for the paper

> **Sample-size-adaptive fusion of expert knowledge and scarce patient data for symptom-based diagnosis: Arithmetic versus geometric pooling**
> Asif Ali Wagan, Sanam Narejo, Shahnawaz Talpur
> Department of Computer Systems Engineering, Mehran University of Engineering & Technology, Jamshoro, Pakistan

The repository regenerates every number, table and figure in the paper with one
command, in about 11 minutes on two CPU cores.

## The question

A diagnostic model for 41 diseases has a disease-symptom knowledge table, which may
be incomplete, and only a handful of labelled patients per disease. Should the two
sources be combined, and if so how? We compare two pooling rules for Naive Bayes
likelihoods, each with a weight that fades as a disease accumulates cases:

* **Geometric (log-linear) pool**: averages log-probabilities.
* **Arithmetic pool**: averages probabilities. With an adaptive weight it is
  exactly a Dirichlet prior centred on the knowledge table (Proposition 1 in the paper).

## Main findings

Setting: 30% of the knowledge table's associations are missing, and patients report
about one unrelated symptom each. Accuracy is the mean over 10 repetitions.

| Training cases per disease | Data-only Naive Bayes | Logistic regression | Gradient boosting | Knowledge only | Geometric pool (adaptive) | **Arithmetic pool (adaptive)** |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.634 | 0.648 | 0.419 | 0.769 | 0.781 | **0.820** |
| 3   | 0.809 | 0.809 | 0.524 | 0.769 | 0.801 | **0.863** |
| 10  | 0.894 | 0.892 | 0.743 | 0.769 | 0.853 | **0.900** |
| 100 | 0.921 | **0.930** | 0.904 | 0.769 | 0.919 | 0.922 |

* Knowledge helps a lot when data are scarce, and the benefit is gone by about
  20 cases per disease.
* Gradient boosting, a strong learner on large tabular data, is the weakest data-driven
  model at small sample sizes; it needs about 20 cases per disease to overtake the knowledge base alone.
* The geometric pool lost to the arithmetic pool in every setting of the main
  experiments with an incomplete knowledge base. A missing association acts as a
  veto that patient data cannot overturn (Proposition 2).
* In long-tailed training data, accuracy on rare diseases (1-3 training cases) rose
  from 0.563 (data-only NB), 0.661 (logistic regression) and 0.447 (gradient boosting)
  to 0.823.
* When the knowledge base is complete, the knowledge base alone is already about
  as good as any fusion. Benchmarks that test on cases drawn from the knowledge
  table cannot tell fusion methods apart.



## Repository structure

```
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── raw/                       # the three public source files (see data/raw/README.md)
│   └── processed/                 # knowledge table, drug effectiveness, mappings (generated)
├── src/kbfusion/
│   ├── data.py                    # loading, cleaning and auditing the sources
│   ├── simulate.py                # simulated patients and incomplete knowledge bases
│   ├── models.py                  # all models + temperature scaling
│   ├── metrics.py                 # accuracy, ECE, NLL, Brier, paired tests
│   ├── treatment.py               # Drugs.com effectiveness, brand->ingredient table, ranking
│   ├── experiments.py             # every experiment in the paper (settings fixed in advance)
│   └── report.py                  # figures, LaTeX tables, key_numbers.json
├── scripts/run_all.py             # one command to reproduce everything
├── notebooks/reproduce_paper.ipynb
├── tests/test_core.py             # checks of the propositions and data audit
└── results/
    ├── raw/                       # per-repetition results (CSV) + dataset_audit.json
    ├── tables/                    # Tables 1-5 (LaTeX + CSV)
    ├── figures/                   # Figures 2-5 (PDF + PNG)
    └── key_numbers.json           # every number quoted in the paper
```

## Reproducing the paper

```bash
git clone https://github.com/waganAsif/bayesian-fusion-clinical-diagnosis.git
cd bayesian-fusion-clinical-diagnosis
pip install -r requirements.txt

python scripts/run_all.py          # full run, 10 repetitions (about 11 min on 2 cores)
python scripts/run_all.py --quick  # 2 repetitions, smoke test (about 2 minutes)
python -m pytest -q                # unit tests (needs pytest)
```

Or open `notebooks/reproduce_paper.ipynb` in Jupyter and run all cells. All seeds
are fixed, so the outputs match the committed files in `results/`.

## Where each item of the paper comes from

| Paper | File |
|---|---|
| Table 1 (notation) | written in the manuscript |
| Table 2 (data sources) | `results/tables/datasets.tex` |
| Table 3 (accuracy vs. training size) | `results/tables/accuracy.tex`, `accuracy.csv`, `paired_tests.csv` |
| Table 4 (long-tailed data) | `results/tables/long_tail.tex`, `long_tail.csv` |
| Table 5 (calibration) | `results/tables/calibration.tex`, `calibration.csv` |
| Table 6 (sensitivity to kappa, eps) | `results/tables/sensitivity.tex`, `sensitivity_full.csv` |
| Figure 1 (framework) | TikZ in the LaTeX project |
| Figures 2-5 | `results/figures/fig2_learning_curves.pdf` ... `fig5_treatment_examples.pdf` |

Figure styling in `report.py` is set so that every label prints between 8 and 10 pt
at the width the figure is placed at in the manuscript, which is what IEEE Access
asks for, and uses a Times-metric serif face to match the body text. If you change
`figsize` or any `fontsize`, check the printed sizes again.
| Numbers quoted in the text | `results/key_numbers.json` |

## Using the models on your own data

```python
import numpy as np
from kbfusion import models

# K: diseases x symptoms knowledge table, K[d, s] = P(symptom s | disease d), 0 if not listed
# X: cases x symptoms binary matrix, y: disease index per case
model = models.DirichletPriorNB(kappa=10, eps=0.01).fit(X_train, y_train, K)

scaler = models.TemperatureScaler().fit(model.logits(X_calib), y_calib)
posterior = models.softmax_proba(scaler.transform(model.logits(X_new)))
```

## Data sources and licences

| Source | Use | Licence / citation |
|---|---|---|
| Kaggle, *Disease Prediction Using Machine Learning* (kaushil268), Training.csv | Knowledge table and simulation template | See the Kaggle dataset page |
| SymbiPredict, Mendeley Data, DOI [10.17632/dv5z3v2xyd.1](https://doi.org/10.17632/dv5z3v2xyd.1) | Duplication check only | See the Mendeley page |
| UCI Drug Review Dataset (Drugs.com), DOI [10.24432/C5SK5S](https://doi.org/10.24432/C55G6J) | Treatment illustration | CC BY 4.0; cite Gräßer et al. (2018) |

The treatment ranking is an illustration of how uncertainty propagates. It ignores
side effects, contraindications, dosing and cost, and must not be used for clinical
decisions.

## Citation

See `CITATION.cff`. Until the paper is published, please cite the repository.

## Licence

Code: MIT (see `LICENSE`). Data files keep their original licences.
