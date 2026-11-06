"""Reproduce every number, table and figure in the paper.

    python scripts/run_all.py            # full run (about 11 minutes on two CPU cores)
    python scripts/run_all.py --quick    # 2 repetitions, for a smoke test

Outputs go to results/raw (CSV), results/tables (LaTeX + CSV) and
results/figures (PDF + PNG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files without a display

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from kbfusion import data, experiments as E, report, treatment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repetitions instead of 10")
    args = ap.parse_args()
    reps = 2 if args.quick else E.MAIN["reps"]

    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    kaggle = data.load_kaggle_table(ROOT / "data/raw/kaggle_disease_symptom_training.csv")
    table = data.knowledge_table(kaggle)
    Q, diseases, symptoms = data.as_arrays(table)
    audit = data.audit_kaggle(kaggle)
    audit.update(data.audit_symbipredict(kaggle, ROOT / "data/raw/symbipredict_2022.csv"))
    reviews = data.load_drug_reviews(ROOT / "data/raw/drugscom_reviews.csv.gz")
    audit["drug_reviews_clean"] = int(len(reviews))
    mapped = treatment.mapped_reviews(reviews)
    audit["drug_reviews_mapped"] = int(len(mapped))
    eff = treatment.effectiveness_table(reviews)
    audit["drug_pairs"] = int(len(eff))
    audit["drug_diseases"] = int(eff["disease"].nunique())
    audit["drug_ingredients"] = int(eff["ingredient"].nunique())
    (raw_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2))

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / "knowledge_table_p_symptom_given_disease.csv")
    eff.to_csv(processed / "treatment_effectiveness.csv", index=False)
    treatment.mapping_table().to_csv(processed / "condition_to_disease_map.csv", index=False)
    treatment.synonym_table().to_csv(processed / "brand_to_active_ingredient.csv", index=False)
    print(f"[data] audit and processed tables written ({time.time() - t0:.0f}s)")

    steps = [
        ("learning_curve", lambda: E.learning_curve(Q, reps=reps)),
        ("fidelity_grid", lambda: E.fidelity_grid(Q, reps=reps)),
        ("long_tail", lambda: E.long_tail(Q, reps=reps)),
        ("lambda_sweep", lambda: E.lambda_sweep(Q, reps=reps)),
        ("sensitivity", lambda: E.sensitivity(Q, reps=reps)),
        ("world_sensitivity", lambda: E.world_sensitivity(Q, reps=reps)),
        ("bigram_ablation", lambda: E.bigram_ablation(Q, reps=reps)),
    ]
    for name, fn in steps:
        t = time.time()
        out = fn()
        if isinstance(out, tuple):  # learning_curve also returns example probabilities
            out = out[0]
        out.to_csv(raw_dir / f"{name}.csv", index=False)
        print(f"[{name}] {len(out)} rows ({time.time() - t:.0f}s)")

    report.make_all(ROOT, Q, diseases, symptoms, reviews)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
