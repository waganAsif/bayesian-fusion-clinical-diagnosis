"""Loading and cleaning the two public data sources.

1. Kaggle "Disease Prediction Using Machine Learning" (Training.csv):
   4,920 rows, 41 diseases, 132 binary symptom columns. We use it only as a
   disease-by-symptom *knowledge table* (how often each symptom appears with
   each disease), never as patient data, because it contains just 304 distinct
   symptom patterns.
2. UCI Drug Review Dataset (Drugs.com), train + test splits (215,063 reviews).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column fixes for the Kaggle file. "fluid_overload" is an all-zero duplicate;
# the populated column is "fluid_overload.1".
SYMPTOM_RENAMES = {
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
    "dischromic _patches": "dischromic_patches",
    "fluid_overload.1": "fluid_overload",
}

# Disease names in the Kaggle file contain typos and trailing spaces.
DISEASE_RENAMES = {
    "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal positional vertigo",
    "Osteoarthristis": "Osteoarthritis",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids (piles)",
    "hepatitis A": "Hepatitis A",
    "Chicken pox": "Chickenpox",
    "Bronchial Asthma": "Bronchial asthma",
    "Common Cold": "Common cold",
    "Drug Reaction": "Drug reaction",
    "Fungal infection": "Fungal infection",
    "Heart attack": "Heart attack",
    "Urinary tract infection": "Urinary tract infection",
    "Varicose veins": "Varicose veins",
}


def _clean_kaggle_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    if "fluid_overload" in df.columns and "fluid_overload.1" in df.columns:
        df = df.drop(columns=["fluid_overload"])
    df = df.rename(columns=SYMPTOM_RENAMES)
    df.columns = [c.strip() for c in df.columns]
    df["prognosis"] = df["prognosis"].str.strip()
    df["prognosis"] = df["prognosis"].replace(
        {k.strip(): v for k, v in DISEASE_RENAMES.items()})
    return df


def load_kaggle_table(path: str | Path) -> pd.DataFrame:
    """Return the cleaned Kaggle frame (rows = records, 131 symptoms + prognosis)."""
    return _clean_kaggle_frame(pd.read_csv(path))


def knowledge_table(kaggle: pd.DataFrame) -> pd.DataFrame:
    """Disease x symptom table of P(symptom | disease) from the Kaggle records."""
    symptoms = [c for c in kaggle.columns if c != "prognosis"]
    table = kaggle.groupby("prognosis")[symptoms].mean()
    table = table.loc[sorted(table.index)]
    return table


def audit_kaggle(kaggle: pd.DataFrame) -> dict:
    """Facts about the Kaggle file that the paper reports."""
    symptoms = [c for c in kaggle.columns if c != "prognosis"]
    patterns = kaggle[symptoms].astype(int).astype(str).agg("".join, axis=1)
    per_pattern = kaggle.assign(p=patterns).groupby("p")["prognosis"].nunique()
    table = knowledge_table(kaggle)
    profile = (table > 0).sum(axis=1)
    nonzero = table.values[table.values > 0]
    return {
        "records": int(len(kaggle)),
        "diseases": int(kaggle["prognosis"].nunique()),
        "symptoms": int(len(symptoms)),
        "unique_patterns": int(patterns.nunique()),
        "patterns_shared_by_two_diseases": int((per_pattern > 1).sum()),
        "records_per_disease": int(kaggle["prognosis"].value_counts().iloc[0]),
        "profile_size_min": int(profile.min()),
        "profile_size_mean": float(profile.mean()),
        "profile_size_max": int(profile.max()),
        "associations": int((table.values > 0).sum()),
        "frequency_min": float(nonzero.min()),
        "frequency_max": float(nonzero.max()),
    }


def audit_symbipredict(kaggle: pd.DataFrame, symbi_path: str | Path) -> dict:
    """Compare SymbiPredict with the Kaggle file (it is a re-release of it)."""
    sym = _clean_kaggle_frame(pd.read_csv(symbi_path))
    s_k = [c for c in kaggle.columns if c != "prognosis"]
    s_s = [c for c in sym.columns if c != "prognosis"]
    shared = sorted(set(s_k) & set(s_s))
    pk = set(map(tuple, kaggle[shared].astype(int).values))
    ps = set(map(tuple, sym[shared].astype(int).values))
    return {
        "symbipredict_records": int(len(sym)),
        "symbipredict_symptoms": int(len(s_s)),
        "same_symptom_columns": set(s_k) == set(s_s),
        "kaggle_unique_patterns": len(pk),
        "symbipredict_unique_patterns": len(ps),
        "shared_unique_patterns": len(pk & ps),
    }


def load_drug_reviews(path: str | Path) -> pd.DataFrame:
    """Load the Drugs.com reviews and drop malformed rows.

    About 1,100 rows carry an HTML fragment ("... users found this comment
    helpful.") instead of a condition, and some have no condition at all.
    """
    df = pd.read_csv(path)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["drugName", "condition", "rating"])
    df = df[~df["condition"].str.contains("</span>", regex=False)]
    df = df[df["rating"].between(1, 10)]
    df["drugName"] = df["drugName"].str.strip()
    df["condition"] = df["condition"].str.strip()
    return df.reset_index(drop=True)


def as_arrays(table: pd.DataFrame):
    """Return (Q, diseases, symptoms) with Q as a float array."""
    return table.values.astype(float), list(table.index), list(table.columns)
