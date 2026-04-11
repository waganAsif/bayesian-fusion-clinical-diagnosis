"""
================================================================================
Step 1: Generate Disease-Symptom Probability Matrix
================================================================================
Input:  data/raw/Training.csv (Kaggle Disease Prediction Dataset)
Output: data/generated/disease_symptom_probabilities_unified_corrected.csv

Computes P(symptom|disease) for each disease-symptom pair from the Kaggle
training data. Probabilities are rounded to nearest 0.05 and only strong
associations (P >= 0.85) are retained.

Source: Kaushil. Disease Prediction Using Machine Learning. Kaggle, 2019.
  https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
================================================================================
"""

import pandas as pd
import numpy as np
import os

PROBABILITY_THRESHOLD = 0.85
ROUND_TO = 0.05  # Round to nearest 0.05 (gives 0.85, 0.90, 0.95, 1.00)

print("=" * 70)
print("STEP 1: Generate Disease-Symptom Probabilities")
print("=" * 70)

input_file = os.path.join('data', 'raw', 'Training.csv')
output_file = os.path.join('data', 'generated', 'disease_symptom_probabilities_unified_corrected.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"\n[1] Loading {input_file}...")
df = pd.read_csv(input_file)
df.columns = [c.strip() for c in df.columns]

symptom_cols = [c for c in df.columns if c != 'prognosis']
print(f"    {len(df)} records, {len(symptom_cols)} symptoms, {df['prognosis'].nunique()} diseases")

print(f"\n[2] Computing P(symptom|disease) with threshold >= {PROBABILITY_THRESHOLD}...")

records = []
for disease in sorted(df['prognosis'].unique()):
    disease_subset = df[df['prognosis'] == disease]
    n_cases = len(disease_subset)

    for symptom in symptom_cols:
        count = int(disease_subset[symptom].sum())
        if count == 0:
            continue
        raw_prob = count / n_cases
        # Round to nearest 0.05
        rounded_prob = round(round(raw_prob / ROUND_TO) * ROUND_TO, 2)
        if rounded_prob >= PROBABILITY_THRESHOLD:
            records.append({
                'disease': disease,
                'symptom': symptom.strip().lower().replace(' ', '_'),
                'probability': rounded_prob
            })

result_df = pd.DataFrame(records)

print(f"\n[3] Summary:")
print(f"    Associations: {len(result_df)}")
print(f"    Diseases: {result_df['disease'].nunique()}")
print(f"    Symptoms: {result_df['symptom'].nunique()}")
print(f"    Probability values: {sorted(result_df['probability'].unique())}")

result_df.to_csv(output_file, index=False)
print(f"\n[4] Saved: {output_file}")
print("=" * 70)
