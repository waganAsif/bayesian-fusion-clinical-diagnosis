"""
================================================================================
Step 2: Generate Drug-Disease Probability Matrix
================================================================================
Input:  data/raw/drugsComTrain_raw.tsv (UCI Drug Reviews Dataset)
Output: data/generated/disease_drug_probabilities_unified_corrected.csv

Computes P(drug|disease) by:
1. Mapping UCI 'condition' fields to Training.csv disease names
2. Counting reviews per drug for each mapped condition
3. Normalizing: P(drug|disease) = review_count / total_reviews_for_condition

Source: Kallumadi, S. & Gräßer, F. (2018). Drug Review Dataset (Drugs.com).
  UCI Machine Learning Repository. DOI: 10.24432/C5SK5S
  https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
================================================================================
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os

print("=" * 70)
print("STEP 2: Generate Drug-Disease Probabilities")
print("=" * 70)

input_file = os.path.join('data', 'raw', 'drugsComTrain_raw.tsv')
output_file = os.path.join('data', 'generated', 'disease_drug_probabilities_unified_corrected.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"\n[1] Loading {input_file}...")
drug_reviews = pd.read_csv(input_file, sep='\t', on_bad_lines='skip')
drug_reviews.columns = [c.strip().lower() for c in drug_reviews.columns]
drug_reviews['condition'] = drug_reviews['condition'].astype(str).str.strip()
drug_reviews['drugname'] = drug_reviews['drugname'].astype(str).str.strip()
drug_reviews = drug_reviews.dropna(subset=['condition', 'drugname'])
drug_reviews = drug_reviews[drug_reviews['condition'] != 'nan']
print(f"    {len(drug_reviews)} reviews, {drug_reviews['condition'].nunique()} conditions")

# ---- Disease name mapping: UCI condition -> Training.csv disease name ----
print("\n[2] Mapping UCI conditions to Training.csv disease names...")

# UCI conditions are case-insensitive searched; Training.csv names preserved exactly
condition_to_disease = {
    'Vertigo': '(vertigo) Paroymsal  Positional Vertigo',
    'HIV Infection': 'AIDS',
    'HIV/AIDS': 'AIDS',
    'Acne': 'Acne',
    'Allergies': 'Allergy',
    'Allergic Reactions': 'Allergy',
    'Rheumatoid Arthritis': 'Arthritis',
    'Osteoarthritis': 'Arthritis',
    'Arthritis': 'Arthritis',
    'Asthma': 'Bronchial Asthma',
    'Asthma, Maintenance': 'Bronchial Asthma',
    'Asthma, acute': 'Bronchial Asthma',
    'Neck Pain': 'Cervical spondylosis',
    'Cervical Dystonia': 'Cervical spondylosis',
    'Chickenpox': 'Chicken pox',
    'Cold Symptoms': 'Common Cold',
    'Common Cold': 'Common Cold',
    'Cough': 'Common Cold',
    'Diabetes, Type 2': 'Diabetes',
    'Diabetes, Type 1': 'Diabetes',
    'Diabetes Insipidus': 'Diabetes',
    "Athlete's Foot": 'Fungal infection',
    'Fungal Infection': 'Fungal infection',
    'Gastroenteritis': 'Gastroenteritis',
    'Heart Attack': 'Heart attack',
    'Heart Failure': 'Heart attack',
    'Angina': 'Heart attack',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'High Blood Pressure': 'Hypertension',
    'Hypertension': 'Hypertension',
    'Hyperthyroidism': 'Hyperthyroidism',
    "Graves' Disease": 'Hyperthyroidism',
    'Hypoglycemia': 'Hypoglycemia',
    'Hypothyroidism': 'Hypothyroidism',
    'Impetigo': 'Impetigo',
    'Skin Infection': 'Impetigo',
    'Jaundice': 'Jaundice',
    'Malaria': 'Malaria',
    'Malaria Prevention': 'Malaria',
    'Migraine': 'Migraine',
    'Migraine Prevention': 'Migraine',
    'Osteoporosis': 'Osteoarthristis',
    'Joint Pain': 'Osteoarthristis',
    'Stroke': 'Paralysis (brain hemorrhage)',
    'Pneumonia': 'Pneumonia',
    'Psoriasis': 'Psoriasis',
    'Plaque Psoriasis': 'Psoriasis',
    'Tuberculosis': 'Tuberculosis',
    'Tuberculosis, Latent': 'Tuberculosis',
    'Typhoid Fever': 'Typhoid',
    'Urinary Tract Infection': 'Urinary tract infection',
    'Bladder Infection': 'Urinary tract infection',
    'Varicose Veins': 'Varicose veins',
    'Hepatitis A': 'hepatitis A',
    'Hepatitis A, Prevention': 'hepatitis A',
}

# ---- Count reviews per drug per disease ----
print("\n[3] Computing P(drug|disease) via review counts...")

drug_counts = defaultdict(lambda: defaultdict(int))
disease_totals = defaultdict(int)

for _, row in drug_reviews.iterrows():
    condition = row['condition']
    drug = row['drugname']

    # Try exact match first
    disease = condition_to_disease.get(condition)

    # Try case-insensitive match
    if disease is None:
        for cond_key, disease_val in condition_to_disease.items():
            if cond_key.lower() == condition.lower():
                disease = disease_val
                break

    if disease is not None:
        drug_counts[disease][drug] += 1
        disease_totals[disease] += 1

# ---- Normalize to probabilities ----
print("\n[4] Normalizing probabilities...")

records = []
for disease in sorted(drug_counts.keys()):
    total = disease_totals[disease]
    for drug, count in drug_counts[disease].items():
        prob = count / total
        records.append({
            'disease': disease,
            'drugName': drug,
            'probability': prob
        })

result_df = pd.DataFrame(records)

# Verify sums
sums = result_df.groupby('disease')['probability'].sum()
print(f"    All probabilities sum to 1.0: {all(abs(s - 1.0) < 0.001 for s in sums)}")

print(f"\n[5] Summary:")
print(f"    Associations: {len(result_df)}")
print(f"    Diseases with drug data: {result_df['disease'].nunique()}")
print(f"    Unique drugs: {result_df['drugName'].nunique()}")

drugs_per = result_df.groupby('disease')['drugName'].count()
print(f"    Drugs per disease: min={drugs_per.min()}, max={drugs_per.max()}")

result_df.to_csv(output_file, index=False)
print(f"\n[6] Saved: {output_file}")
print("=" * 70)
