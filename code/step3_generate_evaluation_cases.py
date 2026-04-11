"""
================================================================================
Step 3: Generate Evaluation Test Cases
================================================================================
Input:  data/raw/symbipredict_2022.csv (SymbiPredict Dataset, Mendeley Data)
Output: data/generated/test_cases_from_training_balanced_diverse.csv

Constructs 12,300 evaluation cases (300 per disease, 41 diseases) by:
1. Loading SymbiPredict binary symptom matrix (4,961 rows, 131 symptoms, 41 diseases)
2. Extracting symptom profiles per disease
3. Mapping SymbiPredict disease names to Training.csv naming convention
4. Generating 300 diverse cases per disease by sampling 3-8 symptom subsets

Source: Tucker, J. (2024). SymbiPredict: Symptom-Disease Prediction Dataset.
  Mendeley Data, V1. DOI: 10.17632/dv5z3v2xyd.1
  https://data.mendeley.com/datasets/dv5z3v2xyd/1
================================================================================
"""

import pandas as pd
import numpy as np
import random
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CASES_PER_DISEASE = 300
MIN_SYMPTOMS = 3
MAX_SYMPTOMS = 8

print("=" * 70)
print("STEP 3: Generate Evaluation Test Cases")
print("=" * 70)

input_file = os.path.join('data', 'raw', 'symbipredict_2022.csv')
output_file = os.path.join('data', 'generated', 'test_cases_from_training_balanced_diverse.csv')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"\n[1] Loading {input_file}...")
df = pd.read_csv(input_file)
df.columns = [c.strip() for c in df.columns]

symptom_cols = [c for c in df.columns if c != 'prognosis']
diseases = sorted(df['prognosis'].str.strip().unique())
print(f"    {len(df)} records, {len(symptom_cols)} symptoms, {len(diseases)} diseases")

# ---- Disease name mapping: SymbiPredict -> Training.csv ----
print("\n[2] Mapping SymbiPredict disease names to Training.csv convention...")

symbi_to_training = {
    'AIDS': 'AIDS',
    'Acne': 'Acne',
    'Alcoholic Hepatitis': 'Alcoholic hepatitis',
    'Allergy': 'Allergy',
    'Arthritis': 'Arthritis',
    'Bronchial Asthma': 'Bronchial Asthma',
    'Cervical Spondylosis': 'Cervical spondylosis',
    'Chickenpox': 'Chicken pox',
    'Chronic Cholestasis': 'Chronic cholestasis',
    'Common Cold': 'Common Cold',
    'Dengue': 'Dengue',
    'Diabetes': 'Diabetes',
    'Diabetes ': 'Diabetes',  # trailing space variant
    'Dimorphic Hemmorhoids (piles)': 'Dimorphic hemmorhoids(piles)',
    'Drug Reaction': 'Drug Reaction',
    'Fungal Infection': 'Fungal infection',
    'GERD': 'GERD',
    'Gastroenteritis': 'Gastroenteritis',
    'Heart Attack': 'Heart attack',
    'Hepatitis A': 'hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'Hypertension': 'Hypertension',
    'Hypertension ': 'Hypertension',  # trailing space variant
    'Hyperthyroidism': 'Hyperthyroidism',
    'Hypoglycemia': 'Hypoglycemia',
    'Hypothyroidism': 'Hypothyroidism',
    'Impetigo': 'Impetigo',
    'Jaundice': 'Jaundice',
    'Malaria': 'Malaria',
    'Migraine': 'Migraine',
    'Osteoarthritis': 'Osteoarthristis',
    'Paralysis (brain hemorrhage)': 'Paralysis (brain hemorrhage)',
    'Peptic Ulcer Disease': 'Peptic ulcer diseae',
    'Pneumonia': 'Pneumonia',
    'Psoriasis': 'Psoriasis',
    'Tuberculosis': 'Tuberculosis',
    'Typhoid': 'Typhoid',
    'Urinary Tract Infection': 'Urinary tract infection',
    'Varicose Veins': 'Varicose veins',
    'Vertigo': '(vertigo) Paroymsal  Positional Vertigo',
}

# ---- Extract symptom profiles per disease ----
print("\n[3] Extracting symptom profiles...")

disease_symptom_profiles = {}
for disease in diseases:
    disease_subset = df[df['prognosis'].str.strip() == disease]
    n_cases = len(disease_subset)

    symptom_probs = {}
    for symptom in symptom_cols:
        count = int(disease_subset[symptom].sum())
        if count > 0:
            prob = count / n_cases
            symptom_probs[symptom.strip().lower()] = prob

    training_name = symbi_to_training.get(disease, disease)
    disease_symptom_profiles[training_name] = symptom_probs

print(f"    Extracted profiles for {len(disease_symptom_profiles)} diseases")

# ---- Generate test cases ----
print(f"\n[4] Generating {CASES_PER_DISEASE} cases per disease...")

all_cases = []
case_id = 1

for disease in sorted(disease_symptom_profiles.keys()):
    symptom_probs = disease_symptom_profiles[disease]
    symptoms = list(symptom_probs.keys())
    probs = np.array(list(symptom_probs.values()))
    probs_normalized = probs / probs.sum()

    count = 0
    attempts = 0
    max_attempts = CASES_PER_DISEASE * 50

    while count < CASES_PER_DISEASE and attempts < max_attempts:
        attempts += 1

        n_syms = random.randint(MIN_SYMPTOMS, min(MAX_SYMPTOMS, len(symptoms)))

        if len(symptoms) >= n_syms:
            selected_indices = np.random.choice(
                len(symptoms), size=n_syms, replace=False, p=probs_normalized
            )
            selected = sorted([symptoms[i] for i in selected_indices])
        else:
            selected = sorted(symptoms)

        symptom_str = ', '.join(selected)

        all_cases.append({
            'case_id': case_id,
            'symptoms': symptom_str,
            'true_disease': disease
        })
        case_id += 1
        count += 1

result_df = pd.DataFrame(all_cases)

# ---- Verify ----
print(f"\n[5] Verification:")
print(f"    Total cases: {len(result_df)}")
print(f"    Diseases: {result_df['true_disease'].nunique()}")
disease_counts = result_df['true_disease'].value_counts()
print(f"    Cases per disease: min={disease_counts.min()}, max={disease_counts.max()}")
print(f"    Balanced: {disease_counts.min() == disease_counts.max() == CASES_PER_DISEASE}")

sym_per_case = result_df['symptoms'].apply(lambda x: len(x.split(',')))
print(f"    Symptoms per case: min={sym_per_case.min()}, max={sym_per_case.max()}, mean={sym_per_case.mean():.1f}")

all_syms = set()
for syms in result_df['symptoms']:
    for s in syms.split(','):
        all_syms.add(s.strip())
print(f"    Unique symptoms: {len(all_syms)}")

result_df.to_csv(output_file, index=False)
print(f"\n[6] Saved: {output_file}")
print("=" * 70)
