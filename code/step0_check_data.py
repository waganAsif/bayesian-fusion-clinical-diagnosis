"""
================================================================================
Step 0: Verify Source Datasets
================================================================================
Checks that all three raw source datasets are present in data/raw/

Required files:
  1. Training.csv           - Kaggle Disease Prediction Dataset (Kaushil268)
     https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

  2. drugsComTrain_raw.tsv  - UCI Drug Review Dataset (Drugs.com)
     https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com

  3. symbipredict_2022.csv  - SymbiPredict Dataset (Mendeley Data)
     https://data.mendeley.com/datasets/dv5z3v2xyd/1
================================================================================
"""

import os
import sys

def check_files():
    raw_dir = os.path.join('data', 'raw')

    files = {
        'Training.csv': {
            'source': 'Kaggle Disease Prediction Dataset',
            'url': 'https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning'
        },
        'drugsComTrain_raw.tsv': {
            'source': 'UCI Drug Review Dataset (Drugs.com)',
            'url': 'https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com'
        },
        'symbipredict_2022.csv': {
            'source': 'SymbiPredict Dataset (Mendeley Data)',
            'url': 'https://data.mendeley.com/datasets/dv5z3v2xyd/1'
        }
    }

    print("=" * 70)
    print("CHECKING SOURCE DATASETS")
    print("=" * 70)

    all_present = True
    for filename, info in files.items():
        filepath = os.path.join(raw_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"\n  [OK] {filename} ({size:,} bytes)")
        else:
            all_present = False
            print(f"\n  [MISSING] {filename}")
            print(f"    Source: {info['source']}")
            print(f"    URL: {info['url']}")

    if all_present:
        print("\n" + "=" * 70)
        print("All source files present. Run pipeline:")
        print("  python code/step1_generate_symptom_probabilities.py")
        print("  python code/step2_generate_drug_probabilities.py")
        print("  python code/step3_generate_evaluation_cases.py")
        print("  python code/step4_run_experiments.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("DOWNLOAD MISSING FILES BEFORE PROCEEDING")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    check_files()
