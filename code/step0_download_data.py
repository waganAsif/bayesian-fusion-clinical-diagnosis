"""
================================================================================
Step 0: Download Source Datasets
================================================================================
This script provides instructions and automated download where possible
for the three source datasets used in the paper.

Source 1: Disease Prediction Dataset (Kaggle)
  - Manual download required (needs Kaggle account)
  - URL: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
  - Download Training.csv and place in data/raw/

Source 2: UCI Drug Reviews Dataset (Drugs.com)
  - URL: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Drugs.com)
  - Download drugsComTrain_raw.tsv and place in data/raw/

Source 3: Columbia University Disease-Symptom Knowledge Database
  - Reference: Wang et al., AMIA Annual Symposium Proceedings, 2008:783-787
  - The Kaggle dataset is derived from this source
  - No separate download needed

After downloading, your data/raw/ folder should contain:
  - Training.csv          (from Kaggle, 4920 rows)
  - drugsComTrain_raw.tsv (from UCI, ~161k rows)

Then run the scripts in order:
  python code/step1_generate_symptom_probabilities.py
  python code/step2_generate_drug_probabilities.py
  python code/step3_generate_evaluation_cases.py
  python code/step4_run_experiments.py
================================================================================
"""

import os
import sys

def check_files():
    """Check if source files are present"""
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    files_needed = {
        'Training.csv': {
            'source': 'Kaggle Disease Prediction Dataset',
            'url': 'https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning',
            'instructions': 'Log in to Kaggle → Download → Extract Training.csv → Place in data/raw/'
        },
        'drugsComTrain_raw.tsv': {
            'source': 'UCI Drug Reviews Dataset',
            'url': 'https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Drugs.com)',
            'instructions': 'Download the dataset → Extract drugsComTrain_raw.tsv → Place in data/raw/'
        }
    }
    
    all_present = True
    
    print("=" * 70)
    print("CHECKING SOURCE DATASETS")
    print("=" * 70)
    
    for filename, info in files_needed.items():
        filepath = os.path.join(raw_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"\n  [OK] {filename} ({size:,} bytes)")
        else:
            all_present = False
            print(f"\n  [MISSING] {filename}")
            print(f"    Source: {info['source']}")
            print(f"    URL: {info['url']}")
            print(f"    Steps: {info['instructions']}")
    
    if all_present:
        print("\n" + "=" * 70)
        print("All source files present! You can now run the pipeline:")
        print("  python code/step1_generate_symptom_probabilities.py")
        print("  python code/step2_generate_drug_probabilities.py")
        print("  python code/step3_generate_evaluation_cases.py")
        print("  python code/step4_run_experiments.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("PLEASE DOWNLOAD THE MISSING FILES BEFORE PROCEEDING")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    check_files()
