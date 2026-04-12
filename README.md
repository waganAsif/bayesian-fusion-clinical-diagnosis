# Scalable Bayesian Fusion with Log-Space Convex Combination and Compositional Prior Construction

## Overview

Code and data for:

> **Scalable Bayesian Fusion with Log-Space Convex Combination and Compositional Prior Construction**  
> Asif Ali Wagan, Sanam Narejo, Shahnawaz Talpur  
> Department of Computer Systems Engineering, Mehran University of Engineering & Technology, Jamshoro, Pakistan

## Repository Structure

```
├── README.md
├── LICENSE
├── code/
│   ├── step0_check_data.py                        # Verify raw datasets are present
│   ├── step1_generate_symptom_probabilities.py    # Training.csv → disease-symptom matrix
│   ├── step2_generate_drug_probabilities.py       # drugsComTrain_raw.tsv → drug-disease matrix
│   ├── step3_generate_evaluation_cases.py         # symbipredict_2022.csv → 12,300 test cases
│   ├── step4_run_experiments.py                   # Run all experiments, generate figures/tables
│   └── requirements.txt
├── data/
│   ├── raw/                                       # Place 3 source datasets here (see below)
│   │   └── README.md                              # Download instructions
│   └── generated/                                 # Pre-generated datasets (ready to use)
│       ├── disease_symptom_probabilities_unified_corrected.csv
│       ├── disease_drug_probabilities_unified_corrected.csv
│       └── test_cases_from_training_balanced_diverse.csv
└── outputs/                                       # Pre-computed results
    ├── fig1_accuracy_vs_training_size.pdf
    ├── fig2_lambda_effect.pdf
    ├── fig3_calibration_curves.pdf
    ├── fig4_confusion_matrix.pdf
    ├── fig5_roc_curves.pdf
    ├── fig6_treatment_recommendation.pdf
    ├── table1_dataset_statistics.csv
    ├── table2_main_results.csv
    ├── table3_lambda_ablation.csv
    └── table4_training_size.csv
```

## Data Sources

| # | Dataset | Source | Role | DOI/URL |
|---|---------|--------|------|---------|
| 1 | Disease Prediction Dataset | Kaggle (Kaushil268, 2019) | Knowledge base priors | [Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| 2 | Drug Review Dataset | UCI ML Repository (Kallumadi & Gräßer, 2018) | Treatment effectiveness | [DOI: 10.24432/C5SK5S](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) |
| 3 | SymbiPredict Dataset | Mendeley Data (Tucker, 2024) | Evaluation corpus | [DOI: 10.17632/dv5z3v2xyd.1](https://data.mendeley.com/datasets/dv5z3v2xyd/1) |

### Data Pipeline

```
Training.csv ──→ step1 ──→ disease_symptom_probabilities_unified_corrected.csv
                              (P(symptom|disease) ≥ 0.85, 271 associations)
                           ──→ provides naming convention for step3

symbipredict_2022.csv ──→ step3 ──→ test_cases_from_training_balanced_diverse.csv
                                     (12,300 cases, 300/disease, 3-8 symptoms each)
                                  ──→ step4 splits 50/50: KB construction + experiments

drugsComTrain_raw.tsv ──→ step2 ──→ disease_drug_probabilities_unified_corrected.csv
                                     (P(drug|disease) via review counts, 2,103 associations)

symbipredict_2022.csv ──→ step3 ──→ test_cases_from_training_balanced_diverse.csv
                                     (12,300 cases, 300/disease, 3-8 symptoms each)
```

## Reproducing Results

### Quick Start (using pre-generated data)

```bash
pip install -r code/requirements.txt
python code/step4_run_experiments.py
```

### Full Pipeline (from raw sources)

```bash
# 1. Download raw datasets into data/raw/ (see data/raw/README.md)
python code/step0_check_data.py

# 2. Generate curated datasets
python code/step1_generate_symptom_probabilities.py
python code/step2_generate_drug_probabilities.py
python code/step3_generate_evaluation_cases.py

# 3. Run experiments
python code/step4_run_experiments.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Hybrid framework accuracy | 95.7% |
| Improvement over classical lookup | +28.5 pp |
| Accuracy gain in data-scarce regime (N<600) | +5.1% |
| Calibration error reduction (ECE) | 62% |
| Crossover point | N ≈ 600 |

## Citation

```bibtex
@article{wagan2025bayesian,
  title={Scalable Bayesian Fusion with Log-Space Convex Combination 
         and Compositional Prior Construction},
  author={Wagan, Asif Ali and Narejo, Sanam and Talpur, Shahnawaz},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
