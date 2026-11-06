# Raw data

| File | Source | Notes |
|---|---|---|
| `kaggle_disease_symptom_training.csv` | Kaggle, *Disease Prediction Using Machine Learning* by kaushil268, file `Training.csv` | 4,920 rows, 41 diseases, 132 symptom columns plus `prognosis`. Unchanged copy. One column (`fluid_overload`) is an all-zero duplicate of `fluid_overload.1`; `kbfusion.data` fixes this and three misspelled column names. |
| `symbipredict_2022.csv` | SymbiPredict, Mendeley Data V1, DOI 10.17632/dv5z3v2xyd.1 (Tucker, 2024) | 4,961 rows. Contains the same 304 symptom patterns as the Kaggle file; used only to document this. Unchanged copy. |
| `drugscom_reviews.csv.gz` | UCI Machine Learning Repository, *Drug Review Dataset (Drugs.com)*, DOI 10.24432/C5SK5S | Train and test splits combined (215,063 reviews). Columns kept: `review_id, drugName, condition, rating, usefulCount, date, split`. The free-text review column was dropped to keep the file small. Licence CC BY 4.0. |

Version 1 of this repository shipped a `drugsComTrain_raw.csv` in which the names
of about 19,900 combination products had been cut at the first "/" (for example
"Abacavir / dolutegravir / lamivudine" became "Abacavir "). The file here was
rebuilt from the original tab-separated train and test files.

Some condition labels are truncated in the original UCI data itself (for example
"Peptic Ulce", "Typhoid Feve"). They are matched as they appear.
