# Pancreatic Cancer Prognosis using Clinical and Radiomic Data
UE24CS352A Machine Learning - Mini-Project (Problem #6)
Team: Pratik Patil(PES2UG24CS372), Ruhika Kolla(PES2UG24CS908) | Section: G

Survival analysis (Cox proportional hazards) on clinical, radiomic and combined features, with
Naive Bayes risk-threshold selection and SMOTE augmentation. Based on A. Jamalian, Stanford CS229 (2020):
https://cs229.stanford.edu/proj2020spr/report/Jamalian.pdf

## Dataset note
The original Stanford dataset is under NDA and not public. `src/generate_data.py` creates a **synthetic cohort
with the same structure** (80 patients, 12 clinical + 900 radiomic features, right-censored survival).
Results show the pipeline works; they are not clinical findings. To use real data, provide a CSV with the same
columns (`survival_days`, `event`, clinical and radiomic columns) at `data/synthetic_pancreatic.csv`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/generate_data.py     # creates data/synthetic_pancreatic.csv
python src/pipeline.py          # NB threshold search, SMOTE, Cox models, KM plots -> results/
python src/demo.py P003         # live demo: score one patient
python src/make_writeup.py      # rebuilds the PDF write-up from results/
```

## Structure
```
src/generate_data.py   synthetic cohort
src/pipeline.py        full ML pipeline (SMOTE + feature filtering inside CV folds)
src/demo.py            patient risk scoring demo
src/make_writeup.py    builds MiniProject_Writeup.pdf
results/               metrics.json and figures
docs/                  write-up PDF and slides
```
