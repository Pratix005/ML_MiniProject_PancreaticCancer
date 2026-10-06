"""Live demo: train the combined Cox model and score a patient's risk.
Usage: python src/demo.py [patient_id]   (default P000)
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from pipeline import load, CLIN, standardize, smote_augment, fit_cox

pid = sys.argv[1] if len(sys.argv) > 1 else "P000"
df, rad = load()
feats = CLIN + list(df[rad].var().sort_values(ascending=False).index[:150])
d = df.copy(); d[feats], _ = standardize(d, d, feats)
cph, used = fit_cox(smote_augment(d, feats, 270), feats, True)
score = cph.predict_partial_hazard(d[used])
row = d[d.patient_id == pid]
pct = (score < float(cph.predict_partial_hazard(row[used]).iloc[0])).mean() * 100
sf = cph.predict_survival_function(row[used], times=[180, 365, 540]).iloc[:, 0]
print(f"Patient {pid}: risk percentile {pct:.0f}  ->  {'HIGH' if pct > 50 else 'LOW'} risk")
print("Predicted survival probability:", {f"{int(t)}d": round(float(p), 2) for t, p in sf.items()})
print(f"Observed: {int(row.survival_days.iloc[0])} days, event={int(row.event.iloc[0])}")
