"""Pancreatic cancer survival pipeline (replicates Jamalian, CS229 2020).

Steps: Naive Bayes risk-threshold selection -> SMOTE augmentation ->
Cox PH on clinical / radiomic / combined features -> 10-fold C-index ->
Kaplan-Meier high/low risk groups + log-rank test.
SMOTE and radiomic feature filtering are done INSIDE each CV training fold
to avoid leakage into the held-out fold.
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
SEED = 42
PEN = 0.1                      # L2 penalizer (l1_ratio = 0), as in the paper
CLIN = ["Radiation Therapy", "NCCN Resectability", "Chemotherapy Time", "Tumor Location",
        "Stage Grouping", "Primary Tumor (T)", "KPS", "Gender", "Tumor Width",
        "Tumor Depth", "Tumor Height", "Tumor Volume"]
NB_FEATS = ["Stage Grouping", "Chemotherapy Time", "Tumor Width", "NCCN Resectability", "Radiation Therapy"]


def load(path="data/synthetic_pancreatic.csv"):
    df = pd.read_csv(path)
    rad = [c for c in df.columns if c not in CLIN + ["patient_id", "survival_days", "event"]]
    return df, rad


def standardize(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return (train[cols] - mu) / sd, (test[cols] - mu) / sd


# ---------- 1. Naive Bayes threshold selection ----------
def nb_threshold_search(df, thresholds=(240, 270, 300, 330, 360)):
    X = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile").fit_transform(df[NB_FEATS]).astype(int)
    out = {}
    for th in thresholds:
        y = (df.survival_days < th).astype(int)          # 1 = high risk
        if y.sum() < 10:
            out[th] = float("nan"); continue
        accs = []
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y):
            m = CategoricalNB(alpha=1.0, min_categories=3).fit(X[tr], y.iloc[tr])   # Laplace smoothing
            accs.append(balanced_accuracy_score(y.iloc[te], m.predict(X[te])))
        out[th] = float(np.mean(accs))
    return out


# ---------- 2. SMOTE on survival data ----------
def smote_augment(train, feats, threshold):
    """Oversample the high-risk minority; time/event are interpolated with the features."""
    y = (train.survival_days < threshold).astype(int)
    if y.sum() < 6 or (1 - y).sum() < 6:
        return train
    cols = feats + ["survival_days", "event"]
    Xr, yr = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(train[cols], y)
    aug = pd.DataFrame(Xr, columns=cols)
    aug["event"] = (aug["event"] >= 0.5).astype(int)
    return aug


# ---------- 3. Cox model with fold-internal feature filtering ----------
def fit_cox(train, feats, is_radiomic):
    cph = CoxPHFitter(penalizer=PEN, l1_ratio=0.0)
    cph.fit(train[feats + ["survival_days", "event"]], "survival_days", "event")
    if is_radiomic:   # drop features whose 95% CI width on log(HR) > 1 (paper: 410/900 removed)
        s = cph.summary
        width = s["coef upper 95%"] - s["coef lower 95%"]
        keep = [f for f in feats if width[f] <= 1.0]
        keep = keep if len(keep) >= 5 else feats
        cph = CoxPHFitter(penalizer=PEN, l1_ratio=0.0)
        cph.fit(train[keep + ["survival_days", "event"]], "survival_days", "event")
        feats = keep
    return cph, feats


def cv_cindex(df, feats, is_radiomic, threshold, k=10):
    strat = (df.survival_days < threshold).astype(int)
    scores, kept = [], []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(df, strat):
        a, b = df.iloc[tr].copy(), df.iloc[te].copy()
        a[feats], b[feats] = standardize(a, b, feats)
        a = smote_augment(a, feats, threshold)
        cph, used = fit_cox(a, feats, is_radiomic)
        scores.append(concordance_index(b.survival_days, -cph.predict_partial_hazard(b[used]), b.event))
        kept.append(len(used))
    return float(np.mean(scores)), float(np.std(scores)), int(np.mean(kept))


def main():
    df, rad = load()
    res = {}
    print("Naive Bayes threshold search...")
    nb = nb_threshold_search(df); res["nb_balanced_acc"] = nb
    best = max(nb, key=lambda t: -1 if np.isnan(nb[t]) else nb[t]); res["best_threshold"] = best
    print(nb, "-> best", best)

    # radiomic pre-screen: top-150 by |univariate corr| inside full data is leakage, so use variance filter only
    rad_use = list(df[rad].var().sort_values(ascending=False).index[:150])
    sets = {"Clinical": (CLIN, False), "Radiomic": (rad_use, True), "Clinical + Radiomic": (CLIN + rad_use, True)}
    res["cindex"] = {}
    for name, (feats, isr) in sets.items():
        m, s, nk = cv_cindex(df, feats, isr, best)
        res["cindex"][name] = {"mean": m, "std": s, "n_features_in": len(feats), "n_features_used": nk}
        print(f"{name:22s} C-index {m:.3f} +/- {s:.3f}  (features {len(feats)} -> {nk})")

    # ---- final model on combined set (SMOTE-augmented) for KM + coefficients ----
    feats = CLIN + rad_use
    d = df.copy(); d[feats], _ = standardize(d, d, feats)
    aug = smote_augment(d, feats, best)
    cph, used = fit_cox(aug, feats, True)
    d_scores = cph.predict_partial_hazard(d[used])
    hi = d_scores > d_scores.median()
    lr = logrank_test(d.survival_days[hi], d.survival_days[~hi], d.event[hi], d.event[~hi])
    res["logrank"] = {"statistic": float(lr.test_statistic), "p": float(lr.p_value)}
    print("log-rank", res["logrank"])

    # figures
    fig, ax = plt.subplots(figsize=(4.4, 3))
    for lab, mask, c in [("Low risk", ~hi, "tab:blue"), ("High risk", hi, "tab:orange")]:
        KaplanMeierFitter().fit(d.survival_days[mask], d.event[mask], label=lab).plot_survival_function(ax=ax, color=c, ci_show=False)
    ax.set_xlabel("Survival days"); ax.set_ylabel("Survival probability"); ax.set_title("Kaplan-Meier by predicted risk", fontsize=9)
    fig.tight_layout(); fig.savefig("results/km_curves.png", dpi=200); plt.close(fig)

    clin_only = cph.summary.loc[[c for c in CLIN if c in cph.summary.index]]
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.errorbar(clin_only["coef"], range(len(clin_only)),
                xerr=[clin_only["coef"] - clin_only["coef lower 95%"], clin_only["coef upper 95%"] - clin_only["coef"]],
                fmt="s", color="k", capsize=2, ms=3)
    ax.axvline(0, ls="--", c="gray", lw=.8); ax.set_yticks(range(len(clin_only))); ax.set_yticklabels(clin_only.index, fontsize=6)
    ax.set_xlabel("log(HR) with 95% CI", fontsize=8); ax.set_title("Clinical coefficients (combined model)", fontsize=9)
    fig.tight_layout(); fig.savefig("results/coefficients.png", dpi=200); plt.close(fig)

    labels = list(res["cindex"]); vals = [res["cindex"][k]["mean"] for k in labels]
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"]); ax.set_ylim(0.4, 1)
    for i, v in enumerate(vals): ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("10-fold C-index"); ax.set_title("Model comparison", fontsize=9); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig("results/cindex_comparison.png", dpi=200); plt.close(fig)

    json.dump(res, open("results/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
