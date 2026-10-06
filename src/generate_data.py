"""Generate a SYNTHETIC stand-in for the Stanford pancreatic cancer cohort.

The real dataset (80 patients, 900 QIFP radiomic features) is under NDA and not
public, so this script builds a cohort with the same *structure*: 80 patients,
12 clinical features, 900 correlated radiomic features, right-censored survival.
A hidden hazard model links a few features to survival so the pipeline has
signal to find. Results are therefore illustrative, not clinical findings.
"""
import numpy as np
import pandas as pd

N_RADIOMIC = 900


def generate(n=80, seed=42):
    rng = np.random.default_rng(seed)
    stage = rng.choice([0, 1, 2, 3], n, p=[0.04, 0.26, 0.10, 0.60])   # IB, IIA, IIB, III
    clin = pd.DataFrame({
        "Radiation Therapy": rng.integers(0, 2, n),
        "NCCN Resectability": rng.integers(0, 3, n),
        "Chemotherapy Time": rng.integers(0, 2, n),
        "Tumor Location": rng.integers(0, 3, n),
        "Stage Grouping": stage,
        "Primary Tumor (T)": np.clip(stage + rng.integers(-1, 2, n), 0, 3),
        "KPS": rng.choice([60, 70, 80, 90, 100], n),
        "Gender": rng.integers(0, 2, n),
        "Tumor Width": rng.gamma(6, 5, n),
    })
    clin["Tumor Depth"] = clin["Tumor Width"] * rng.uniform(0.7, 1.1, n)
    clin["Tumor Height"] = clin["Tumor Width"] * rng.uniform(0.6, 1.0, n)
    clin["Tumor Volume"] = clin[["Tumor Width", "Tumor Depth", "Tumor Height"]].prod(axis=1) / 1000

    # 900 radiomic features: 30 latent factors -> blocks of highly correlated features
    k = 30
    latent = rng.normal(size=(n, k))
    load = rng.normal(size=(k, N_RADIOMIC)) * (rng.random((k, N_RADIOMIC)) < 0.15)
    load[0, :40] = 1.0; load[1, 40:80] = 1.0     # make the two prognostic factors visible
    rad = latent @ load + 0.6 * rng.normal(size=(n, N_RADIOMIC))
    prefixes = ["wavelet-LLH_glrlm", "wavelet-HHH_glszm", "log-sigma-5-mm-3D_firstorder",
                "original_shape", "original_glcm", "wavelet-HLL_firstorder"]
    cols = [f"{prefixes[i % len(prefixes)]}_f{i:03d}" for i in range(N_RADIOMIC)]
    rad = pd.DataFrame(rad, columns=cols)

    # hidden hazard: chemo protective, T-stage / stage harmful, two radiomic factors
    z = lambda s: (s - s.mean()) / s.std()
    lin = (-1.0 * clin["Chemotherapy Time"] + 0.35 * z(clin["Primary Tumor (T)"])
           + 0.25 * z(clin["Stage Grouping"]) - 0.15 * z(clin["KPS"])
           + 0.55 * z(latent[:, 0]) + 0.45 * z(latent[:, 1]))
    t = rng.weibull(2.0, n) * 380 * np.exp(-0.5 * lin)
    censor = rng.uniform(150, 1300, n)
    days = np.minimum(t, censor).round().clip(30)
    event = (t <= censor).astype(int)
    out = pd.concat([clin, rad], axis=1)
    out.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])
    out["survival_days"] = days
    out["event"] = event
    return out


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/synthetic_pancreatic.csv", index=False)
    print(df.shape, "events:", df.event.sum(), "median days:", df.survival_days.median())
