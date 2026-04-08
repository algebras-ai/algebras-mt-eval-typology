#!/usr/bin/env python3
"""Bootstrap CI + permutation p for distance vs PA advantage; Holm on five primary tests."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

PRIMARY = [
    "lang2vec_syntax_average",
    "lang2vec_fam",
    "family_depth",
    "word_order",
    "lexical_inv_chrf",
]

N_BOOT = int(os.environ.get("TYPOLOGY_BOOT", "10000"))
N_PERM = int(os.environ.get("TYPOLOGY_PERM", "10000"))
SEED = int(os.environ.get("TYPOLOGY_SEED", "42"))


def distance_correlations(td: pd.DataFrame) -> pd.DataFrame:
    y = td["pa_advantage"].astype(float).to_numpy()
    corr_rows = []
    dist_cols = [c for c in td.columns if c.startswith("distance_")]
    for col in dist_cols:
        x = td[col].astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < 5:
            corr_rows.append(
                {
                    "distance_name": col.replace("distance_", "", 1),
                    "column": col,
                    "spearman": np.nan,
                    "spearman_p": np.nan,
                    "pearson": np.nan,
                    "pearson_p": np.nan,
                    "n": n,
                }
            )
            continue
        rho, pr = stats.spearmanr(x[mask], y[mask])
        r, pp = stats.pearsonr(x[mask], y[mask])
        corr_rows.append(
            {
                "distance_name": col.replace("distance_", "", 1),
                "column": col,
                "spearman": float(rho),
                "spearman_p": float(pr),
                "pearson": float(r),
                "pearson_p": float(pp),
                "n": n,
            }
        )
    return pd.DataFrame(corr_rows)


def bootstrap_and_perm(td: pd.DataFrame) -> pd.DataFrame:
    td = td.dropna(subset=["pa_advantage"]).copy()
    distance_cols = [c for c in td.columns if c.startswith("distance_")]
    rng_perm = np.random.default_rng(SEED)
    rows = []
    for z, dcol in enumerate(distance_cols):
        sub = td.dropna(subset=[dcol])
        if len(sub) < 5:
            continue
        x = sub[dcol].values
        y = sub["pa_advantage"].values
        n = len(sub)
        observed_rho, observed_p = stats.spearmanr(x, y)
        boot_rhos: list[float] = []
        rloc = np.random.default_rng(SEED + z * 1017)
        for _ in range(N_BOOT):
            idx = rloc.choice(n, size=n, replace=True)
            xb, yb = x[idx], y[idx]
            if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
                continue
            boot_rhos.append(float(stats.spearmanr(xb, yb).correlation))
        boot_rhos = np.array(boot_rhos, dtype=float)
        ci_low, ci_high = (
            np.percentile(boot_rhos, [2.5, 97.5]) if len(boot_rhos) > 100 else (np.nan, np.nan)
        )
        count = 0
        for _ in range(N_PERM):
            y_perm = rng_perm.permutation(y)
            rc = stats.spearmanr(x, y_perm).correlation
            if rc == rc and abs(rc) >= abs(observed_rho):
                count += 1
        perm_p = count / N_PERM
        rows.append(
            {
                "distance_name": dcol.replace("distance_", ""),
                "rho_observed": float(observed_rho),
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
                "spearman_p": float(observed_p),
                "perm_p": float(perm_p),
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def apply_holm(boot: pd.DataFrame) -> pd.DataFrame:
    boot = boot.copy()
    if "holm_p_primary" in boot.columns:
        boot = boot.drop(columns=["holm_p_primary"])
    primary_rows = boot[boot["distance_name"].isin(PRIMARY)].copy()
    supp_rows = boot[~boot["distance_name"].isin(PRIMARY)].copy()
    pvals = primary_rows["perm_p"].values
    reject, adjusted, _, _ = multipletests(pvals, method="holm")
    primary_rows["holm_p_primary"] = adjusted
    primary_rows["significant_holm_005"] = reject
    primary_rows["note"] = ""
    supp_rows["holm_p_primary"] = np.nan
    supp_rows["significant_holm_005"] = False
    supp_rows["note"] = "supplementary (excluded from correction)"
    return pd.concat([primary_rows, supp_rows], ignore_index=True)


def main() -> None:
    td = pd.read_csv(DATA / "typological_distances.csv")
    corr_df = distance_correlations(td)
    corr_df.to_csv(DATA / "distance_vs_advantage_correlations.csv", index=False)
    print("Distance vs advantage (Spearman):")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {DATA / 'distance_vs_advantage_correlations.csv'}")

    boot = bootstrap_and_perm(td)
    boot = apply_holm(boot)
    boot.to_csv(DATA / "bootstrap_ci_all_distances.csv", index=False)
    print("\nBootstrap + permutation (primary Holm):")
    print(
        boot[boot["distance_name"].isin(PRIMARY)][
            ["distance_name", "rho_observed", "ci_lower", "ci_upper", "perm_p", "holm_p_primary", "n"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"Wrote {DATA / 'bootstrap_ci_all_distances.csv'} (n_boot={N_BOOT}, n_perm={N_PERM})")


if __name__ == "__main__":
    main()
