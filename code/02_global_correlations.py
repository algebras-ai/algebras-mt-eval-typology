#!/usr/bin/env python3
"""Table 1: global Spearman, Kendall, pairwise accuracy per metric."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from utils import (
    _subset_for_metric,
    build_disambiguated,
    pairwise_stats,
    spearman_kendall,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

METRICS_ORDER = [
    "comet",
    "fluency2_raw",
    "fluency3_",
    "fluency2",
    "idiomatic",
    "collocational",
    "calque",
    "chrf",
    "bleu",
]


def main() -> None:
    df = pd.read_csv(DATA / "merged.csv")
    rows_corr = []
    rows_pa = []

    for metric in METRICS_ORDER:
        if metric not in df.columns:
            continue
        sub = _subset_for_metric(df, metric)
        sp, ke, n = spearman_kendall(sub, metric)
        rows_corr.append({"metric": metric, "spearman": sp, "kendall": ke, "n": n})

        dm = build_disambiguated(sub, [metric])
        ps = pairwise_stats(dm, metric)
        rows_pa.append(
            {
                "metric": metric,
                "pairwise_accuracy": ps["pairwise_accuracy"],
                "n_pairs": ps["n_pairs"],
                "tie_rate": ps["tie_rate"],
            }
        )

    corr = pd.DataFrame(rows_corr)
    pwa = pd.DataFrame(rows_pa)
    merged = corr.merge(pwa, on="metric", how="outer")
    merged["sort_key"] = merged["metric"].map({m: i for i, m in enumerate(METRICS_ORDER)}).fillna(99)
    merged = merged.sort_values("sort_key").drop(columns=["sort_key"])
    corr.to_csv(DATA / "correlations_all_variants.csv", index=False)
    pwa.to_csv(DATA / "pairwise_accuracy_all_variants.csv", index=False)

    print(merged.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {DATA / 'correlations_all_variants.csv'}")


if __name__ == "__main__":
    main()
