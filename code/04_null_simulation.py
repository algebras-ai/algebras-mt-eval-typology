#!/usr/bin/env python3
"""
Null pairwise-accuracy simulation: shuffle metric scores within each segment
(1000 permutations per metric; matches paper protocol).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from utils import (
    _subset_for_metric,
    build_disambiguated,
    pairwise_stats,
    permute_metric_within_segments,
    segment_row_lists,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

METRICS = ["fluency2_raw", "chrf", "comet", "bleu"]
N_PERM = int(os.environ.get("NULL_SIM_PERM", "1000"))
SEED = int(os.environ.get("NULL_SIM_SEED", "42"))


def main() -> None:
    df = pd.read_csv(DATA / "merged.csv")
    rng = np.random.default_rng(SEED)
    rows = []

    for metric in METRICS:
        if metric not in df.columns:
            continue
        sub = _subset_for_metric(df, metric)
        dm0 = build_disambiguated(sub, [metric])
        gkeys = [c for c in ("dataset", "lp", "doc_id", "seg_id") if c in dm0.columns]
        if len(gkeys) < 4:
            print(f"Skip {metric}: missing segment keys")
            continue

        actual_pa = pairwise_stats(dm0, metric)["pairwise_accuracy"]
        row_lists = segment_row_lists(dm0)
        null_pas: list[float] = []
        for i in range(N_PERM):
            dm_p = permute_metric_within_segments(dm0, metric, row_lists, rng)
            pa = pairwise_stats(dm_p, metric)["pairwise_accuracy"]
            null_pas.append(float(pa))
            if (i + 1) % max(1, N_PERM // 5) == 0:
                print(f"  {metric}: {i + 1}/{N_PERM}")

        arr = np.asarray(null_pas, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        p25, p975 = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        cmp = "below" if actual_pa < mean else "above" if actual_pa > mean else "equal"
        p_value = float(np.mean(arr >= actual_pa)) if cmp != "above" else float(np.mean(arr <= actual_pa))

        rows.append(
            {
                "metric": metric,
                "actual_pa": actual_pa,
                "null_mean": mean,
                "null_std": std,
                "null_p2.5": p25,
                "null_p97.5": p975,
                "actual_vs_null": cmp,
                "p_value": p_value,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "null_simulation_pa.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {DATA / 'null_simulation_pa.csv'} (n_perm={N_PERM})")


if __name__ == "__main__":
    main()
