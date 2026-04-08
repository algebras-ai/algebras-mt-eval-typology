#!/usr/bin/env python3
"""Per-language-pair pairwise accuracy (Table 2 / fig2 input)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from utils import _subset_for_metric, build_disambiguated, pairwise_stats

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

METRICS = ["fluency2_raw", "fluency3_", "chrf", "bleu", "comet"]


def main() -> None:
    df = pd.read_csv(DATA / "merged.csv")
    rows = []
    for lp, chunk in df.groupby("lp", dropna=False):
        for metric in METRICS:
            if metric not in chunk.columns:
                continue
            sub = _subset_for_metric(chunk, metric)
            dm = build_disambiguated(sub, [metric])
            s = pairwise_stats(dm, metric)
            rows.append(
                {
                    "lp": lp,
                    "metric": metric,
                    "pairwise_accuracy": s["pairwise_accuracy"],
                    "n_pairs": s["n_pairs"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "pairwise_accuracy_by_lp.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {DATA / 'pairwise_accuracy_by_lp.csv'} rows={len(out)}")


if __name__ == "__main__":
    main()
