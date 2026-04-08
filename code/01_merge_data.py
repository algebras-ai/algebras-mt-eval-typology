#!/usr/bin/env python3
"""Merge segment, judge, lexical, and COMET CSVs into one analysis-ready table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
KEYS = ["lp", "dataset", "doc_id", "seg_id", "system_id"]


def main() -> None:
    segments = pd.read_csv(DATA / "segments.csv")
    judge = pd.read_csv(DATA / "judge_scores.csv")
    lexical = pd.read_csv(DATA / "lexical_scores.csv")
    comet = pd.read_csv(DATA / "comet_scores.csv")

    j_extra = [c for c in judge.columns if c not in segments.columns]
    merged = segments.merge(judge[KEYS + j_extra], on=KEYS, how="inner", validate="one_to_one")

    lex_extra = [c for c in ("bleu", "chrf") if c in lexical.columns]
    merged = merged.merge(lexical[KEYS + lex_extra], on=KEYS, how="left", validate="one_to_one")

    comet_extra = [c for c in ("comet",) if c in comet.columns]
    merged = merged.merge(comet[KEYS + comet_extra], on=KEYS, how="left", validate="one_to_one")

    out = DATA / "merged.csv"
    merged.to_csv(out, index=False)
    print(f"Wrote {out} shape={merged.shape}")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    main()
