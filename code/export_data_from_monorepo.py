#!/usr/bin/env python3
"""
One-time export of CSVs from the benchmarks monorepo into this repo's data/.

Usage (from this repository root):

  MONO=/path/to/benchmarks python code/export_data_from_monorepo.py

Requires the evaluation checkpoint parquet and typology tables from the
internal WMT fluency evaluation run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def main() -> None:
    mono = os.environ.get("MONO", "").strip()
    if not mono:
        print("Set MONO to the benchmarks repository root.", file=sys.stderr)
        sys.exit(1)
    mono = os.path.expanduser(mono)
    base = pd.read_parquet(
        f"{mono}/notebooks/wmt/fluency2/outputs/fluency2wmt_evaluation_report/checkpoints/02_with_lexical.parquet"
    )
    comet = pd.read_csv(
        f"{mono}/notebooks/wmt/fluency2/outputs/fluency2wmt_evaluation_report/checkpoints/comet_scores_partial.csv"
    )
    td = pd.read_csv(
        f"{mono}/notebooks/wmt/fluency2/outputs/fluency2wmt_evaluation_report/tables/typological_distances.csv"
    )
    rp = pd.read_csv(
        f"{mono}/notebooks/wmt/fluency2/outputs/fluency2wmt_evaluation_report/tables/resource_proxy_data.csv"
    )

    base = base.copy()
    if "comet" not in base.columns:
        base["comet"] = float("nan")
    base.loc[comet["df_index"].values, "comet"] = comet["comet"].values

    seg_cols = ["lp", "dataset", "doc_id", "seg_id", "system_id", "src_text", "mt_text", "human_score"]
    seg_cols = [c for c in seg_cols if c in base.columns]
    segments = base[seg_cols].rename(columns={"src_text": "source_text", "mt_text": "target_text"})
    DATA.mkdir(parents=True, exist_ok=True)
    segments.to_csv(DATA / "segments.csv", index=False)

    judge_cols = ["lp", "dataset", "doc_id", "seg_id", "system_id"]
    flu_cols = [
        c
        for c in base.columns
        if c
        in (
            "idiomatic",
            "collocational",
            "discourse",
            "pragmatic",
            "calque",
            "fluency2",
            "fluency2_raw",
            "fluency2_recalc_g6.25",
            "fluency3_",
            "confidence",
        )
    ]
    judge_cols = [c for c in judge_cols if c in base.columns] + flu_cols
    base[judge_cols].to_csv(DATA / "judge_scores.csv", index=False)

    lex_cols = [c for c in ["lp", "dataset", "doc_id", "seg_id", "system_id", "bleu", "chrf"] if c in base.columns]
    base[lex_cols].to_csv(DATA / "lexical_scores.csv", index=False)

    comet_cols = [c for c in ["lp", "dataset", "doc_id", "seg_id", "system_id", "comet"] if c in base.columns]
    base[comet_cols].dropna(subset=["comet"]).to_csv(DATA / "comet_scores.csv", index=False)

    td.to_csv(DATA / "typological_distances.csv", index=False)
    rp.to_csv(DATA / "resource_proxy.csv", index=False)

    if "model_name" in base.columns:
        systems = (
            base.groupby(["system_id", "model_name"], dropna=False).size().reset_index()[["system_id", "model_name"]]
        )
    else:
        systems = base[["system_id"]].drop_duplicates()
    systems.sort_values("system_id").reset_index(drop=True).to_csv(DATA / "systems.csv", index=False)

    print("Exported CSVs to", DATA)


if __name__ == "__main__":
    main()
