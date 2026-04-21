#!/usr/bin/env python3
"""
Merge precomputed GEMBA segment scores from MicrosoftTranslator/GEMBA
(mt-metrics-eval-v2) into data_interim/wmt_mqm/all_metrics_mqm.parquet.

The upstream repo currently ships only WMT22 under mt-metrics-eval-v2.
We map segment id -> score index as int(seg_id) - 1 (1-based global seg ids).

File naming (GPT-4 + refA, matching README recommendations):
  gemba_da  <- GEMBA-GPT4-DA-refA.seg.score
  gemba_mqm <- GEMBA-GPT4-SQM-refA.seg.score   (GEMBA-SQM in gemba code)
  gemba_esa <- not present in published scores -> column left all-NaN
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_seg_scores(path: Path) -> dict[str, list[float]]:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["system", "score"],
        dtype={"system": str},
    )
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    out: dict[str, list[float]] = {}
    for sys_name, grp in df.groupby("system", sort=False):
        out[str(sys_name)] = grp["score"].tolist()
    return out


def seg_index(seg_id: object) -> int | None:
    try:
        s = str(seg_id).strip()
        if not s or s.lower() == "nan":
            return None
        v = int(float(s))
        return v - 1
    except (ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gemba-root",
        type=Path,
        default=Path.home() / "Code" / "GEMBA" / "mt-metrics-eval-v2",
        help="Path to mt-metrics-eval-v2 inside GEMBA clone",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data_interim/wmt_mqm/all_metrics_mqm.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/wmt_mqm/all_metrics_mqm.parquet"),
    )
    args = ap.parse_args()
    root: Path = args.gemba_root.expanduser().resolve()

    df = pd.read_parquet(args.input)
    for col in ("gemba_mqm", "gemba_da", "gemba_esa"):
        if col not in df.columns:
            df[col] = pd.NA

    # (year, pair) -> relative metric dir
    targets = {
        ("wmt22", "en-de"): root / "wmt22" / "metric-scores" / "en-de",
        ("wmt22", "zh-en"): root / "wmt22" / "metric-scores" / "zh-en",
        ("wmt22", "en-ru"): root / "wmt22" / "metric-scores" / "en-ru",
    }

    metric_files = {
        "gemba_da": "GEMBA-GPT4-DA-refA.seg.score",
        "gemba_mqm": "GEMBA-GPT4-SQM-refA.seg.score",
    }

    for (year, pair), metric_dir in targets.items():
        if not metric_dir.is_dir():
            print(f"Skip {year} {pair}: missing {metric_dir}")
            continue

        for col, fname in metric_files.items():
            path = metric_dir / fname
            if not path.is_file():
                print(f"Skip {year} {pair} {col}: no {path.name}")
                continue

            scores_by_sys = load_seg_scores(path)
            mask = (df["year"] == year) & (df["pair"] == pair)
            idxs = df.index[mask]
            filled = 0
            missing_sys = 0
            bad_idx = 0
            for i in idxs:
                sys_id = str(df.at[i, "system_id"])
                si = seg_index(df.at[i, "seg_id"])
                if si is None:
                    bad_idx += 1
                    continue
                if sys_id not in scores_by_sys:
                    missing_sys += 1
                    continue
                arr = scores_by_sys[sys_id]
                if si < 0 or si >= len(arr):
                    bad_idx += 1
                    continue
                df.at[i, col] = arr[si]
                filled += 1

            print(
                f"{year} {pair} {col} <- {fname}: filled {filled} rows "
                f"(missing_sys={missing_sys}, bad_idx={bad_idx})"
            )

    # gemba_esa: no published *.seg.score in GEMBA mt-metrics-eval-v2
    print("gemba_esa: no GEMBA-GPT4-ESA-refA (or similar) in repo; column unchanged")

    for col in ("gemba_mqm", "gemba_da", "gemba_esa"):
        n = df[col].notna().sum()
        print(f"  {col} coverage: {n}/{len(df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
