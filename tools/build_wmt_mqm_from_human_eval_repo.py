#!/usr/bin/env python3
"""
Build WMT22–24 MQM flat table from google/wmt-mqm-human-evaluation clone.

Uses *.avg_seg_scores.tsv when present (2022 pairs); otherwise aggregates raw
MQM TSV by summing severity weights (Major=-5, Minor=-1, No-error=0, etc.).
"""
from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# (year, pair, subdir under base, src, tgt)
SOURCES: list[tuple[str, str, str, str, str]] = [
    ("wmt22", "en-de", "generalMT2022/ende", "en", "de"),
    ("wmt22", "zh-en", "generalMT2022/zhen", "zh", "en"),
    ("wmt22", "en-ru", "generalMT2022/enru", "en", "ru"),
    ("wmt23", "en-de", "generalMT2023/ende", "en", "de"),
    ("wmt23", "zh-en", "generalMT2023/zhen", "zh", "en"),
    ("wmt23", "he-en", "generalMT2023/heen", "he", "en"),
    ("wmt24", "en-de", "generalMT2024", "en", "de"),
    ("wmt24", "en-es", "generalMT2024", "en", "es"),
    ("wmt24", "ja-zh", "generalMT2024", "ja", "zh"),
]

N_SEGMENTS: dict[tuple[str, str], int] = {
    ("wmt22", "en-de"): 10,
    ("wmt23", "en-de"): 10,
    ("wmt24", "en-de"): 10,
    ("wmt22", "zh-en"): 15,
    ("wmt23", "zh-en"): 15,
    ("wmt22", "en-ru"): 30,
    ("wmt23", "he-en"): 30,
    ("wmt24", "en-es"): 30,
    ("wmt24", "ja-zh"): 30,
}

WEIGHTS: dict[str, float] = {
    "no-error": 0.0,
    "no_error": 0.0,
    "neutral": 0.0,
    "minor": -1.0,
    "major": -5.0,
    "critical": -25.0,
}


def _norm_sev(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "no-error"
    t = str(s).strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    return t


def severity_weight(s: object) -> float:
    k = _norm_sev(s)
    return WEIGHTS.get(k, 0.0)


def find_tsv_pair_files(subdir: Path) -> tuple[Path | None, Path | None]:
    """Return (avg_seg_scores path or None, raw mqm path or None)."""
    if not subdir.is_dir():
        return None, None
    avg = None
    raw = None
    for p in subdir.iterdir():
        if p.suffix != ".tsv":
            continue
        name = p.name.lower()
        if "avg_seg_scores" in name:
            avg = p
        elif name.startswith("mqm_") and "avg" not in name and "3rating" not in name and "sxs" not in name:
            raw = p
    return avg, raw


def load_from_avg(path: Path) -> pd.DataFrame:
    # Hypothesis/source may contain stray tabs; skip malformed lines.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    # sys	hyp	domain	doc	source	ref	score	seg_id
    colmap = {
        "sys": "system_id",
        "hyp": "translation",
        "source": "original_text",
        "ref": "ref_text",
        "score": "human_mqm_score",
        "seg_id": "seg_id",
    }
    for k, v in colmap.items():
        if k not in df.columns:
            raise ValueError(f"{path}: missing column {k}, got {df.columns.tolist()}")
    out = df.rename(columns=colmap)
    out["human_mqm_score"] = pd.to_numeric(out["human_mqm_score"], errors="coerce")
    out["seg_id"] = out["seg_id"].astype(str)
    return out[["seg_id", "system_id", "original_text", "translation", "ref_text", "human_mqm_score"]]


def load_from_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    # Strip comment / doc rows
    df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0].astype(str).str.len() > 0)]
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")]

    cols = {c.lower(): c for c in df.columns}
    sys_c = cols.get("system")
    src_c = cols.get("source")
    tgt_c = cols.get("target")
    sev_c = cols.get("severity")
    # seg id: prefer globalSegId for 2023+
    seg_c = cols.get("globalsegid") or cols.get("global_seg_id") or cols.get("seg_id")
    if not all([sys_c, src_c, tgt_c, sev_c, seg_c]):
        raise ValueError(
            f"{path}: need system, source, target, severity, seg id; got {df.columns.tolist()}"
        )

    df["_w"] = df[sev_c].map(severity_weight)
    gcols = [sys_c, seg_c]
    agg = df.groupby(gcols, dropna=False, as_index=False).agg(
        human_mqm_score=("_w", "sum"),
        original_text=(src_c, "first"),
        translation=(tgt_c, "first"),
    )
    # Reference: optional column
    ref_c = cols.get("ref") or cols.get("reference")
    if ref_c and ref_c in df.columns:
        ref_first = df.groupby(gcols, dropna=False)[ref_c].first().reset_index()
        agg = agg.merge(ref_first, on=gcols, how="left")
        agg = agg.rename(columns={ref_c: "ref_text"})
    else:
        agg["ref_text"] = ""

    out = agg.rename(columns={sys_c: "system_id", seg_c: "seg_id"})
    out["seg_id"] = out["seg_id"].astype(str)
    return out[
        ["seg_id", "system_id", "original_text", "translation", "ref_text", "human_mqm_score"]
    ]


def stratified_seg_ids(
    seg_mean: pd.Series, n_target: int, rng: random.Random
) -> list[str]:
    valid = seg_mean.dropna()
    if len(valid) == 0:
        return []
    ids = valid.index.astype(str).tolist()
    if len(ids) <= n_target:
        return ids
    s = valid.reset_index()
    s.columns = ["seg_id", "mean_score"]
    try:
        s["quintile"] = pd.qcut(s["mean_score"], 5, labels=False, duplicates="drop")
    except Exception:
        s["quintile"] = 0
    picked: list[str] = []
    nq = int(s["quintile"].nunique())
    per = max(1, n_target // max(nq, 1))
    for q in sorted(s["quintile"].dropna().unique()):
        pool = s.loc[s["quintile"] == q, "seg_id"].astype(str).tolist()
        rng.shuffle(pool)
        picked.extend(pool[: min(per, len(pool))])
    rng.shuffle(picked)
    return picked[:n_target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=Path.home() / "Code" / "wmt-mqm-human-evaluation",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data_interim/wmt_mqm/sampled_segments.parquet"),
    )
    args = ap.parse_args()
    base: Path = args.base.expanduser().resolve()
    rng = random.Random(args.seed)

    parts: list[pd.DataFrame] = []
    for year, pair, rel, src_lang, tgt_lang in SOURCES:
        subdir = base / rel
        # 2024: files are directly under generalMT2024/ with names mqm_generalMT2024_ende.tsv
        if rel == "generalMT2024":
            lp_suffix = pair.replace("-", "")  # en-de -> ende
            tsv_glob = f"mqm_generalMT2024_{lp_suffix}.tsv"
            candidates = list(subdir.glob(tsv_glob)) if subdir.is_dir() else []
            raw_path = candidates[0] if candidates else None
            avg_path = None
            if raw_path and not raw_path.is_file():
                raw_path = None
        else:
            avg_path, raw_path = find_tsv_pair_files(subdir)

        n_target = N_SEGMENTS.get((year, pair), 30)

        if avg_path and avg_path.is_file():
            print(f"{year} {pair}: using AVG {avg_path.name}")
            df = load_from_avg(avg_path)
        elif raw_path and raw_path.is_file():
            print(f"{year} {pair}: using RAW {raw_path.name}")
            df = load_from_raw(raw_path)
        else:
            print(f"{year} {pair}: NO TSV under {subdir}")
            continue

        seg_mean = df.groupby("seg_id")["human_mqm_score"].mean()
        sampled = stratified_seg_ids(seg_mean, n_target, rng)
        d = df[df["seg_id"].isin(sampled)].copy()
        d["year"] = year
        d["pair"] = pair
        d["source_lang"] = src_lang
        d["target_lang"] = tgt_lang
        # fluency2 expects original_text / translation; already set
        parts.append(d)
        print(
            f"  -> {d['seg_id'].nunique()} segs × {d['system_id'].nunique()} systems = {len(d)} rows"
        )

    if not parts:
        raise SystemExit("No data loaded; check --base path.")

    out_df = pd.concat(parts, ignore_index=True)
    out_df["dataset"] = out_df["year"] + "_mqm_tsv"
    out_df["lp"] = out_df["pair"]
    out_df["doc_id"] = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
