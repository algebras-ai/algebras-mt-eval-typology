#!/usr/bin/env python3
"""
Join wmt25_participant_fluency2_full.parquet with WMT25 automatic_scores JSONL
(GEMBA-ESA-GPT4.1, MetricX-24-Hybrid-XL, XCOMET-XL, CometKiwi-XL).

Matching uses humeval doc_id (via seg_id line index): (language_pair, domain,
document_id, segment_index). This is required for stratified segment samples;
plain sort-by-seg_id within pair does not match automatic_scores row order.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METRICS = {
    "gemba_score": "GEMBA-ESA-GPT4.1",
    "metricx_score": "MetricX-24-Hybrid-XL",
    "xcomet_score": "XCOMET-XL",
    "cometKiwi_score": "CometKiwi-XL",
}


def parquet_pair_to_language_pair(pair: str) -> str:
    if "→" not in pair:
        return pair
    src, tgt = pair.split("→", 1)
    return f"{src.lower()}-{tgt}"


def parse_doc_id(doc_id: str) -> tuple[str, str, str, int]:
    parts = str(doc_id).split("_#_")
    if len(parts) < 4:
        return "", "", "", -1
    lp, domain, document_id, seg_s = parts[0], parts[1], parts[2], parts[3]
    try:
        pos = int(seg_s)
    except ValueError:
        pos = -1
    return lp, domain, document_id, pos


def load_seg_meta(humeval_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with humeval_path.open(encoding="utf-8") as f:
        for seg_id, line in enumerate(f):
            o = json.loads(line)
            lp, domain, document_id, pos = parse_doc_id(o.get("doc_id", ""))
            rows.append(
                {
                    "seg_id": seg_id,
                    "language_pair": lp,
                    "domain": domain,
                    "document_id": document_id,
                    "pos_in_doc": pos,
                }
            )
    return pd.DataFrame(rows)


def load_automatic_scores(auto_dir: Path) -> dict[tuple[str, str, str, str, int], dict[str, float]]:
    """
    Key: (system_id, language_pair, domain, document_id, pos_in_doc)
    Value: metric column name -> score
    """
    out: dict[tuple[str, str, str, str, int], dict[str, float]] = {}
    for path in sorted(auto_dir.glob("*.jsonl")):
        system_id = path.stem
        with path.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                lp = str(o.get("language_pair", ""))
                domain = str(o.get("domain", ""))
                document_id = str(o.get("document_id", ""))
                ms = o.get("metric_scores") or {}
                # length from first present metric list
                lengths = [len(v) for v in ms.values() if isinstance(v, list)]
                if not lengths:
                    continue
                n = min(lengths)
                for i in range(n):
                    rec: dict[str, float] = {}
                    for col, mname in METRICS.items():
                        arr = ms.get(mname)
                        if not isinstance(arr, list) or i >= len(arr):
                            continue
                        try:
                            rec[col] = float(arr[i])
                        except (TypeError, ValueError):
                            pass
                    if rec:
                        out[(system_id, lp, domain, document_id, i)] = rec
    return out


def verify_sample(
    df: pd.DataFrame,
    seg_meta: pd.DataFrame,
    auto: dict[tuple[str, str, str, str, int], dict[str, float]],
    pair: str,
    system_id: str,
    n: int = 12,
) -> None:
    lp = parquet_pair_to_language_pair(pair)
    sub = (
        df[(df["pair"] == pair) & (df["system_id"] == system_id)]
        .merge(seg_meta, on="seg_id", how="left")
        .sort_values("seg_id")
    )
    print(f"\n=== Alignment check: pair={pair} ({lp}) system={system_id} (first {n} rows) ===\n")
    print(f"{'seg_id':>6}  {'pos':>3}  {'mt_text[:50]':50}  GEMBA")
    for _, r in sub.head(n).iterrows():
        key = (
            system_id,
            str(r.get("language_pair", "")),
            str(r.get("domain", "")),
            str(r.get("document_id", "")),
            int(r["pos_in_doc"]) if pd.notna(r["pos_in_doc"]) else -1,
        )
        g = auto.get(key, {}).get("gemba_score", float("nan"))
        mt = str(r.get("mt_text", ""))[:50].replace("\n", " ")
        print(f"{int(r['seg_id']):6d}  {int(r['pos_in_doc']):3d}  {mt:50}  {g}")


def join_metrics(
    df: pd.DataFrame,
    seg_meta: pd.DataFrame,
    auto: dict[tuple[str, str, str, str, int], dict[str, float]],
) -> pd.DataFrame:
    m = df.merge(seg_meta, on="seg_id", how="left")
    gemba, metricx, xcomet, kiwi = [], [], [], []
    matched = []
    for _, r in m.iterrows():
        key = (
            str(r["system_id"]),
            str(r["language_pair"]),
            str(r["domain"]),
            str(r["document_id"]),
            int(r["pos_in_doc"]) if pd.notna(r["pos_in_doc"]) else -1,
        )
        rec = auto.get(key)
        matched.append(rec is not None)
        if rec is None:
            gemba.append(float("nan"))
            metricx.append(float("nan"))
            xcomet.append(float("nan"))
            kiwi.append(float("nan"))
        else:
            gemba.append(rec.get("gemba_score", float("nan")))
            metricx.append(rec.get("metricx_score", float("nan")))
            xcomet.append(rec.get("xcomet_score", float("nan")))
            kiwi.append(rec.get("cometKiwi_score", float("nan")))
    out = m.drop(columns=["language_pair", "domain", "document_id", "pos_in_doc"], errors="ignore")
    out["gemba_score"] = gemba
    out["metricx_score"] = metricx
    out["xcomet_score"] = xcomet
    out["cometKiwi_score"] = kiwi
    out["_auto_matched"] = matched
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data_interim" / "runs" / "wmt25_verify" / "wmt25_participant_fluency2_full.parquet",
    )
    ap.add_argument(
        "--humeval",
        type=Path,
        default=ROOT / "data_raw" / "wmt25" / "wmt25-genmt-humeval.jsonl",
    )
    ap.add_argument(
        "--auto-dir",
        type=Path,
        default=Path(os.environ.get("WMT25_AUTO_DIR", "/tmp/wmt25_auto/all")),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data_interim" / "wmt25_full_analysis" / "merged_all_metrics.parquet",
    )
    ap.add_argument("--verify-pair", default="EN→uk_UA")
    ap.add_argument("--verify-system", default="Claude-4")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    print("Loading seg meta from humeval …", flush=True)
    seg_meta = load_seg_meta(args.humeval)
    print("Loading automatic scores …", flush=True)
    auto = load_automatic_scores(args.auto_dir)
    print(f"Automatic score entries: {len(auto):,}", flush=True)

    df = pd.read_parquet(args.parquet)
    n_total = len(df)

    if not args.skip_verify:
        verify_sample(df, seg_meta, auto, args.verify_pair, args.verify_system)

    print("\nFull join …", flush=True)
    out = join_metrics(df, seg_meta, auto)
    matched = out["_auto_matched"].sum()
    out = out.drop(columns=["_auto_matched"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print("\n--- Report ---")
    print(f"Total rows (our data): {n_total}")
    print(f"Rows with automatic lookup key match: {int(matched)}")
    print(f"Rows with no match: {int(n_total - matched)}")
    for c in ["gemba_score", "metricx_score", "xcomet_score", "cometKiwi_score"]:
        na = out[c].isna().sum()
        print(f"  Missing {c}: {int(na)} ({100 * na / n_total:.2f}%)")

    print("\nSample of 5 joined rows (non-null gemba):")
    samp = out[out["gemba_score"].notna()].head(5)
    cols = [
        "seg_id",
        "pair",
        "system_id",
        "gemba_score",
        "metricx_score",
        "xcomet_score",
        "cometKiwi_score",
    ]
    cols = [c for c in cols if c in samp.columns]
    print(samp[cols].to_string(index=False))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
