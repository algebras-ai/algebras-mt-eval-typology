"""Shared evaluation helpers (pairwise accuracy, disambiguation, correlations)."""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

KEYS_AGG = ("dataset", "lp", "doc_id", "seg_id", "system_id")
GROUP_KEYS_PW = ("dataset", "lp", "doc_id", "seg_id")

JUDGE_METRIC_TIE_BAND = 0.5
COMET_METRIC_TIE_BAND = 0.01

JUDGE_METRIC_COLUMNS: frozenset[str] = frozenset(
    {
        "fluency2",
        "fluency2_raw",
        "fluency3_",
        "idiomatic",
        "collocational",
        "discourse",
        "pragmatic",
        "calque",
        "weighted_score",
        "fluency2_recalc_g6.25",
    }
)


def _human_tie(dh: float) -> bool:
    return bool(np.isclose(dh, 0.0, rtol=0.0, atol=1e-9))


def _metric_tie(metric_col: str, dv: float, *, band: float | None = None) -> bool:
    if band is not None:
        return abs(dv) <= band
    if metric_col in ("comet", "comet_kiwi", "comet_da"):
        return abs(dv) <= COMET_METRIC_TIE_BAND
    if metric_col in JUDGE_METRIC_COLUMNS or metric_col.startswith("fluency2_g_"):
        return abs(dv) <= JUDGE_METRIC_TIE_BAND
    return bool(np.isclose(dv, 0.0, rtol=0.0, atol=1e-12))


def pairwise_stats(
    dm: pd.DataFrame,
    metric_col: str,
    *,
    metric_tie_band: float | None = None,
) -> dict[str, Any]:
    gk = [c for c in GROUP_KEYS_PW if c in dm.columns]
    if len(gk) < 4:
        return {
            "metric": metric_col,
            "pairwise_accuracy": float("nan"),
            "n_pairs": 0,
            "tie_rate": float("nan"),
        }
    sub = dm.dropna(subset=["human_score", metric_col])
    human_skips = metric_skips = valid = agree = 0
    for _, g in sub.groupby(gk, dropna=False):
        if len(g) < 2:
            continue
        hs = pd.to_numeric(g["human_score"], errors="coerce").astype(float).tolist()
        ms = pd.to_numeric(g[metric_col], errors="coerce").astype(float).tolist()
        for ia, ib in itertools.combinations(range(len(g)), 2):
            dh = hs[ia] - hs[ib]
            if _human_tie(dh):
                human_skips += 1
                continue
            hm = 1 if dh > 0 else -1
            dv = ms[ia] - ms[ib]
            if _metric_tie(metric_col, dv, band=metric_tie_band):
                metric_skips += 1
                continue
            mv = 1 if dv > 0 else -1
            valid += 1
            agree += int(mv == hm)
    denom_pairs = human_skips + metric_skips + valid
    tie_rate = float((human_skips + metric_skips) / denom_pairs) if denom_pairs else float("nan")
    acc = float(agree / valid) if valid else float("nan")
    return {
        "metric": metric_col,
        "pairwise_accuracy": acc,
        "n_pairs": valid,
        "tie_rate": tie_rate,
    }


def build_disambiguated(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    base = df.copy()
    for k in KEYS_AGG:
        if k not in base.columns:
            base[k] = np.nan
    agg: dict = {"human_score": ("human_score", "mean")}
    for c in metric_cols:
        if c in base.columns:
            agg[c] = (c, "mean")
    return base.groupby(list(KEYS_AGG), as_index=False, dropna=False).agg(**agg)


def _subset_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric != "comet":
        return df
    s = pd.to_numeric(df["comet"], errors="coerce")
    if bool(s.notna().all()):
        return df
    return df.loc[s.notna()].copy()


def spearman_kendall(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    x = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df["human_score"], errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 25:
        return float("nan"), float("nan"), n
    r, _ = stats.spearmanr(x[m], y[m])
    t, _ = stats.kendalltau(x[m], y[m])
    return float(r), float(t), n


def segment_row_lists(dm: pd.DataFrame) -> list[np.ndarray]:
    gkeys = [c for c in GROUP_KEYS_PW if c in dm.columns]
    if len(gkeys) < 4:
        return []
    keys = pd.MultiIndex.from_frame(dm[gkeys].astype(str))
    codes, _ = pd.factorize(keys)
    by_seg: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(codes):
        by_seg[c].append(i)
    return [np.array(rows, dtype=np.int64) for rows in by_seg.values()]


def permute_metric_within_segments(
    dm: pd.DataFrame,
    metric: str,
    row_lists: list[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = dm.copy()
    vals = out[metric].to_numpy(copy=True)
    for rows in row_lists:
        chunk = vals[rows].copy()
        rng.shuffle(chunk)
        vals[rows] = chunk
    out[metric] = vals
    return out


def pairwise_for_weight_on_subset(
    df_sub: pd.DataFrame,
    scores: np.ndarray,
    metric_col: str = "weighted_score",
) -> float:
    d = df_sub.copy()
    d[metric_col] = scores
    dm = build_disambiguated(d, [metric_col])
    return float(pairwise_stats(dm, metric_col)["pairwise_accuracy"])
