#!/usr/bin/env python3
"""Leave-one-LP-out CV over sub-dimension weight grid (same grid as paper)."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from utils import pairwise_for_weight_on_subset

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

SUB_DIMS = ["idiomatic", "collocational", "discourse", "pragmatic", "calque"]
CURRENT_WEIGHTS = (0.40, 0.28, 0.09, 0.06, 0.17)


def weight_grid_5d(resolution: int = 20):
    n_dims = 5
    for combo in itertools.combinations(range(resolution + n_dims - 1), n_dims - 1):
        weights: list[int] = []
        prev = -1
        for c in combo:
            weights.append(c - prev - 1)
            prev = c
        weights.append(resolution + n_dims - 2 - prev)
        yield tuple(w / resolution for w in weights)


def weight_grid_3d(resolution: int = 20):
    n_dims = 3
    for combo in itertools.combinations(range(resolution + n_dims - 1), n_dims - 1):
        weights = []
        prev = -1
        for c in combo:
            weights.append(c - prev - 1)
            prev = c
        weights.append(resolution + n_dims - 2 - prev)
        a, b, c = (w / resolution for w in weights)
        yield (a, b, 0.0, 0.0, c)


def weight_grid_2d_ic(resolution: int = 20):
    for k in range(resolution + 1):
        w = k / resolution
        yield (w, 1.0 - w, 0.0, 0.0, 0.0)


def all_weight_tuples() -> list[tuple[float, ...]]:
    seen: set[tuple[float, ...]] = set()
    out: list[tuple[float, ...]] = []

    def add(w: tuple[float, ...]) -> None:
        key = tuple(round(x, 8) for x in w)
        if key not in seen:
            seen.add(key)
            out.append(w)

    for w in weight_grid_5d(20):
        if min(w) >= 0.02:
            add(w)
    for w in weight_grid_3d(20):
        add(w)
    for w in weight_grid_2d_ic(20):
        add(w)
    for i in range(5):
        z = [0.0] * 5
        z[i] = 1.0
        add(tuple(z))
    add((0.2, 0.2, 0.2, 0.2, 0.2))
    add(CURRENT_WEIGHTS)
    add((1 / 3, 1 / 3, 0.0, 0.0, 1 / 3))
    return out


def weight_id(w: tuple[float, ...]) -> str:
    s = ",".join(f"{x:.6f}" for x in w)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def spearman_cols(scores: np.ndarray, h: np.ndarray) -> np.ndarray:
    d = pd.DataFrame(np.asarray(scores, dtype=float))
    d["__h"] = np.asarray(h, dtype=float)
    ser = d.drop(columns="__h").corrwith(d["__h"], method="spearman")
    return ser.reindex(range(scores.shape[1])).to_numpy(dtype=float)


def kendall_cols(scores: np.ndarray, h: np.ndarray) -> np.ndarray:
    from scipy import stats as scipy_stats

    h = np.asarray(h, dtype=float)
    n, k = scores.shape
    out = np.full(k, np.nan, dtype=float)
    for j in range(k):
        s = scores[:, j]
        m = np.isfinite(s) & np.isfinite(h)
        if m.sum() < 25:
            continue
        t, _ = scipy_stats.kendalltau(s[m], h[m])
        out[j] = float(t)
    return out


def _find_weight_index(weights: list[tuple[float, ...]], w: tuple[float, ...]) -> int | None:
    for j, ww in enumerate(weights):
        if np.allclose(ww, w, atol=1e-5):
            return j
    return None


def main() -> None:
    df = pd.read_csv(DATA / "merged.csv")
    work = df.dropna(subset=["human_score"]).copy()
    for d in SUB_DIMS:
        if d not in work.columns:
            raise SystemExit(f"Missing column {d} in merged.csv")
    X = work[SUB_DIMS].to_numpy(dtype=float)
    h = work["human_score"].to_numpy(dtype=float)
    lps = work["lp"].astype(str).to_numpy()
    uniq_lp = sorted(work["lp"].unique())
    print("Language pairs:", len(uniq_lp), "rows:", len(work))

    weights = all_weight_tuples()
    W = np.asarray(weights, dtype=float)
    K = W.shape[0]
    meta = [{"weight_id": weight_id(tuple(w)), "weights": list(w)} for w in weights]
    (DATA / "weight_optimization_weights_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_folds = len(uniq_lp)
    train_spear = np.full((n_folds, K), np.nan, dtype=float)
    test_spear = np.full((n_folds, K), np.nan, dtype=float)
    test_kend = np.full((n_folds, K), np.nan, dtype=float)

    for fi, held in enumerate(uniq_lp):
        tr = lps != held
        te = lps == held
        X_tr, h_tr = X[tr], h[tr]
        X_te, h_te = X[te], h[te]
        S_tr = X_tr @ W.T
        S_te = X_te @ W.T
        train_spear[fi, :] = spearman_cols(S_tr, h_tr)
        test_spear[fi, :] = spearman_cols(S_te, h_te)
        test_kend[fi, :] = kendall_cols(S_te, h_te)
        print(f"Fold {fi + 1}/{n_folds} ({held})")

    mean_tr = np.nanmean(train_spear, axis=0)
    mean_te = np.nanmean(test_spear, axis=0)
    std_te = np.nanstd(test_spear, axis=0)
    std_tr = np.nanstd(train_spear, axis=0)

    order = np.argsort(-mean_te)
    feasible = np.array([min(weights[j]) >= 0.02 - 1e-9 for j in range(K)], dtype=bool)
    masked_te = np.where(feasible, mean_te, -np.inf)
    j_best_feasible = int(np.argmax(masked_te))
    top_k = 50
    top_idx = list(order[:top_k])
    named_spec = [
        ("current", CURRENT_WEIGHTS),
        ("equal5", (0.2,) * 5),
        ("equal3_icg", (1 / 3, 1 / 3, 0.0, 0.0, 1 / 3)),
    ]
    for _label, wv in named_spec:
        j = _find_weight_index(weights, wv)
        if j is not None and j not in top_idx:
            top_idx.append(j)

    pw_fold = np.full((n_folds, K), np.nan, dtype=float)
    for j in top_idx:
        wkey = meta[j]["weight_id"]
        for fi, held in enumerate(uniq_lp):
            sub = work.loc[work["lp"] == held]
            sc = (sub[SUB_DIMS].to_numpy(dtype=float) @ np.asarray(weights[j], dtype=float)).reshape(-1)
            sub2 = sub.copy()
            sub2["weighted_score"] = sc
            pw_fold[fi, j] = pairwise_for_weight_on_subset(sub2, sc, "weighted_score")
        print(f"Pairwise CV for weight {wkey} ({weights[j]})")

    pw_mean = np.nanmean(pw_fold, axis=0)
    pw_std = np.nanstd(pw_fold, axis=0)

    rows = []
    for j in range(K):
        w = weights[j]
        rows.append(
            {
                "weight_id": meta[j]["weight_id"],
                "w_idiomatic": w[0],
                "w_collocational": w[1],
                "w_discourse": w[2],
                "w_pragmatic": w[3],
                "w_calque": w[4],
                "mean_test_spearman": mean_te[j],
                "std_test_spearman": std_te[j],
                "mean_test_pairwise": pw_mean[j],
                "std_test_pairwise": pw_std[j],
                "mean_train_spearman": mean_tr[j],
                "std_train_spearman": std_tr[j],
                "mean_test_kendall": float(np.nanmean(test_kend[:, j])),
            }
        )
    cv_df = pd.DataFrame(rows).sort_values("mean_test_spearman", ascending=False, na_position="last")
    named_lookup = {}
    for _label, wv in named_spec:
        jj = _find_weight_index(weights, wv)
        if jj is not None:
            named_lookup[_label] = jj
    named_ids = {meta[jj]["weight_id"] for jj in named_lookup.values()}
    top50 = cv_df.head(top_k)
    extra = cv_df[cv_df["weight_id"].isin(named_ids)]
    out_cv = pd.concat([top50, extra], ignore_index=True).drop_duplicates(subset=["weight_id"])
    out_cv = out_cv.sort_values("mean_test_spearman", ascending=False, na_position="last")
    out_cv.to_csv(DATA / "weight_optimization_cv.csv", index=False)
    print("Top rows:")
    print(out_cv.head(12).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {DATA / 'weight_optimization_cv.csv'}")
    print(f"Best min-weight≥0.02 idx={j_best_feasible} weights={weights[j_best_feasible]}")


if __name__ == "__main__":
    main()
