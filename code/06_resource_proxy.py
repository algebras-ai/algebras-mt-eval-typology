#!/usr/bin/env python3
"""Partial Spearman: pa_advantage vs syntax distance, controlling for Wikipedia size proxy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def partial_spearman_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Rank-based partial correlation via Spearman = Pearson on ranks, residualized on control covariate."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan"), float("nan")
    rx = stats.rankdata(x[m])
    ry = stats.rankdata(y[m])
    rz = stats.rankdata(z[m])
    Xc = np.column_stack([np.ones(len(rz)), rz])
    bx, _, _, _ = np.linalg.lstsq(Xc, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(Xc, ry, rcond=None)
    ex = rx - Xc @ bx
    ey = ry - Xc @ by
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def main() -> None:
    rp = pd.read_csv(DATA / "resource_proxy.csv")
    x = rp["distance_lang2vec_syntax_average"].values
    y = rp["pa_advantage"].values
    z = rp["log_wiki"].values
    rho, p = partial_spearman_pearson(x, y, z)
    raw_r, raw_p = stats.spearmanr(x, y)
    print("Resource proxy (subset with Wikipedia counts):")
    print(rp.to_string(index=False))
    print()
    print(f"Spearman(pa_advantage, syntax_distance): rho={raw_r:.4f}, p={raw_p:.4f}")
    print(f"Partial Spearman (controlling log_wiki): rho={rho:.4f}, p={p:.4f}")


if __name__ == "__main__":
    main()
