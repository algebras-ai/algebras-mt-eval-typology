#!/usr/bin/env python3
"""Regenerate all four paper figures into ../figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG = REPO / "figures"

METRICS_FIG2 = ["fluency2_raw", "chrf", "bleu", "comet"]
COL_RENAME = {
    "fluency2_raw": "LLM Judge",
    "chrf": "chrF",
    "bleu": "BLEU",
    "comet": "COMET",
}
COLOR = {
    "LLM Judge": "#2563eb",
    "chrF": "#dc2626",
    "BLEU": "#f59e0b",
    "COMET": "#16a34a",
}


def fig1_syntax_distance_vs_advantage(td: pd.DataFrame) -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10
    tdf = td.dropna(subset=["pa_advantage", "distance_lang2vec_syntax_average"])
    x = tdf["distance_lang2vec_syntax_average"]
    y = tdf["pa_advantage"]
    rho, p = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=60, zorder=5, color="#2563eb")
    xs = np.linspace(float(x.min()) - 0.02, float(x.max()) + 0.02, 100)
    ax.plot(xs, slope * xs + intercept, color="#dc2626", linewidth=1.5, linestyle="--", alpha=0.7)
    offsets = {
        "EN-it_IT": (-28, -10),
        "EN-zh_CN": (6, -12),
        "EN-mas_KE": (-42, 6),
        "EN-uk_UA": (6, -14),
    }
    for _, row in tdf.iterrows():
        lp = row["lp"]
        dx, dy = offsets.get(lp, (6, 4))
        ax.annotate(
            lp,
            (row["distance_lang2vec_syntax_average"], row["pa_advantage"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
        )
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Syntactic distance from English (lang2vec)", fontsize=10)
    ax.set_ylabel("Pairwise accuracy advantage\n(fluency judge − chrF)", fontsize=10)
    ax.set_title(f"ρ = {rho:.2f}, p = {p:.3f}, n = {len(tdf)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_syntax_distance_vs_advantage.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig1_syntax_distance_vs_advantage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig1")


def fig2_perlp(pa_lp: pd.DataFrame, td: pd.DataFrame) -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 9
    piv = pa_lp.pivot(index="lp", columns="metric", values="pairwise_accuracy")
    merged = td[["lp", "distance_lang2vec_syntax_average"]].merge(piv, on="lp", how="inner")
    merged = merged.sort_values("distance_lang2vec_syntax_average", na_position="last")
    lps = merged["lp"].tolist()
    merged_i = merged.set_index("lp")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(lps), dtype=float)
    active = [m for m in METRICS_FIG2 if m in merged_i.columns]
    n_met = len(active)
    bar_w = 0.16
    offsets = tuple((i - (n_met - 1) / 2) * bar_w for i in range(n_met))
    for i, m in enumerate(active):
        ys = []
        for lp in lps:
            v = merged_i.at[lp, m] if lp in merged_i.index and m in merged_i.columns else float("nan")
            ys.append(float(v) if np.isfinite(v) else float("nan"))
        label = COL_RENAME.get(m, m)
        ax.bar(
            x + offsets[i],
            ys,
            width=bar_w,
            color=COLOR[label],
            edgecolor="none",
            zorder=2,
        )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(lps, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Pairwise accuracy")
    ax.set_xlabel("Language pair (sorted by syntactic distance from English →)")
    ax.set_title("Per-language-pair pairwise accuracy by metric.")
    ax.set_ylim(0.28, 0.68)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_elements = [
        Patch(facecolor=COLOR[COL_RENAME[m]], edgecolor="none", label=COL_RENAME[m]) for m in active
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_perlp_pa_by_metric.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig2_perlp_pa_by_metric.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig2")


def fig3_loo(td: pd.DataFrame) -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 9
    sub = td.dropna(subset=["pa_advantage", "distance_lang2vec_syntax_average"])
    x_col = "distance_lang2vec_syntax_average"
    y_col = "pa_advantage"
    full_rho, full_p = stats.spearmanr(sub[x_col], sub[y_col])
    drops = []
    rhos = []
    ps = []
    for drop_lp in sub["lp"].values:
        part = sub[sub["lp"] != drop_lp]
        r, p = stats.spearmanr(part[x_col], part[y_col])
        drops.append(drop_lp)
        rhos.append(r)
        ps.append(p)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axhline(full_rho, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Full sample ρ = {full_rho:.2f}")
    pos = np.arange(len(drops))
    ax.bar(pos, rhos, color="#2563eb", edgecolor="none")
    ax.set_xticks(pos)
    ax.set_xticklabels(drops, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Spearman ρ (syntax vs advantage)")
    ax.set_xlabel("Dropped language pair")
    ax.set_title("Leave-one-out: correlation after dropping each LP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_loo_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig3_loo_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig3 (full ρ=%.3f, p=%.4f)" % (full_rho, full_p))


def fig4_quadrant(df: pd.DataFrame, pa_lp: pd.DataFrame, td: pd.DataFrame) -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 9
    rows = []
    for lp, g in df.groupby("lp"):
        for metric, col in [("fluency2_raw", "fluency2_raw"), ("chrf", "chrf")]:
            if col not in g.columns:
                continue
            x = pd.to_numeric(g[col], errors="coerce")
            y = pd.to_numeric(g["human_score"], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 25:
                continue
            r, _ = stats.spearmanr(x[m], y[m])
            rows.append({"lp": lp, "metric": metric, "spearman": r})
    strat = pd.DataFrame(rows)
    flu = strat[strat["metric"] == "fluency2_raw"][["lp", "spearman"]].rename(
        columns={"spearman": "spearman_fluency"}
    )
    flu_pa = pa_lp[pa_lp["metric"] == "fluency2_raw"][["lp", "pairwise_accuracy"]].rename(
        columns={"pairwise_accuracy": "pa_fluency"}
    )
    chrf_spearman = strat[strat["metric"] == "chrf"][["lp", "spearman"]].rename(
        columns={"spearman": "spearman_chrf"}
    )
    chrf_pa = pa_lp[pa_lp["metric"] == "chrf"][["lp", "pairwise_accuracy"]].rename(
        columns={"pairwise_accuracy": "pa_chrf"}
    )
    merged = (
        flu.merge(flu_pa, on="lp", how="outer")
        .merge(chrf_spearman, on="lp", how="outer")
        .merge(chrf_pa, on="lp", how="outer")
        .merge(td[["lp", "distance_lang2vec_syntax_average"]], on="lp", how="left")
    )
    plot_base = merged.dropna(
        subset=["spearman_fluency", "pa_fluency", "distance_lang2vec_syntax_average"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    left, right, bottom, top, wspace = 0.07, 0.98, 0.14, 0.92, 0.28
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)
    norm = plt.Normalize(
        plot_base["distance_lang2vec_syntax_average"].min(),
        plot_base["distance_lang2vec_syntax_average"].max(),
    )
    cmap = cm.coolwarm
    ax = axes[0]
    for _, row in plot_base.iterrows():
        sc = row.get("spearman_chrf")
        if not np.isfinite(sc):
            continue
        color = cmap(norm(row["distance_lang2vec_syntax_average"]))
        ax.scatter(
            sc,
            row["spearman_fluency"],
            c=[color],
            s=80,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.annotate(
            row["lp"],
            (float(sc), float(row["spearman_fluency"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
        )
    lims = [-0.35, 0.30]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("chrF Spearman ρ", fontsize=9)
    ax.set_ylabel("LLM Judge Spearman ρ", fontsize=9)
    ax.set_title("(a) Rank correlation with human scores", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    for _, row in plot_base.iterrows():
        pc = row.get("pa_chrf")
        if not np.isfinite(pc):
            continue
        color = cmap(norm(row["distance_lang2vec_syntax_average"]))
        ax.scatter(
            pc,
            row["pa_fluency"],
            c=[color],
            s=80,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.annotate(
            row["lp"],
            (float(pc), float(row["pa_fluency"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
        )
    lims = [0.35, 0.65]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("chrF pairwise accuracy", fontsize=9)
    ax.set_ylabel("LLM Judge pairwise accuracy", fontsize=9)
    ax.set_title("(b) Pairwise accuracy", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    smm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smm.set_array([])
    gap_above_bottom_axis = 0.048
    cbar_height = 0.055
    cbar_width = 0.028
    cax_right = 0.924
    cax_left = cax_right - cbar_width
    cax_bottom = bottom + gap_above_bottom_axis
    cax = fig.add_axes([cax_left, cax_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(smm, cax=cax)
    cbar.ax.tick_params(labelsize=5, length=2, width=0.4)
    cbar.set_label("Syntactic\ndistance\nfrom English", fontsize=5.5, labelpad=5)
    for ext, dpi in (("pdf", 300), ("png", 200)):
        p = FIG / f"fig4_spearman_vs_pa_quadrant.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print("Wrote", p)
    plt.close()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    td = pd.read_csv(DATA / "typological_distances.csv")
    pa_lp = pd.read_csv(DATA / "pairwise_accuracy_by_lp.csv")
    df = pd.read_csv(DATA / "merged.csv")
    fig1_syntax_distance_vs_advantage(td)
    fig2_perlp(pa_lp, td)
    fig3_loo(td)
    fig4_quadrant(df, pa_lp, td)


if __name__ == "__main__":
    main()
