#!/usr/bin/env python3
"""Emit LaTeX tables tab1–tab4 into preprint/tables/."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TAB = REPO / "preprint" / "tables"

PRIMARY = [
    "lang2vec_syntax_average",
    "lang2vec_fam",
    "family_depth",
    "word_order",
    "lexical_inv_chrf",
]


def escape_tex(s: str) -> str:
    s = str(s).replace("&", r"\&").replace("_", r"\_")
    return re.sub(r"(?<!\\)%", r"\%", s)


def fmt(v, digits: int = 3) -> str:
    if pd.isna(v) or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)

    corr = pd.read_csv(DATA / "correlations_all_variants.csv")
    pa = pd.read_csv(DATA / "pairwise_accuracy_all_variants.csv")
    merged = corr.merge(pa, on="metric", how="outer")
    order = [
        "comet",
        "fluency2_raw",
        "fluency3_",
        "fluency2",
        "idiomatic",
        "collocational",
        "calque",
        "chrf",
        "bleu",
    ]
    merged["sort_key"] = merged["metric"].map({m: i for i, m in enumerate(order)}).fillna(99)
    merged = merged.sort_values("sort_key").drop(columns=["sort_key"])

    n_comet = 0
    if "comet" in merged["metric"].values:
        v = merged.loc[merged["metric"] == "comet", "n"]
        if len(v) and pd.notna(v.values[0]):
            n_comet = int(v.values[0])

    lines = [
        r"\begin{table*}[t]" + "\n",
        r"\centering" + "\n",
        r"\small" + "\n",
        rf"\caption{{Segment-level correlation with human judgments and pairwise accuracy. "
        rf"COMET on full 4,720 rows; column $n$ is rows with human scores used for Spearman/Kendall (COMET: {n_comet}).}}"
        + "\n",
        r"\label{tab:global}" + "\n",
        r"\begin{tabular}{lrrrr}" + "\n",
        r"\toprule" + "\n",
        r"Metric & Spearman & Kendall & Pairwise acc. & $n$ \\" + "\n",
        r"\midrule" + "\n",
    ]
    for _, row in merged.iterrows():
        m = str(row["metric"]).replace("_", r"\_")
        n_raw = row.get("n")
        n_val = str(int(n_raw)) if pd.notna(n_raw) else "—"
        lines.append(
            f"{m} & {fmt(row.get('spearman'))} & {fmt(row.get('kendall'))} & "
            f"{fmt(row.get('pairwise_accuracy'))} & {n_val} \\\\\n"
        )
    lines.extend([r"\bottomrule" + "\n", r"\end{tabular}" + "\n", r"\end{table*}" + "\n"])
    (TAB / "tab1_global_results.tex").write_text("".join(lines), encoding="utf-8")

    pa_lp = pd.read_csv(DATA / "pairwise_accuracy_by_lp.csv")
    td = pd.read_csv(DATA / "typological_distances.csv")
    piv = pa_lp.pivot(index="lp", columns="metric", values="pairwise_accuracy").reset_index()
    piv = piv.merge(td[["lp", "distance_lang2vec_syntax_average", "pa_advantage"]], on="lp", how="left")
    piv = piv.sort_values("distance_lang2vec_syntax_average")

    lines2 = [
        r"\begin{table*}[t]" + "\n",
        r"\centering" + "\n",
        r"\small" + "\n",
        r"\caption{Pairwise accuracy by language pair, sorted by syntactic distance from English.}" + "\n",
        r"\label{tab:perlp}" + "\n",
        r"\begin{tabular}{lrrrrrr}" + "\n",
        r"\toprule" + "\n",
        r"LP & Syntax dist. & fluency2\_raw & chrF & BLEU & COMET & Advantage \\" + "\n",
        r"\midrule" + "\n",
    ]
    for _, row in piv.iterrows():
        lp = str(row["lp"]).replace("_", r"\_")
        lines2.append(
            f"{lp} & {fmt(row.get('distance_lang2vec_syntax_average'))} & "
            f"{fmt(row.get('fluency2_raw'))} & {fmt(row.get('chrf'))} & {fmt(row.get('bleu'))} & "
            f"{fmt(row.get('comet'))} & {fmt(row.get('pa_advantage'))} \\\\\n"
        )
    lines2.extend([r"\bottomrule" + "\n", r"\end{tabular}" + "\n", r"\end{table*}" + "\n"])
    (TAB / "tab2_per_lp_results.tex").write_text("".join(lines2), encoding="utf-8")

    boot = pd.read_csv(DATA / "bootstrap_ci_all_distances.csv")
    boot = boot[boot["distance_name"].isin(PRIMARY)].copy()
    boot = boot.sort_values("rho_observed", ascending=False, key=lambda s: s.abs())

    lines3 = [
        r"\begin{table*}[t]" + "\n",
        r"\centering" + "\n",
        r"\small" + "\n",
        r"\caption{Correlation between typological distance and fluency judge advantage over chrF. "
        r"Five primary measures selected by theoretical relevance and data sufficiency ($n \geq 10$); "
        r"five additional measures in supplementary. "
        r"Holm--Bonferroni correction applied across the five primary tests.}" + "\n",
        r"\label{tab:dist}" + "\n",
        r"\begin{tabular}{lrrrrr}" + "\n",
        r"\toprule" + "\n",
        r"Distance measure & $\rho$ & 95\% CI & Perm.\ $p$ & Holm $p$ & $n$ \\" + "\n",
        r"\midrule" + "\n",
    ]
    for _, row in boot.iterrows():
        name = str(row["distance_name"]).replace("_", r"\_")
        rho = fmt(row.get("rho_observed"))
        ci = f"[{fmt(row.get('ci_lower'))}, {fmt(row.get('ci_upper'))}]"
        pp = fmt(row.get("perm_p"), 4)
        hp = fmt(row.get("holm_p_primary"), 4)
        n = str(int(row.get("n", 0)))
        lines3.append(f"{name} & {rho} & {ci} & {pp} & {hp} & {n} \\\\\n")
    lines3.extend([r"\bottomrule" + "\n", r"\end{tabular}" + "\n", r"\end{table*}" + "\n"])
    (TAB / "tab3_distance_correlations.tex").write_text("".join(lines3), encoding="utf-8")

    wcv = pd.read_csv(DATA / "weight_optimization_cv.csv").sort_values(
        "mean_test_spearman", ascending=False
    ).head(12)
    cols = [c for c in [
        "w_idiomatic",
        "w_collocational",
        "w_discourse",
        "w_pragmatic",
        "w_calque",
        "mean_test_spearman",
        "mean_test_pairwise",
    ] if c in wcv.columns]
    lines4 = [
        r"\begin{table*}[t]" + "\n",
        r"\centering" + "\n",
        r"\small" + "\n",
        r"\caption{Weight optimization (LOO CV): top candidates by mean test Spearman.}" + "\n",
        r"\label{tab:weight}" + "\n",
        r"\begin{tabular}{lrrrrrr}" + "\n",
        r"\toprule" + "\n",
        r" w\_idiomatic & w\_collocational & w\_discourse & w\_pragmatic & w\_calque & mean\_test\_spearman & mean\_test\_pairwise \\" + "\n",
        r"\midrule" + "\n",
    ]
    for _, row in wcv.iterrows():
        cells = [fmt(row[c], 3) for c in cols]
        lines4.append(" " + " & ".join(cells) + r" \\" + "\n")
    lines4.extend([r"\bottomrule" + "\n", r"\end{tabular}" + "\n", r"\end{table*}" + "\n"])
    (TAB / "tab4_weight_ablation.tex").write_text("".join(lines4), encoding="utf-8")

    print("Wrote tab1_global_results.tex, tab2_per_lp_results.tex, tab3_distance_correlations.tex, tab4_weight_ablation.tex")


if __name__ == "__main__":
    main()
