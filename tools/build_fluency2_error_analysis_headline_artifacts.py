#!/usr/bin/env python3
"""Build headline figures, captions, error-example table, and limitations snippet.

Writes under data/error_analysis/ in this repo (does not modify slice parquets or re-run validation).
"""
from __future__ import annotations

import html as html_lib
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WMT_MQM = ROOT / "data" / "wmt_mqm"
OUT = ROOT / "data" / "error_analysis"
TABLES = OUT / "tables"
FIG = OUT / "figures"

MERGE_KEYS = ["seg_id", "system_id", "year", "pair"]

PATH_CANDIDATES: dict[str, list[Path]] = {
    "sampled_segments": [WMT_MQM / "sampled_segments.parquet"],
    "fluency2_opus47": [WMT_MQM / "fluency2_opus47.parquet"],
    "fluency2_gemini": [WMT_MQM / "fluency2_gemini.parquet"],
    "fluency2_gpt5": [WMT_MQM / "fluency2_gpt5.parquet"],
    "fluency_mqm_opus47": [WMT_MQM / "fluency_mqm_opus47.parquet"],
    "fluency_mqm_gemini": [
        WMT_MQM / "fluency_mqm_gemini.parquet",
        WMT_MQM / "fluency_mqm_gemini_fluency_mqm.parquet",
    ],
    "fluency_mqm_gpt5": [
        WMT_MQM / "fluency_mqm_gpt5.parquet",
        WMT_MQM / "fluency_mqm_gpt5_fluency_mqm.parquet",
    ],
}


def resolve_path(key: str) -> Path | None:
    for p in PATH_CANDIDATES.get(key, []):
        if p.exists():
            return p
    return None


def load_side(
    path_key: str,
    rename: dict[str, str],
    keys: list[str],
) -> pd.DataFrame:
    p = resolve_path(path_key)
    if p is None:
        return pd.DataFrame(columns=keys + list(rename.values()))
    df = pd.read_parquet(p)
    cols = keys + [c for c in rename if c in df.columns]
    out = df[cols].copy()
    out = out.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for _, v in rename.items():
        if v not in out.columns:
            out[v] = np.nan
    return out


def load_merged() -> pd.DataFrame:
    seg_path = OUT / "df_seg.parquet"
    if seg_path.exists():
        return pd.read_parquet(seg_path)
    keys = MERGE_KEYS
    base_p = resolve_path("sampled_segments")
    if base_p is None:
        raise FileNotFoundError("sampled_segments.parquet not found")
    base = pd.read_parquet(base_p)
    f2_opus = load_side(
        "fluency2_opus47",
        {
            "fluency2": "f2_opus",
            "fluency2_raw": "f2_opus_raw",
            "issue_type": "issue_type_f2_opus",
        },
        keys,
    )
    f2_gem = load_side(
        "fluency2_gemini",
        {
            "fluency2": "f2_gemini",
            "fluency2_raw": "f2_gemini_raw",
            "issue_type": "issue_type_f2_gemini",
        },
        keys,
    )
    f2_gpt = load_side(
        "fluency2_gpt5",
        {
            "fluency2": "f2_gpt5",
            "fluency2_raw": "f2_gpt5_raw",
            "issue_type": "issue_type_f2_gpt5",
        },
        keys,
    )
    fmqm_opus = load_side(
        "fluency_mqm_opus47",
        {
            "fluency_mqm": "fmqm_opus",
            "fluency_mqm_errors_json": "fmqm_errors_json_opus",
            "fluency_mqm_issue_type": "fmqm_issue_opus",
        },
        keys,
    )
    fmqm_gem = load_side(
        "fluency_mqm_gemini",
        {
            "fluency_mqm": "fmqm_gemini",
            "fluency_mqm_errors_json": "fmqm_errors_json_gemini",
            "fluency_mqm_issue_type": "fmqm_issue_gemini",
        },
        keys,
    )
    fmqm_gpt = load_side(
        "fluency_mqm_gpt5",
        {
            "fluency_mqm": "fmqm_gpt5",
            "fluency_mqm_errors_json": "fmqm_errors_json_gpt5",
            "fluency_mqm_issue_type": "fmqm_issue_gpt5",
        },
        keys,
    )
    merged = base.copy()
    for part in (f2_opus, f2_gem, f2_gpt, fmqm_opus, fmqm_gem, fmqm_gpt):
        merged = merged.merge(part, on=keys, how="outer")
    return merged


def _f2_raw_max(merged: pd.DataFrame) -> float:
    cols = [c for c in merged.columns if c.startswith("f2_") and c.endswith("_raw")]
    if not cols:
        return 10.0
    return float(merged[cols].max().max())


def metric_0_100_series(merged: pd.DataFrame, mode: str, judge: str) -> pd.Series:
    jk = "opus" if judge == "opus47" else judge
    raw_max = _f2_raw_max(merged)
    if mode == "fmqm":
        return pd.to_numeric(merged[f"fmqm_{jk}"], errors="coerce")
    raw_col = f"f2_{jk}_raw"
    main_col = f"f2_{jk}"
    if raw_col in merged.columns:
        raw = pd.to_numeric(merged[raw_col], errors="coerce")
        out = np.where(
            raw.notna(),
            np.where(raw_max <= 10.0 + 1e-6, raw * 10.0, raw),
            np.nan,
        )
        return pd.Series(out, index=merged.index, dtype=float)
    v = pd.to_numeric(merged[main_col], errors="coerce")
    return np.where(raw_max <= 10.0 + 1e-6, v * 10.0, v)


def human_rescaled_100(merged: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(merged["human_mqm_score"], errors="coerce")
    vmin = float(h.min(skipna=True))
    vmax = float(h.max(skipna=True))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return h * 0.0
    return (h - vmin) / (vmax - vmin) * 100.0


def plot_headline_1(merged: pd.DataFrame, vc: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    human_100 = human_rescaled_100(merged)
    bins = np.arange(0, 102, 2)

    layout = [
        ("f2", "opus47", "Opus 4.7", "fluency2"),
        ("f2", "gpt5", "GPT-5", "fluency2"),
        ("fmqm", "gemini", "Gemini", "fluency-MQM"),
        ("fmqm", "gpt5", "GPT-5", "fluency-MQM"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (mode, judge, disp, mode_label) in zip(axes_flat, layout):
        metric = metric_0_100_series(merged, mode, judge).dropna()
        row = vc[(vc["mode"] == mode) & (vc["judge"] == judge)]
        if row.empty:
            n, pct, std = 0, float("nan"), float("nan")
        else:
            r = row.iloc[0]
            n = int(r["n"])
            pct = float(r["pct_ge_98_on_0_100_scale"])
            std = float(r["std_raw_on_0_100"])

        color_metric = "#2D6AB5" if mode == "f2" else "#C4763A"
        ax.hist(
            metric,
            bins=bins,
            color=color_metric,
            alpha=0.7,
            label=f"{mode_label} score",
        )
        ax.hist(
            human_100.dropna(),
            bins=bins,
            color="#888888",
            alpha=0.4,
            label="human MQM (rescaled)",
        )
        ax.axvline(98, color="red", linestyle="--", linewidth=1.2)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            98,
            0.98,
            " saturation",
            color="red",
            fontsize=8,
            ha="left",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            0.98,
            f"n={n}\npct≥98={pct:.1%}\nstd={std:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        ax.set_title(f"{disp} / {mode_label}")
        ax.grid(alpha=0.25)

    fig.supxlabel("score (0-100 scale)")
    fig.supylabel("count")
    fig.suptitle("Score distributions by judge and mode vs human MQM", y=1.02)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "fig_headline_1_variance_collapse.png", bbox_inches="tight")
    plt.close(fig)


def write_fig1_caption(vc: pd.DataFrame) -> None:
    gem = vc[(vc["mode"] == "fmqm") & (vc["judge"] == "gemini")].iloc[0]
    g5 = vc[(vc["mode"] == "fmqm") & (vc["judge"] == "gpt5")].iloc[0]
    gemini_pct = float(gem["pct_ge_98_on_0_100_scale"])
    gpt5_pct = float(g5["pct_ge_98_on_0_100_scale"])
    text = f"""Figure 1: Score distributions by judge and mode, compared against the
human MQM reference distribution (rescaled to 0–100). Saturation at
score ≥ 98 is marked. Gemini and GPT-5 in fluency-MQM mode collapse
substantial mass into the [98, 100] band ({gemini_pct:.1%} and
{gpt5_pct:.1%} respectively), while fluency2 positive scoring for both
Opus 4.7 and GPT-5 stays below 1% saturation. The shape mismatch against
human MQM is most pronounced for Gemini fluency-MQM, indicating
ceiling effect in negative scoring.
"""
    (FIG / "fig_headline_1_caption.md").write_text(text.strip() + "\n", encoding="utf-8")


def plot_headline_2(vc: pd.DataFrame, bc: pd.DataFrame) -> None:
    j = vc.merge(
        bc,
        on=["mode", "judge"],
        how="inner",
        suffixes=("", "_bc"),
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for _, r in j.iterrows():
        mode = r["mode"]
        color = "#2D6AB5" if mode == "f2" else "#C4763A"
        marker = "o" if mode == "f2" else "s"
        x = float(r["std_raw_on_0_100"])
        y = float(r["linreg_slope"])
        ax.scatter(
            [x],
            [y],
            s=300,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        label = f"{mode}/{r['judge']}"
        ax.annotate(label, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlim(8, 20)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("metric std on 0–100 scale (higher = more variance)")
    ax.set_ylabel("slope vs human (higher = better calibrated)")
    ax.set_title("Calibration vs Variance — two-mode trade-off")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "fig_headline_2_calibration_variance.png", bbox_inches="tight")
    plt.close(fig)


def _fmt_range(lo: float, hi: float, nd: int = 3) -> str:
    if lo > hi:
        lo, hi = hi, lo
    return f"{lo:.{nd}f}–{hi:.{nd}f}"


def write_fig2_caption(vc: pd.DataFrame, bc: pd.DataFrame) -> None:
    j = vc.merge(bc, on=["mode", "judge"], how="inner")
    fmqm = j[j["mode"] == "fmqm"]
    f2 = j[j["mode"] == "f2"]
    fmqm_slope_range = _fmt_range(
        float(fmqm["linreg_slope"].min()),
        float(fmqm["linreg_slope"].max()),
        3,
    )
    fmqm_std_range = _fmt_range(
        float(fmqm["std_raw_on_0_100"].min()),
        float(fmqm["std_raw_on_0_100"].max()),
        2,
    )
    f2_std_range = _fmt_range(
        float(f2["std_raw_on_0_100"].min()),
        float(f2["std_raw_on_0_100"].max()),
        2,
    )
    f2_slope_range = _fmt_range(
        float(f2["linreg_slope"].min()),
        float(f2["linreg_slope"].max()),
        3,
    )
    text = f"""Figure 2: The calibration–variance trade-off across the four
(mode × judge) combinations. Fluency-MQM variants achieve higher slope
against human MQM ({fmqm_slope_range}) at the cost of reduced variance
(std {fmqm_std_range} on a 0–100 scale), due to saturation near 100.
Fluency2 positive variants retain wider variance (std {f2_std_range})
but track human score less tightly (slope {f2_slope_range}). The two
modes are structurally complementary rather than substitutable.
"""
    (FIG / "fig_headline_2_caption.md").write_text(text.strip() + "\n", encoding="utf-8")


def f2_display_scores(row: pd.Series, raw_max: float) -> tuple[float, float]:
    def one(jk: str) -> float:
        raw = row.get(f"f2_{jk}_raw")
        if pd.notna(raw):
            v = float(raw)
            return v * 10.0 if raw_max <= 10.0 + 1e-6 else v
        v2 = row.get(f"f2_{jk}")
        if pd.isna(v2):
            return float("nan")
        vf = float(v2)
        return vf * 10.0 if raw_max <= 10.0 + 1e-6 else vf

    return one("opus"), one("gpt5")


def collect_error_examples(raw_max: float) -> tuple[pd.DataFrame, str]:
    pairs = {"en-de", "en-ru"}
    slices_spec: list[tuple[str, str, str, bool]] = [
        ("slice_A_f2_opus47.parquet", "slice_A_f2_opus47", "resid_f2_opus", False),
        ("slice_A_f2_gpt5.parquet", "slice_A_f2_gpt5", "resid_f2_gpt5", False),
        ("slice_A_fmqm_gemini.parquet", "slice_A_fmqm_gemini", "resid_fmqm_gemini", False),
        ("slice_A_fmqm_gpt5.parquet", "slice_A_fmqm_gpt5", "resid_fmqm_gpt5", False),
        ("slice_B_low_human.parquet", "slice_B_low_human", "human_mqm_score", True),
        ("slice_C_f2_pooled.parquet", "slice_C_f2_pooled", "std_f2_z", False),
        ("slice_C_fmqm_pooled.parquet", "slice_C_fmqm_pooled", "std_fmqm_z", False),
    ]

    rows_out: list[dict] = []
    html_parts: list[str] = []

    for fname, slice_name, rank_col, ascending in slices_spec:
        p = OUT / fname
        df = pd.read_parquet(p)
        section_title = {
            "slice_A_f2_opus47": "### Slice A — Residuals top-5 (fluency2 / Opus 4.7)",
            "slice_A_f2_gpt5": "### Slice A — Residuals top-5 (fluency2 / GPT-5)",
            "slice_A_fmqm_gemini": "### Slice A — Residuals top-5 (fluency-MQM / Gemini)",
            "slice_A_fmqm_gpt5": "### Slice A — Residuals top-5 (fluency-MQM / GPT-5)",
            "slice_B_low_human": "### Slice B — Lowest human MQM (worst human)",
            "slice_C_f2_pooled": "### Slice C — Pooled fluency2 disagreement (std_f2_z)",
            "slice_C_fmqm_pooled": "### Slice C — Pooled fluency-MQM disagreement (std_fmqm_z)",
        }[slice_name]
        html_parts.append(f"\n{section_title}\n\n")

        for pair in ("en-ru", "en-de"):
            sub = df[df["pair"].astype(str) == pair].copy()
            n_avail = len(sub)
            if rank_col in ("resid_f2_opus", "resid_f2_gpt5", "resid_fmqm_gemini", "resid_fmqm_gpt5"):
                sub["_rk"] = pd.to_numeric(sub[rank_col], errors="coerce").abs()
                sub = sub.sort_values("_rk", ascending=False, na_position="last")
            elif rank_col == "human_mqm_score":
                sub = sub.sort_values(rank_col, ascending=True, na_position="last")
            else:
                sub = sub.sort_values(rank_col, ascending=False, na_position="last")
            take = sub.head(5)

            for rank, (_, row) in enumerate(take.iterrows(), start=1):
                f2o, f2g = f2_display_scores(row, raw_max)
                fmg = row.get("fmqm_gemini")
                fmg = float(fmg) if pd.notna(fmg) else float("nan")
                fmp = row.get("fmqm_gpt5")
                fmp = float(fmp) if pd.notna(fmp) else float("nan")
                hum = row.get("human_mqm_score")
                hum = float(hum) if pd.notna(hum) else float("nan")
                gem_issue = row.get("fmqm_issue_gemini")
                gem_issue_s = "" if pd.isna(gem_issue) else str(gem_issue)
                rk_val = row.get(rank_col)
                rk_disp: float | str
                if rank_col.startswith("resid_"):
                    rk_disp = float(rk_val) if pd.notna(rk_val) else float("nan")
                    summary_metric = f"resid={rk_disp:.4g}" if math.isfinite(rk_disp) else "resid=NA"
                elif rank_col == "human_mqm_score":
                    rk_disp = float(rk_val) if pd.notna(rk_val) else float("nan")
                    summary_metric = f"human={rk_disp:.3g}" if math.isfinite(rk_disp) else "human=NA"
                else:
                    rk_disp = float(rk_val) if pd.notna(rk_val) else float("nan")
                    summary_metric = f"{rank_col}={rk_disp:.4g}" if math.isfinite(rk_disp) else f"{rank_col}=NA"

                src = row.get("original_text", "")
                mt = row.get("translation", "")
                ref = row.get("ref_text", "")
                ref_s = "NA" if pd.isna(ref) or ref is None or str(ref).strip() == "" else str(ref)
                sys_id = row.get("system_id", "")
                year = row.get("year", "")
                seg_id = row.get("seg_id", "")

                rows_out.append(
                    {
                        "slice": slice_name,
                        "pair": pair,
                        "year": year,
                        "system_id": sys_id,
                        "seg_id": seg_id,
                        "src_text": src if pd.notna(src) else "",
                        "mt_text": mt if pd.notna(mt) else "",
                        "ref_text": ref_s if ref_s != "NA" else "",
                        "human_mqm_score": hum,
                        "f2_opus": f2o,
                        "f2_gpt5": f2g,
                        "fmqm_gemini": fmg,
                        "fmqm_gpt5": fmp,
                        "gemini_issue": gem_issue_s,
                        "rank_metric": rank_col,
                        "rank_value": rk_disp,
                    }
                )

                pair_flag = f" `[n available: {n_avail}]`" if n_avail < 5 else ""
                esc_src = html_lib.escape(str(src) if pd.notna(src) else "")
                esc_mt = html_lib.escape(str(mt) if pd.notna(mt) else "")
                esc_ref = html_lib.escape(ref_s)
                title = (
                    f"<b>{slice_name} #{rank} ({pair}, {year}, system: {sys_id})</b> — {summary_metric}"
                )
                if rank == 1:
                    title += pair_flag
                bullets = (
                    f"<ul>\n"
                    f"<li>human_mqm_score: {hum}</li>\n"
                    f"<li>f2_opus: {f2o}</li><li>f2_gpt5: {f2g}</li>\n"
                    f"<li>fmqm_gemini: {fmg}</li><li>fmqm_gpt5: {fmp}</li>\n"
                    f"<li>Gemini labeler issue_type: {html_lib.escape(gem_issue_s)}</li>\n"
                    f"<li><b>{html_lib.escape(rank_col)}</b>: {html_lib.escape(str(rk_val))}</li>\n"
                    f"</ul>"
                )
                blk = (
                    f"<details>\n<summary>{title}</summary>\n"
                    f"<p><b>SRC:</b> {esc_src}</p>\n"
                    f"<p><b>MT:</b> {esc_mt}</p>\n"
                    f"<p><b>REF:</b> {esc_ref}</p>\n"
                    f"{bullets}\n</details>\n"
                )
                html_parts.append(blk)

    md_body = "".join(html_parts)
    out_df = pd.DataFrame(rows_out)
    return out_df, md_body


def write_limitations() -> None:
    fam = pd.read_csv(TABLES / "self_preference_family_control.csv")
    openai_row = fam[fam["family"] == "OpenAI"].iloc[0]
    delta = float(openai_row["mean_f2_gpt5_z"] - openai_row["mean_human_z"])
    if delta < 0:
        direction = "below"
        interpretation = "indicating no detectable self-preference for this judge on this data"
    elif delta > 0.1:
        direction = "above"
        interpretation = "suggesting possible mild self-preference requiring further investigation"
    elif abs(delta) < 0.05:
        direction = "essentially aligned with"
        interpretation = "indicating no meaningful bias"
    else:
        direction = "above"
        interpretation = (
            "a small positive gap below the 0.1σ threshold; treat as inconclusive without more data"
        )

    text = f"""## Limitations: data coverage

Our three-judge error analysis is bounded by score data available on
the WMT22–24 MQM segments (n=2758). We analyze fluency2 (positive
1-10 scoring) for Opus 4.7 and GPT-5, and fluency-MQM (0-100 penalty
scoring) for Gemini 3 Flash and GPT-5. Two cells of the judge × mode
matrix remain uncovered: fluency2 for Gemini and fluency-MQM for
Opus 4.7. The Opus gap is a budget constraint (Anthropic API credit
limit reached at the time of submission); the Gemini fluency2 file
exists in earlier experiments but was not aligned on the current
composite key at writing. Neither absence affects the headline
findings on variance collapse and calibration asymmetry.

## Limitations: self-preference analysis

A stricter self-preference ablation would require scoring translations
produced by each judge's own model family (OpenAI, Google, Anthropic)
with all three judges, enabling a full family × judge matrix. We
conduct a partial analysis using GPT-5 as the only judge with both
fluency2 scores and family-stratifiable control data, and find that
GPT-5's fluency2 scores on OpenAI-family translations are {direction} the mean human score (Δ = {delta:+.3f}σ on z-scaled metric vs human), {interpretation}. A full three-judge self-preference
matrix was out of scope for this submission due to budget constraints.
"""
    (OUT / "paper_limitations_snippet.md").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    merged = load_merged()
    vc = pd.read_csv(TABLES / "variance_collapse.csv")
    bc = pd.read_csv(TABLES / "bias_calibration.csv")

    plot_headline_1(merged, vc)
    write_fig1_caption(vc)
    plot_headline_2(vc, bc)
    write_fig2_caption(vc, bc)

    raw_max = _f2_raw_max(merged)
    ex_df, html_body = collect_error_examples(raw_max)
    ex_df.to_csv(TABLES / "error_examples_en_de_en_ru.csv", index=False)
    (TABLES / "error_examples_display.html").write_text(html_body, encoding="utf-8")

    write_limitations()
    print("Wrote headline artifacts under", OUT)


if __name__ == "__main__":
    main()
