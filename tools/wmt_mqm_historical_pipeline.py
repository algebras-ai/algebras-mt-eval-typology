#!/usr/bin/env python3
"""
WMT22–24 MQM slice: inspect MTME data, stratified segment sampling, chrF++,
stored metrics (GEMBA-MQM, MetricX, XCOMET, CometKiwi), optional Fluency2
(Gemini), correlations vs human MQM, and comparison to WMT25 ESA CSV.

Requires extracted MT Metrics Eval v2 tree (see MT_METRICS_EVAL_ROOT).

Note: PyPI package ``mt-metrics-eval`` is unrelated; install from GitHub:
  git clone https://github.com/google-research/mt-metrics-eval.git && pip install .

Official tarball may require authenticated access; set MT_METRICS_EVAL_ROOT
to an existing ``mt-metrics-eval-v2`` directory if download fails.
"""
from __future__ import annotations

import argparse
import csv
import glob
import shutil
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
ALGEBRAS_ML = ROOT / "algebras-ml"
OUT_DIR = ROOT / "data_interim" / "wmt_mqm"
DEFAULT_MTME = Path.home() / ".mt-metrics-eval" / "mt-metrics-eval-v2"
TGZ_URL = "https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz"
# Same as ``compute_fluency2_gemini`` (override via ``FLUENCY2_GEMINI_MODEL`` in .env).
DEFAULT_FLUENCY2_GEMINI_MODEL = "gemini-3-flash-preview"


def _parse_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        os.environ[key] = val


def bootstrap_dotenv() -> None:
    """Load API keys / model from the same .env paths as ``compute_fluency2_gemini``."""
    raw: list[Path] = []
    forced = os.environ.get("FLUENCY2_ENV_FILE", "").strip()
    if forced:
        raw.append(Path(forced).expanduser())
    raw.extend([Path.home() / ".env", ROOT / ".env", ALGEBRAS_ML / ".env"])
    seen: set[str] = set()
    paths: list[Path] = []
    for p in raw:
        try:
            rp = str(p.resolve(strict=False))
        except OSError:
            rp = str(p)
        if rp not in seen:
            seen.add(rp)
            paths.append(p)
    try:
        from dotenv import load_dotenv

        for p in paths:
            load_dotenv(p, override=True)
    except ImportError:
        pass
    for p in paths:
        _parse_env_file(p)

# (eval_testset, year label for outputs, pair, n_segments)
SAMPLING_PLAN: tuple[tuple[str, str, str, int], ...] = (
    ("wmt22", "wmt22", "en-de", 10),
    ("wmt23", "wmt23", "en-de", 10),
    ("wmt24", "wmt24", "en-de", 10),
    ("wmt22", "wmt22", "zh-en", 15),
    ("wmt23", "wmt23", "zh-en", 15),
    ("wmt22", "wmt22", "en-ru", 30),
    ("wmt23", "wmt23", "he-en", 30),
    ("wmt24", "wmt24", "en-es", 30),
    ("wmt24", "wmt24", "ja-zh", 30),
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wmt_fluency_meta_eval_variants import (  # noqa: E402
    COMET_METRIC_TIE_BAND,
    pairwise_stats,
)


def _mtme_root() -> Path:
    env = os.environ.get("MT_METRICS_EVAL_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cand = ROOT / "data_interim" / "mt-metrics-eval-v2"
    if cand.is_dir():
        return cand.resolve()
    return DEFAULT_MTME.resolve()


def _ensure_mtme_import():
    try:
        from mt_metrics_eval import data as mt_data  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Cannot import mt_metrics_eval (Google). Install from GitHub:\n"
            "  git clone https://github.com/google-research/mt-metrics-eval.git\n"
            "  cd mt-metrics-eval && pip install .\n"
            f"Import error: {e}"
        ) from e
    return mt_data


def download_mtme_data() -> None:
    """Best-effort download of mt-metrics-eval-v2.tgz into ~/.mt-metrics-eval."""
    mt_data = _ensure_mtme_import()
    dest_root = Path.home() / ".mt-metrics-eval"
    dest_root.mkdir(parents=True, exist_ok=True)
    tgz = dest_root / "mt-metrics-eval-v2.tgz"
    if DEFAULT_MTME.is_dir():
        print(f"MTME data already present at {DEFAULT_MTME}")
        return
    print(f"Attempting download -> {tgz} …")
    try:
        mt_data.Download()
        print("Download() finished.")
    except Exception as e:
        print(f"mt_metrics_eval.data.Download() failed: {e}")
        print("Trying curl …")
        try:
            subprocess.run(
                ["curl", "-fL", "-o", str(tgz), TGZ_URL],
                check=True,
                timeout=7200,
            )
        except Exception as e2:
            raise SystemExit(
                "Could not download MTME data (403/401 common without GCS auth).\n"
                "Obtain ``mt-metrics-eval-v2`` (extracted) and set:\n"
                f"  export MT_METRICS_EVAL_ROOT=/path/to/mt-metrics-eval-v2\n"
                f"Curl error: {e2}"
            ) from e2
        import tarfile

        with tarfile.open(tgz, "r:*") as tar:
            tar.extractall(dest_root)
        print(f"Extracted to {dest_root}")


def _list_seg_metric_basenames(metric_dir: Path) -> list[str]:
    names: list[str] = []
    for p in metric_dir.glob("*.seg.score"):
        stem = p.name[: -len(".seg.score")]
        names.append(stem)
    return sorted(set(names))


def _pick_metric(
    candidates: Iterable[str],
    *,
    includes: tuple[str, ...],
    excludes: tuple[str, ...] = (),
) -> str | None:
    cands = sorted(set(candidates))
    for inc in includes:
        pat = re.compile(inc, re.I)
        for k in cands:
            if not pat.search(k):
                continue
            if any(re.search(x, k, re.I) for x in excludes):
                continue
            return k
    return None


def _pair_langs(lp: str) -> tuple[str, str]:
    a, b = lp.split("-", 1)
    return a, b


def _spearman_kendall(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    x = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df["human_score"], errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, _ = spearmanr(x[m], y[m])
    t, _ = kendalltau(x[m], y[m])
    return float(r), float(t), n


def _pairwise_accuracy_col(df: pd.DataFrame, metric_col: str) -> float:
    band = COMET_METRIC_TIE_BAND if metric_col in ("xcomet", "cometKiwi") else None
    return float(pairwise_stats(df, metric_col, metric_tie_band=band)["pairwise_accuracy"])


def step_inspect_and_csv(mt_root: Path) -> None:
    from mt_metrics_eval import data as mt_data

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for testset, year, pair, _ in SAMPLING_PLAN:
        if (testset, pair) in seen:
            continue
        seen.add((testset, pair))
        evs = mt_data.EvalSet(testset, pair, read_stored_metric_scores=True, strict=False)
        mqm = evs.Scores("seg", "mqm")
        if mqm is None:
            raise SystemExit(f"No seg mqm for {testset} {pair}")
        systems_with_human = sorted(
            s for s, vec in mqm.items() if any(v is not None for v in vec)
        )
        n_seg = len(evs.src)
        flat_h = []
        for s in systems_with_human:
            for v in mqm[s]:
                if v is not None:
                    flat_h.append(float(v))
        mean_h = float(np.mean(flat_h)) if flat_h else float("nan")
        std_h = float(np.std(flat_h)) if flat_h else float("nan")
        mdir = mt_root / testset / "metric-scores" / pair
        metric_basenames = _list_seg_metric_basenames(mdir) if mdir.is_dir() else []
        rows.append(
            {
                "year": year,
                "pair": pair,
                "testset": testset,
                "n_systems_mqm": len(systems_with_human),
                "systems_mqm": " ".join(systems_with_human),
                "n_segments": n_seg,
                "human_mqm_mean": mean_h,
                "human_mqm_std": std_h,
                "metric_seg_files": " ".join(metric_basenames),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "available_systems.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


def _segment_mean_mqm(mqm: dict[str, list[float | None]], n_seg: int) -> np.ndarray:
    acc = np.zeros(n_seg, dtype=float)
    cnt = np.zeros(n_seg, dtype=int)
    for _sys, vec in mqm.items():
        for i, v in enumerate(vec):
            if i >= n_seg:
                break
            if v is not None and np.isfinite(v):
                acc[i] += float(v)
                cnt[i] += 1
    mean = np.divide(acc, np.maximum(cnt, 1), out=np.zeros_like(acc), where=cnt > 0)
    mean[cnt == 0] = np.nan
    return mean


def _stratified_indices(mean_mqm: np.ndarray, n_target: int, rng: random.Random) -> list[int]:
    n = len(mean_mqm)
    valid = np.where(np.isfinite(mean_mqm))[0]
    if len(valid) == 0:
        return []
    if n_target >= len(valid):
        return sorted(int(x) for x in valid)
    s = pd.Series(mean_mqm[valid], index=valid)
    try:
        q = pd.qcut(s, q=min(5, len(s)), labels=False, duplicates="drop")
    except ValueError:
        q = pd.Series(0, index=valid)
    by_q: dict[int, list[int]] = {}
    for idx, lab in zip(q.index, q.values):
        by_q.setdefault(int(lab), []).append(int(idx))
    per = max(1, n_target // max(1, len(by_q)))
    chosen: list[int] = []
    for lab in sorted(by_q.keys()):
        pool = by_q[lab]
        rng.shuffle(pool)
        take = min(per, len(pool), n_target - len(chosen))
        chosen.extend(pool[:take])
    rest = [i for i in valid.tolist() if i not in chosen]
    rng.shuffle(rest)
    while len(chosen) < n_target and rest:
        chosen.append(rest.pop())
    return sorted(chosen[:n_target])


def build_sampled_parquet(mt_root: Path, seed: int = 42) -> pd.DataFrame:
    from mt_metrics_eval import data as mt_data

    rng = random.Random(seed)
    frames: list[pd.DataFrame] = []
    for testset, year, pair, n_target in SAMPLING_PLAN:
        evs = mt_data.EvalSet(testset, pair, read_stored_metric_scores=True, strict=False)
        mqm = evs.Scores("seg", "mqm")
        assert mqm is not None
        n_seg = len(evs.src)
        mean_mqm = _segment_mean_mqm(mqm, n_seg)
        idxs = _stratified_indices(mean_mqm, n_target, rng)
        docs_per_seg = evs.DocsPerSeg()
        std_ref = evs.std_ref
        ref_texts = evs.all_refs[std_ref]
        src_lang, tgt_lang = _pair_langs(pair)
        for seg_pos in idxs:
            src_text = evs.src[seg_pos] or ""
            ref_text = ref_texts[seg_pos] or ""
            doc_id = str(docs_per_seg[seg_pos]) if seg_pos < len(docs_per_seg) else ""
            for sys_id, hyps in evs.sys_outputs.items():
                if sys_id in evs.human_sys_names:
                    continue
                if seg_pos >= len(hyps):
                    continue
                sc = mqm.get(sys_id)
                if sc is None or seg_pos >= len(sc) or sc[seg_pos] is None:
                    continue
                mt_text = hyps[seg_pos]
                if mt_text is None:
                    continue
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "seg_id": int(seg_pos),
                                "year": year,
                                "testset": testset,
                                "pair": pair,
                                "system_id": sys_id,
                                "src_text": src_text,
                                "mt_text": str(mt_text),
                                "ref_text": ref_text,
                                "human_mqm_score": float(sc[seg_pos]),
                                "doc_id": doc_id,
                                "dataset": f"{year}_mqm",
                                "lp": pair,
                            }
                        ]
                    )
                )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df


def step_chrf_pp(df: pd.DataFrame) -> pd.DataFrame:
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    scores: list[float] = []
    for _, r in df.iterrows():
        ref = str(r.get("ref_text") or "").strip()
        if not ref:
            scores.append(float("nan"))
            continue
        scores.append(float(chrf.sentence_score(str(r.get("mt_text") or ""), [ref]).score))
    out = df.copy()
    out["chrf_pp"] = scores
    return out


def _metric_full_names_for_lp(mt_root: Path, testset: str, pair: str) -> list[str]:
    d = mt_root / testset / "metric-scores" / pair
    if not d.is_dir():
        return []
    return _list_seg_metric_basenames(d)


def attach_stored_metrics(df: pd.DataFrame, mt_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mt_metrics_eval import data as mt_data

    ev_cache: dict[tuple[str, str], Any] = {}

    def _ev(testset: str, pair: str):
        k = (testset, pair)
        if k not in ev_cache:
            ev_cache[k] = mt_data.EvalSet(
                testset, pair, read_stored_metric_scores=True, strict=False
            )
        return ev_cache[k]

    gemba_col: list[float | None] = []
    metricx_col: list[float | None] = []
    xcomet_col: list[float | None] = []
    kiwi_col: list[float | None] = []
    miss_rows: list[dict[str, Any]] = []

    for _, r in df.iterrows():
        testset = str(r["testset"])
        pair = str(r["pair"])
        sys_id = str(r["system_id"])
        seg_pos = int(r["seg_id"])
        year = str(r["year"])
        evs = _ev(testset, pair)
        seg_keys = list(evs._scores.get("seg", {}).keys())  # noqa: SLF001

        def pull(predicates: tuple[str, ...], excludes: tuple[str, ...] = ()) -> float | None:
            cand = _pick_metric(seg_keys, includes=predicates, excludes=excludes)
            if not cand:
                return None
            m = evs.Scores("seg", cand)
            if m is None or sys_id not in m:
                return None
            vec = m[sys_id]
            if seg_pos >= len(vec):
                return None
            v = vec[seg_pos]
            return float(v) if v is not None and np.isfinite(v) else None

        g = pull((r"GEMBA.*MQM",))
        if year in ("wmt22", "wmt23"):
            mx = pull(
                (r"MetricX-23", r"metricx_xxl"),
                excludes=(r"QE", r"QE-"),
            )
        else:
            mx = pull((r"MetricX-24", r"MetricX-24-Hybrid"), excludes=(r"QE", r"QE-"))
        xc = pull((r"^XCOMET", r"MS-COMET"), excludes=(r"QE", r"QE-"))
        kw = pull((r"CometKiwi", r"COMETKiwi"), excludes=(r"docWMT",))

        gemba_col.append(g)
        metricx_col.append(mx)
        xcomet_col.append(xc)
        kiwi_col.append(kw)

    out = df.copy()
    out["gemba_mqm"] = gemba_col
    out["metricx"] = metricx_col
    out["xcomet"] = xcomet_col
    out["cometKiwi"] = kiwi_col

    for pair in sorted(out["pair"].astype(str).unique()):
        sub = out[out["pair"].astype(str) == pair]
        rec: dict[str, Any] = {"pair": pair}
        for c in ("gemba_mqm", "metricx", "xcomet", "cometKiwi"):
            rec[f"missing_{c}"] = int(sub[c].isna().sum()) if c in sub.columns else len(sub)
        miss_rows.append(rec)
    return out, pd.DataFrame(miss_rows)


def run_fluency2_checkpoint(inp: Path, outp: Path, ckpt: Path) -> None:
    inp = inp.resolve()
    work = pd.read_parquet(inp)
    work = work.rename(
        columns={"src_text": "original_text", "mt_text": "translation"}
    )
    sl, tl = [], []
    for _, r in work.iterrows():
        a, b = _pair_langs(str(r["pair"]))
        sl.append(a)
        tl.append(b)
    work["source_lang"] = sl
    work["target_lang"] = tl
    tmp = ckpt.parent / f".wmt_mqm_fluency_input{inp.stem}.parquet"
    work.to_parquet(tmp, index=False)
    env = os.environ.copy()
    model = (os.environ.get("FLUENCY2_GEMINI_MODEL") or "").strip()
    env["FLUENCY2_GEMINI_MODEL"] = model or DEFAULT_FLUENCY2_GEMINI_MODEL
    print(f"FLUENCY2_GEMINI_MODEL={env['FLUENCY2_GEMINI_MODEL']}", flush=True)
    # compute_fluency2_gemini writes checkpoints to the same path as --output
    cmd = [
        sys.executable,
        str(ROOT / "algebras-ml" / "tools" / "compute_fluency2_gemini.py"),
        "--input",
        str(tmp),
        "--output",
        str(ckpt),
        "--max-concurrent",
        "20",
        "--checkpoint-every",
        "500",
        "--resume",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    tmp.unlink(missing_ok=True)
    shutil.copyfile(ckpt, outp)


def correlation_block(df: pd.DataFrame, metrics: list[str]) -> list[dict[str, Any]]:
    work = df.copy()
    work["human_score"] = pd.to_numeric(work["human_mqm_score"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for pair, g in work.groupby("pair"):
        for met in metrics:
            sub = g.dropna(subset=[met, "human_score"])
            r, t, n = _spearman_kendall(sub, met)
            pa = _pairwise_accuracy_col(g, met) if met in g.columns else float("nan")
            rows.append(
                {
                    "scope": "pair",
                    "pair": pair,
                    "metric": met,
                    "spearman_rho": r,
                    "kendall_tau": t,
                    "n": n,
                    "pairwise_accuracy": pa,
                }
            )
    for met in metrics:
        sub = work.dropna(subset=[met, "human_score"])
        r, t, n = _spearman_kendall(sub, met)
        pa = _pairwise_accuracy_col(work, met)
        rows.append(
            {
                "scope": "overall_all_pairs",
                "pair": "ALL",
                "metric": met,
                "spearman_rho": r,
                "kendall_tau": t,
                "n": n,
                "pairwise_accuracy": pa,
            }
        )
    return rows


def esa_comparison(mqm_corr: Path, esa_csv: Path, out_csv: Path) -> None:
    esa: dict[str, dict[str, str]] = {}
    with esa_csv.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 6:
                continue
            if parts[0] == "TABLE1_overall" and parts[1] == "metric":
                continue
            if parts[0] == "TABLE1_overall":
                _, metric, sp, _, _, pa = parts[:6]
                esa[metric] = {"ESA_spearman": sp, "ESA_PA": pa}
    mqm_df = pd.read_csv(mqm_corr)
    overall = mqm_df[mqm_df["scope"] == "overall_all_pairs"].copy()
    metric_map = {
        "fluency2": "fluency2",
        "chrf_pp": "chrf_pp",
        "gemba_score": "gemba_mqm",
        "metricx_score": "metricx",
        "xcomet_score": "xcomet",
        "cometKiwi_score": "cometKiwi",
    }
    out_rows = []
    for esa_m, mqm_m in metric_map.items():
        e = esa.get(esa_m, {})
        mrow = overall[overall["metric"] == mqm_m]
        mqm_s = float(mrow["spearman_rho"].iloc[0]) if len(mrow) else float("nan")
        mqm_pa = float(mrow["pairwise_accuracy"].iloc[0]) if len(mrow) else float("nan")
        out_rows.append(
            {
                "metric": esa_m,
                "ESA_spearman": e.get("ESA_spearman", ""),
                "ESA_PA": e.get("ESA_PA", ""),
                "MQM_spearman": mqm_s,
                "MQM_PA": mqm_pa,
            }
        )
    comp = pd.DataFrame(out_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_csv, index=False, na_rep="")
    print("\n=== ESA vs WMT22–24 MQM (overall) ===")
    print(comp.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-fluency2", action="store_true")
    args = ap.parse_args()

    bootstrap_dotenv()

    if not args.skip_download:
        download_mtme_data()
    if args.download_only:
        return

    mt_root = _mtme_root()
    if not mt_root.is_dir():
        raise SystemExit(
            f"MTME root not found: {mt_root}\n"
            "Install the toolkit from https://github.com/google-research/mt-metrics-eval "
            "and obtain extracted ``mt-metrics-eval-v2`` (WMT metric task bundle). "
            "Anonymous download from storage.googleapis.com often returns 403; use an "
            "authenticated mirror or copy from a machine that already has the data, then:\n"
            f"  export MT_METRICS_EVAL_ROOT=/path/to/mt-metrics-eval-v2\n"
            f"or place the tree at: {ROOT / 'data_interim' / 'mt-metrics-eval-v2'}"
        )

    mt_data = _ensure_mtme_import()
    _ = mt_data  # noqa: F841

    step_inspect_and_csv(mt_root)

    sampled = build_sampled_parquet(mt_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = OUT_DIR / "sampled_segments.parquet"
    sampled.to_parquet(sp, index=False)
    print(f"Total rows sampled: {len(sampled)}")
    print(sampled.groupby("pair").size())

    scored = step_chrf_pp(sampled)
    with_metrics, miss_rep = attach_stored_metrics(scored, mt_root)
    wm = OUT_DIR / "sampled_with_metrics.parquet"
    with_metrics.to_parquet(wm, index=False)
    print("Missing counts per pair (metrics):")
    print(miss_rep.to_string(index=False))

    if not args.skip_fluency2:
        final_p = OUT_DIR / "all_metrics_mqm.parquet"
        ckpt = OUT_DIR / "fluency2_checkpoint.parquet"
        try:
            run_fluency2_checkpoint(wm, final_p, ckpt)
        except subprocess.CalledProcessError as e:
            print(f"Fluency2 failed ({e}); wrote metrics without fluency2 to {wm}")
            with_metrics.to_parquet(final_p, index=False)
    else:
        final_p = OUT_DIR / "all_metrics_mqm.parquet"
        with_metrics.to_parquet(final_p, index=False)

    final_df = pd.read_parquet(final_p)
    metrics = ["fluency2", "chrf_pp", "gemba_mqm", "metricx", "xcomet", "cometKiwi"]
    metrics = [m for m in metrics if m in final_df.columns]
    corr_rows = correlation_block(final_df, metrics)
    corr_path = OUT_DIR / "mqm_correlation_results.csv"
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}")

    esa_path = ROOT / "data_interim" / "wmt25_full_analysis" / "full_analysis_results.csv"
    if esa_path.is_file():
        esa_comparison(corr_path, esa_path, OUT_DIR / "comparison_esa_vs_mqm.csv")
    else:
        print(f"Skip ESA comparison; missing {esa_path}")


if __name__ == "__main__":
    main()
