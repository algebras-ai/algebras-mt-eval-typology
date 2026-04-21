# Syntactic Typological Distance Predicts When LLM Judges Outperform Lexical Metrics for MT Evaluation

**Authors:** Airana Mongush, Dmitrii Pukhov  
**Affiliation:** R&D at Algebras AI ([algebras.ai](https://algebras.ai))

## Finding

Syntactic typological distance from English predicts when a structured
LLM judge gains pairwise accuracy advantage over chrF on a 12-pair
EN→X frontier-model benchmark with WMT25 human adequacy scores
(ρ = 0.77, Holm p = 0.04, 95% CI [0.31, 0.96]).

## fluency2 Evaluation (WMT25 ESA + WMT22–24 MQM)

This repository includes a **standalone Jupyter notebook** and **frozen parquet** artifacts for the fluency2 structured judge (Gemini 3 Flash, five subdimensions) evaluated against human scores on:

- **WMT25** General MT Task (ESA protocol, 11 language pairs)
- **WMT22–24** General MT Task (MQM protocol; six language pairs in the sample)

| Artifact | Path |
|----------|------|
| Notebook | [`fluency2_wmt_evaluation.ipynb`](fluency2_wmt_evaluation.ipynb) (repo root; a copy may also exist under `notebooks/`) |
| WMT25 merged metrics | [`data/wmt25/merged_all_metrics.parquet`](data/wmt25/merged_all_metrics.parquet) |
| WMT22–24 MQM metrics | [`data/wmt_mqm/all_metrics_mqm.parquet`](data/wmt_mqm/all_metrics_mqm.parquet) |

**Key results (see notebook for tables and figures).** Segment-level correlations use **fair subsets**—rows where the human score and *all* compared automatic metrics are non-missing—so Spearman and pairwise accuracy (PA) are comparable across metrics.

- **WMT25 ESA (fair, n = 3,274, all six auto metrics + human):** fluency2 Spearman **0.615**, PA **0.726**; GEMBA-ESA **0.555** / **0.709**; MetricX **0.519** / **0.688**.
- **WMT22–24 MQM — fair subset (n = 824, all four metrics + human; WMT22 triples only):** fluency2 Spearman **0.573**, PA **0.753**; GEMBA-MQM **0.520** / **0.699**; GEMBA-DA **0.476** / **0.674**; chrF++ **0.350** / **0.652**.
- **WMT22–24 MQM — full human-scored rows (per-metric n varies):** fluency2 Spearman **0.537**, PA **0.737** on **n = 2,658**; GEMBA/chrF++ remain on **n = 824** as above.
- **Pairwise accuracy by language pair (WMT25):** fluency2 leads on **9/11** pairs vs baselines in the notebook (Table 3).

### Running the notebook

The notebook reads the included parquet files directly (no recomputation).

```bash
pip install pandas numpy scipy matplotlib jupyter
jupyter notebook fluency2_wmt_evaluation.ipynb
```

Launch Jupyter from the **repository root** (or open the notebook from your IDE with cwd at the root) so paths resolve to `data/wmt25/` and `data/wmt_mqm/`.

### Reproducing from scratch

#### WMT25 data

1. Clone the WMT25 General MT repository:

```bash
git clone https://github.com/wmt-conference/wmt25-general-mt.git
```

2. Download human scores:

```bash
wget https://github.com/wmt-conference/wmt25-general-mt/raw/main/data/wmt25-genmt-humeval.jsonl
```

3. Compute fluency2 (requires Gemini API key, ~\$3):

```bash
export GEMINI_API_KEY=your_key
python3 algebras-ml/tools/compute_fluency2_gemini.py \
  --input data_raw/wmt25/wmt25-genmt-humeval.jsonl \
  --output data_interim/fluency2_wmt25.parquet \
  --max-concurrent 20 --resume
```

4. Build merged dataset (from the [benchmarks](https://github.com/algebras-ai/benchmarks) monorepo layout):

```bash
python3 tools/merge_wmt25_auto_metrics.py
```

#### WMT22–24 MQM data

1. Clone MQM annotations and GEMBA precomputed scores:

```bash
git clone https://github.com/google/wmt-mqm-human-evaluation.git
git clone https://github.com/MicrosoftTranslator/GEMBA.git
```

2. Run pipeline (paths may point to your local clones):

```bash
export MT_MQM_ROOT=~/Code/wmt-mqm-human-evaluation
python3 tools/wmt_mqm_historical_pipeline.py
python3 tools/merge_gemba_into_wmt_mqm.py
```

3. Compute fluency2 on MQM data (~\$1):

```bash
python3 algebras-ml/tools/compute_fluency2_gemini.py \
  --input data_interim/wmt_mqm/sampled_segments.parquet \
  --output data_interim/wmt_mqm/all_metrics_mqm.parquet \
  --max-concurrent 20 --resume
```

Helper scripts shipped under [`tools/`](tools/) mirror the [benchmarks](https://github.com/algebras-ai/benchmarks) workflow; `build_wmt_mqm_from_human_eval_repo.py` is the recommended builder when using only the human-evaluation clone.

### Data sources (fluency2 evaluation)

- **WMT25 human scores:** [wmt-conference/wmt25-general-mt](https://github.com/wmt-conference/wmt25-general-mt)
- **WMT25 automatic scores:** GEMBA-ESA-GPT4.1, MetricX-24-Hybrid-XL, XCOMET-XL, CometKiwi-XL from the official WMT25 repository (not recomputed here)
- **WMT22–24 MQM:** [google/wmt-mqm-human-evaluation](https://github.com/google/wmt-mqm-human-evaluation)
- **GEMBA scores:** [MicrosoftTranslator/GEMBA](https://github.com/MicrosoftTranslator/GEMBA) (`mt-metrics-eval-v2` currently ships **WMT22** segment scores used in this release)
- **Syntactic distances:** lang2vec (WALS features), as in the notebook

### Limitations (fluency2 evaluation)

- fluency2 backbone: **Gemini 3 Flash** only
- Typological pattern (ρ ≈ −0.83) **significant only vs chrF++**, not vs neural metrics in the notebook’s tests
- GEMBA segment scores **WMT22 only** (three pairs) in the published GEMBA bundle
- EN→mas_KE: fluency2 PA can be low when all systems score very low on the segment set
- Potential **training-data overlap** with Gemini 3 Flash relative to WMT25 publication dates

## Reproduce (paper pipeline)

```bash
pip install -r requirements.txt

# Run all analyses (writes figures/ and preprint/tables/)
python code/01_merge_data.py
python code/02_global_correlations.py
python code/03_pairwise_accuracy_by_lp.py
PYTHONUNBUFFERED=1 python code/04_null_simulation.py          # ~30–120 min at 1000 permutations
python code/05_typological_correlation.py                      # ~10–30 min at default 10k boot/perm
python code/06_resource_proxy.py
python code/07_weight_optimization.py                            # ~10 min
python code/08_figures.py
python code/09_tables.py
```

Environment variables (optional, for faster smoke tests):

- `NULL_SIM_PERM` — permutations for `04_null_simulation.py` (default `1000`).
- `TYPOLOGY_BOOT`, `TYPOLOGY_PERM` — bootstrap and permutation counts for `05_typological_correlation.py` (defaults `10000`).

The repository ships canonical `data/null_simulation_pa.csv`, `data/bootstrap_ci_all_distances.csv`, `data/distance_vs_advantage_correlations.csv`, and `data/weight_optimization_cv.csv` from the paper evaluation run. Re-running `04`/`05`/`07` with the default settings is deterministic aside from floating-point bootstrap variation; tiny differences in CI endpoints or permutation \(p\)-values are expected if you change `TYPOLOGY_SEED` or bootstrap counts.

### Preprint (no full manuscript PDF)

A **short preprint** (title, authors, abstract summary) is in [`preprint/README.md`](preprint/README.md). The **full LaTeX/PDF paper is not included** in this public repository (pending or venue-specific publication). Optional LaTeX **table fragments** for the results can be generated with `python code/09_tables.py` into `preprint/tables/`.

## Data (paper)

| File | Rows | Description |
|------|------|-------------|
| `data/segments.csv` | 4,720 | Source segments, system translations, human scores |
| `data/judge_scores.csv` | 4,720 | Sub-dimensions + fluency aggregates (v2/v3) |
| `data/lexical_scores.csv` | 4,720 | Sentence BLEU, chrF |
| `data/comet_scores.csv` | 4,720 | COMET (wmt22-comet-da) |
| `data/typological_distances.csv` | 12 | Per-LP typological distances + PA advantage |
| `data/resource_proxy.csv` | 11 | Wikipedia article counts (LPs with coverage) for partial-correlation control |
| `data/systems.csv` | 20 | `system_id` and `model_name` |

Segment-level CSVs were exported from the internal benchmarks evaluation checkpoint (no absolute paths in released files). **Before redistributing translations or references, confirm compliance with the WMT25 data use policy.**

## Judge prompts

Files under `prompts/`:

- `v1_prompt.txt` — full text for the holistic (v1) judge (as used in internal evaluation).
- `v2_prompt.txt`, `v3_prompt.txt` — **high-level descriptions only** (no verbatim system prompts, weights, or internal paths). Full v2/v3 prompt strings are not redistributed in this repository.

## Versioning (judge naming)

| Name | Judge version | Dimensions | Compression |
|------|--------------|------------|-------------|
| fluency v1.0 | v1 (holistic) | 3: Natural, Conventional, Adaptive | None |
| fluency v2.0 | v2 (structured) | 5: idiomatic, collocational, discourse, pragmatic, calque | None (raw aggregate `fluency2_raw`) |
| fluency v2.1 | v2 (structured) | 5 (same) | Display compression (`fluency2`, g=6.25) |
| fluency v3.0 | v3 (focused) | 3: idiomatic, collocational, calque | None (`fluency3_`) |

## GitHub

Repository: [github.com/algebras-ai/algebras-mt-eval-typology](https://github.com/algebras-ai/algebras-mt-eval-typology) (organization [Algebras AI](https://github.com/algebras-ai)).

Clone:

```bash
git clone https://github.com/algebras-ai/algebras-mt-eval-typology.git
```

Push from a machine where [GitHub CLI](https://cli.github.com/) is installed and you have run `gh auth login`:

```bash
gh auth setup-git
git remote set-url origin https://github.com/algebras-ai/algebras-mt-eval-typology.git
git push -u origin main
```

Use SSH (`git@github.com:algebras-ai/algebras-mt-eval-typology.git`) only if an SSH key is added to your GitHub account.

## License

MIT — see `LICENSE`.

## Citation

Typology paper:

```bibtex
@article{mongush-pukhov-2026-typological,
  title={Syntactic Typological Distance Predicts When LLM Judges
         Outperform Lexical Metrics for MT Evaluation},
  author={Mongush, Airana and Pukhov, Dmitrii},
  year={2026},
  note={R\&D at Algebras AI}
}
```

fluency2 evaluation release (this notebook + metrics):

```bibtex
@misc{algebras2026fluency2,
  title={fluency2: A Structured LLM Judge for Machine Translation Evaluation},
  author={Algebras AI},
  year={2026},
  url={https://github.com/algebras-ai/algebras-mt-eval-typology}
}
```
