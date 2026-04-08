# Syntactic Typological Distance Predicts When LLM Judges Outperform Lexical Metrics for MT Evaluation

**Authors:** Airana Mongush, Dmitrii Pukhov  
**Affiliation:** R&D at Algebras AI ([algebras.ai](https://algebras.ai))

## Finding

Syntactic typological distance from English predicts when a structured
LLM judge gains pairwise accuracy advantage over chrF on a 12-pair
EN→X frontier-model benchmark with WMT25 human adequacy scores
(ρ = 0.77, Holm p = 0.04, 95% CI [0.31, 0.96]).

## Reproduce

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

## Data

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

Репозиторий: [github.com/algebras-ai/algebras-mt-eval-typology](https://github.com/algebras-ai/algebras-mt-eval-typology) (организация [Algebras AI](https://github.com/algebras-ai)).

Клонирование:

```bash
git clone https://github.com/algebras-ai/algebras-mt-eval-typology.git
```

Push с машины, где установлен [GitHub CLI](https://cli.github.com/) и выполнен `gh auth login`:

```bash
gh auth setup-git
git remote set-url origin https://github.com/algebras-ai/algebras-mt-eval-typology.git
git push -u origin main
```

Вариант по SSH (`git@github.com:algebras-ai/algebras-mt-eval-typology.git`) — только если в аккаунте GitHub добавлен SSH-ключ.

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@article{mongush-pukhov-2026-typological,
  title={Syntactic Typological Distance Predicts When LLM Judges
         Outperform Lexical Metrics for MT Evaluation},
  author={Mongush, Airana and Pukhov, Dmitrii},
  year={2026},
  note={R\&D at Algebras AI}
}
```
