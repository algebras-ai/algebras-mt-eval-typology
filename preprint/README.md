# Preprint (short)

**Title:** Syntactic Typological Distance Predicts When LLM Judges Outperform Lexical Metrics for Machine Translation Evaluation  

**Authors:** Airana Mongush, Dmitrii Pukhov (R&D at [Algebras AI](https://algebras.ai))

## Abstract

We study when a structured LLM-based fluency judge outperforms lexical overlap metrics (e.g. chrF) on a multi–language-pair benchmark with human adequacy scores. Per–language-pair analysis shows that **syntactic typological distance** from English predicts the judge’s **pairwise accuracy advantage** over chrF, with a strong Spearman correlation and significance after multiple-test correction on primary distance measures. Ablations over sub-dimension weights and score compression show small effects relative to cross–language-pair variance.

## Key result (summary)

Syntactic distance from English is the primary typological predictor of **when** the structured judge gains **pairwise accuracy** over chrF in this setting; the full numerical report is reproduced from the data and scripts in this repository.

## Full manuscript

The **camera-ready / submission PDF and full LaTeX source are not distributed** in this public repository (pending or completed venue-specific publication). This repo provides **data, code, figures, and optional LaTeX table fragments** for independent reproduction of the empirical results.

## LaTeX tables (optional)

Running `python code/09_tables.py` writes `preprint/tables/tab1_global_results.tex` … `tab4_weight_ablation.tex` (ignored by git if listed in `.gitignore`; generate locally after running the analysis pipeline).
