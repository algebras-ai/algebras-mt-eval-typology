## Key discovery (dup_marker_sum per parquet; 0 required)

```
KEY ['seg_id', 'system_id'] dup_marker_sum: {'key': "['seg_id', 'system_id']", 'sampled_segments': 22, 'fluency2_opus47': 22, 'fluency2_gpt5': 22, 'fluency_mqm_gemini': 22, 'fluency_mqm_gpt5': 22}
KEY ['seg_id', 'system_id', 'year'] dup_marker_sum: {'key': "['seg_id', 'system_id', 'year']", 'sampled_segments': 15, 'fluency2_opus47': 15, 'fluency2_gpt5': 15, 'fluency_mqm_gemini': 15, 'fluency_mqm_gpt5': 15}
KEY ['seg_id', 'system_id', 'dataset'] dup_marker_sum: {'key': "['seg_id', 'system_id', 'dataset']", 'sampled_segments': 15, 'fluency2_opus47': 15, 'fluency2_gpt5': 15, 'fluency_mqm_gemini': 15, 'fluency_mqm_gpt5': 15}
KEY ['seg_id', 'system_id', 'year', 'pair'] dup_marker_sum: {'key': "['seg_id', 'system_id', 'year', 'pair']", 'sampled_segments': 0, 'fluency2_opus47': 0, 'fluency2_gpt5': 0, 'fluency_mqm_gemini': 0, 'fluency_mqm_gpt5': 0}
```

## Key schema

**Chosen key:** `['seg_id', 'system_id', 'year', 'pair']`

- Merged rows: 2758; `duplicated(chosen_key).sum()` = **0**

## Scale handling

- `fluency2_opus47` `fluency2` min/max: 1 / 8.95

- `fluency2_opus47` `fluency2_raw` min/max: 1 / 10

- `fluency2_gpt5` `fluency2` min/max: 1 / 8.95

- `fluency2_gpt5` `fluency2_raw` min/max: 1 / 10

- `fluency_mqm_gpt5` columns matching error/issue/json: `[]`


**Decision:** fluency2_raw ×10 → 0–100 iff max(raw)≤10 → **True** (max raw observed: 10)

## Z-score normalization

**groupby columns:** `['year', 'pair']` (groups with n<30 → z = NaN, not 0)

- Groups with n<30: **0** (z suppressed inside those)

## Data availability

```

  ('opus47', 'f2'): True

  ('gemini', 'f2'): False

  ('gpt5', 'f2'): True

  ('opus47', 'fmqm'): False

  ('gemini', 'fmqm'): True

  ('gpt5', 'fmqm'): True

```

## Files removed

- (none)


**B slices:** merged `slice_B_f2_pooled` + `slice_B_fmqm_pooled` → `slice_B_low_human.parquet` (identical row keys).

## Jaccard flags (0.0 / 1.0)

- (none)

## Slice purity

- all checks passed

## Variance collapse (recomputed)

| mode   | judge   |    n |   pct_ge_98_on_0_100_scale |   std_raw_on_0_100 | f2_mult_x10_applied   | note   |
|:-------|:--------|-----:|---------------------------:|-------------------:|:----------------------|:-------|
| f2     | opus47  | 2727 |                 0.00580131 |           18.3203  | True                  |        |
| f2     | gpt5    | 2737 |                 0.00362582 |           17.0253  | True                  |        |
| fmqm   | gemini  | 2747 |                 0.415881   |            9.66281 | False                 |        |
| fmqm   | gpt5    | 2758 |                 0.230239   |           15.9116  | False                 |        |

## Taxonomy coverage

- P(`other`) weighted: **9.1%**


### Proposed taxonomy extensions (top-20 unmapped strings)

- none (535)

## system_id (unique)

- n_unique=59 (showing top 30 by frequency)

| system_id        |   count |
|:-----------------|--------:|
| ONLINE-B         |     125 |
| refA             |     124 |
| IOL_Research     |      84 |
| Unbabel-Tower70B |      70 |
| Llama3-70B       |      70 |
| Gemini-1.5-Pro   |      70 |
| CommandR-plus    |      70 |
| MSLC             |      69 |
| Aya23            |      69 |
| GPT-4            |      68 |
| Claude-3.5       |      68 |
| ONLINE-A         |      65 |
| ONLINE-W         |      62 |
| ZengHuiMT        |      55 |
| chrf_bestmbr     |      55 |
| M2M100_1.2B-B4   |      55 |
| Online-A         |      55 |
| Online-B         |      55 |
| Lan-Bridge       |      55 |
| Online-G         |      55 |
| Online-W         |      55 |
| Online-Y         |      55 |
| JDExploreAcademy |      55 |
| GPT4-5shot       |      55 |
| Lan-BridgeMT     |      55 |
| NLLB_Greedy      |      55 |
| NLLB_MBR_BLEU    |      55 |
| bleu_bestmbr     |      55 |
| ONLINE-G         |      55 |
| ONLINE-Y         |      55 |

## system_id families

| family    |   count |
|:----------|--------:|
| Other     |    2427 |
| OpenAI    |     123 |
| Google    |      70 |
| Meta      |      70 |
| Anthropic |      68 |

## GPT-5 fluency_mqm columns

`[]`

## Taxonomy labeller clarification

Column **`labeller`** in `issue_types.csv` is the **fmqm head** (`opus47`/`gemini`/`gpt5`) whose `fmqm_errors_json_*` / `fmqm_issue_*` column was read for that aggregation row. It is **not** the fluency2 residual judge named in the slice id (e.g. `A:f2:gpt5`).

## Known gaps


- `fluency2_gemini.parquet` missing → no Gemini fluency2 scores.

- `fluency_mqm_opus47.parquet` missing → no Opus fluency-MQM scores.

- GPT-5 fluency-MQM parquet typically has **no** `errors_json` (see above).


## Cleared for interpretation?

**Yes**
