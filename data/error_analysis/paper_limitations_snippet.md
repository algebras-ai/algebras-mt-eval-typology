## Limitations: data coverage

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
GPT-5's fluency2 scores on OpenAI-family translations are below the mean human score (Δ = -0.085σ on z-scaled metric vs human), indicating no detectable self-preference for this judge on this data. A full three-judge self-preference
matrix was out of scope for this submission due to budget constraints.
