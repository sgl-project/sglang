# Double Sparsity vs Native NSA — Comparison Report

- native_nsa source: `runs/20260528_dsv32_mvp/smoke_results/native_nsa_gsp_isl4096_osl512_c16_t1.jsonl`
- double_sparsity source: `runs/20260528_dsv32_mvp/smoke_results/double_sparsity_gsp_isl4096_osl512_c16_t1.jsonl`
- concurrency: 16

| Metric | native_nsa | double_sparsity |
|--------|------------|-----------------|
| Per-request output tok/s P50 | 46.02 | 36.96 |
| Per-request output tok/s P99 | 47.17 | 39.11 |
| TTFT P50 (s) | 33.59 | 119.87 |
| TTFT P99 (s) | 33.78 | 120.03 |
| TPOT P50 (ms) | 21.77 | 27.11 |
| TPOT P99 (ms) | 82.66 | 29.78 |
| Goodput-under-SLO | — | — |
| Selected tokens (mean) | — | — |
| Total tokens (mean) | — | — |
| dense_fallback_total | — | — |

**DS SLO verdict (per-request P50 ≥ 30.0 tok/s, P99 TTFT ≤ 22.0 s):** fail
**No-op detector:** unknown (no-op inputs missing: dense_fallback_total, selected_tokens_mean, total_tokens_mean)
