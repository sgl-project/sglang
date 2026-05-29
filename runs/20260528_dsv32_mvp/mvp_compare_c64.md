# Double Sparsity vs Native NSA — Comparison Report

- native_nsa source: `runs/20260528_dsv32_mvp/smoke_results/native_nsa_gsp_isl4096_osl512_c64_t1.jsonl`
- double_sparsity source: `runs/20260528_dsv32_mvp/smoke_results/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl`
- concurrency: 64

| Metric | native_nsa | double_sparsity |
|--------|------------|-----------------|
| Per-request output tok/s P50 | 33.99 | 38.92 |
| Per-request output tok/s P99 | 40.20 | 39.16 |
| TTFT P50 (s) | 87.64 | 252.73 |
| TTFT P99 (s) | 154.91 | 501.68 |
| TPOT P50 (ms) | 29.48 | 25.74 |
| TPOT P99 (ms) | 193.07 | 28.56 |
| Goodput-under-SLO | — | — |
| Selected tokens (mean) | — | — |
| Total tokens (mean) | — | — |
| dense_fallback_total | — | — |

**DS SLO verdict (per-request P50 ≥ 30.0 tok/s, P99 TTFT ≤ 22.0 s):** fail
**No-op detector:** unknown (no-op inputs missing: dense_fallback_total, selected_tokens_mean, total_tokens_mean)
