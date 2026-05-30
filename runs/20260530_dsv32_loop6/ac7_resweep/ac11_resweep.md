# AC-11 Directional Comparator — DS vs DSA

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ DSA P99 TTFT × 1.10. At least 3 trials per concurrency, median.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |
|------|-------------|------------|-----------|----------|--------------|-------------|------------|-----------|
| 16 | 47.014 | 17.711 | 0.377 | FAIL | 0.717 | 12.838 | 17.905 | FAIL |
| 32 | 37.619 | 11.546 | 0.307 | FAIL | 1.281 | 25.491 | 19.895 | FAIL |
| 64 | 29.563 | 9.796 | 0.331 | FAIL | 2.044 | 100.836 | 49.333 | FAIL |

## Effective vs nominal concurrency (#F)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.994 | 15.998 | 100% |
| 32 | 31.980 | 31.996 | 100% |
| 64 | 63.927 | 46.983 | 73% |

When DS achieved concurrency is below nominal while DSA tracks nominal, the DS P99 TTFT gap is partly queue/admission-bound (a mem-0.6 KV-pool effect), not solely per-request latency. Per DEC-7 a TTFT/TPS miss is a recorded directional follow-up, not a build-break.

## AC-11 verdict: FAIL

**Profiling obligation:** the failing concurrencies below require a captured profile (`development/profile_ds.sh` or equivalent) before the comparator row can be published.

- conc=16: AC-11 TPS gate failed: DS/DSA = 0.3767 < 0.95 (DS=17.71 tok/s, DSA=47.01 tok/s); AC-11 TTFT gate failed: DS/DSA P99 = 17.9046 > 1.1 (DS=12.838 s, DSA=0.717 s)
- conc=32: AC-11 TPS gate failed: DS/DSA = 0.3069 < 0.95 (DS=11.55 tok/s, DSA=37.62 tok/s); AC-11 TTFT gate failed: DS/DSA P99 = 19.8950 > 1.1 (DS=25.491 s, DSA=1.281 s)
- conc=64: AC-11 TPS gate failed: DS/DSA = 0.3314 < 0.95 (DS=9.80 tok/s, DSA=29.56 tok/s); AC-11 TTFT gate failed: DS/DSA P99 = 49.3333 > 1.1 (DS=100.836 s, DSA=2.044 s)
