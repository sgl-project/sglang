# AC-11 Directional Comparator — DS vs DSA

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ DSA P99 TTFT × 1.10. At least 3 trials per concurrency, median.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |
|------|-------------|------------|-----------|----------|--------------|-------------|------------|-----------|
| 16 | 46.876 | 34.040 | 0.726 | FAIL | 0.730 | 57.692 | 79.046 | FAIL |
| 32 | 37.640 | 33.882 | 0.900 | FAIL | 1.374 | 132.853 | 96.721 | FAIL |
| 64 | 29.598 | 33.923 | 1.146 | pass | 2.038 | 291.988 | 143.260 | FAIL |

## Effective vs nominal concurrency (#F)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.994 | 14.502 | 91% |
| 32 | 31.982 | 24.593 | 77% |
| 64 | 63.932 | 35.703 | 56% |

When DS achieved concurrency is below nominal while DSA tracks nominal, the DS P99 TTFT gap is partly queue/admission-bound (a mem-0.6 KV-pool effect), not solely per-request latency. Per DEC-7 a TTFT/TPS miss is a recorded directional follow-up, not a build-break.

## AC-11 verdict: FAIL

**Profiling obligation:** the failing concurrencies below require a captured profile (`development/profile_ds.sh` or equivalent) before the comparator row can be published.

- conc=16: AC-11 TPS gate failed: DS/DSA = 0.7262 < 0.95 (DS=34.04 tok/s, DSA=46.88 tok/s); AC-11 TTFT gate failed: DS/DSA P99 = 79.0460 > 1.1 (DS=57.692 s, DSA=0.730 s)
- conc=32: AC-11 TPS gate failed: DS/DSA = 0.9001 < 0.95 (DS=33.88 tok/s, DSA=37.64 tok/s); AC-11 TTFT gate failed: DS/DSA P99 = 96.7213 > 1.1 (DS=132.853 s, DSA=1.374 s)
- conc=64: AC-11 TTFT gate failed: DS/DSA P99 = 143.2600 > 1.1 (DS=291.988 s, DSA=2.038 s)
