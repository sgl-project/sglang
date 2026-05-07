## Triton Fused Store Cache — AMD Instinct MI300X microbenchmark

**Device:** AMD Instinct MI300X (gfx942, FP8 FNUZ)  |  ROCm 7.2.26015  |  page_size=256
**Iterations:** 200 measured (median µs), 200 warmup

### flashmla (input [N, 512] BF16 → paged SWA KV cache)
| num_tokens | fused (µs) | two-step (µs) | speedup |
| ---------- | ---------- | ------------- | ------- |
| 1 | 52.1 | 82.4 | 1.58× |
| 8 | 52.1 | 86.9 | 1.67× |
| 32 | 50.5 | 81.8 | 1.62× |
| 64 | 48.4 | 81.9 | 1.69× |
| 128 | 48.5 | 82.5 | 1.70× |
| 256 | 48.5 | 81.1 | 1.67× |
| 512 | 48.0 | 81.9 | 1.71× |

### indexer (input [N, 128] BF16 → paged C4 indexer KV cache)
| num_tokens | fused (µs) | ref-quant (µs) | speedup |
| ---------- | ---------- | -------------- | ------- |
| 1 | 46.9 | 51.6 | 1.10× |
| 8 | 47.0 | 51.3 | 1.09× |
| 32 | 48.3 | 51.7 | 1.07× |
| 64 | 49.1 | 52.3 | 1.06× |
| 128 | 47.2 | 51.3 | 1.09× |
| 256 | 49.4 | 54.4 | 1.10× |
| 512 | 52.2 | 54.2 | 1.04× |

### Cross-arch summary

| Chip | flashmla speedup (avg) | µs saved/call |
|------|------------------------|---------------|
| MI350X (gfx950, E4M3) | 1.62× | ~27 µs |
| MI300X (gfx942, FNUZ) | **1.66×** | **~33 µs** |

Speedup is consistent across both AMD architectures. MI300X benefits slightly more in absolute terms because its baseline kernel-launch overhead is higher, so amortising one launch matters more.
