## Triton Fused Store Cache — MI350X microbenchmark

**Device:** AMD Instinct MI350X  |  ROCm 7.2  |  page_size=256
**Iterations:** 200 measured (median µs), 50 warmup

### flashmla (input [N, 512] BF16 → paged SWA KV cache)
| num_tokens | fused (µs) | two-step (µs) | speedup |
| ---------- | ---------- | ------------- | ------- |
| 1 | 43.7 | 70.7 | 1.62× |
| 8 | 42.8 | 69.5 | 1.62× |
| 32 | 42.1 | 69.0 | 1.64× |
| 64 | 42.4 | 69.0 | 1.63× |
| 128 | 42.4 | 69.0 | 1.63× |
| 256 | 42.8 | 69.1 | 1.61× |
| 512 | 42.6 | 69.4 | 1.63× |

### indexer (input [N, 128] BF16 → paged C4 indexer KV cache)
| num_tokens | fused (µs) | ref-quant (µs) | speedup |
| ---------- | ---------- | -------------- | ------- |
| 1 | 40.1 | 42.7 | 1.07× |
| 8 | 40.7 | 42.5 | 1.04× |
| 32 | 40.9 | 42.7 | 1.04× |
| 64 | 40.8 | 42.9 | 1.05× |
| 128 | 40.6 | 42.8 | 1.05× |
| 256 | 41.1 | 43.1 | 1.05× |
| 512 | 41.0 | 43.0 | 1.05× |

### Notes
- flashmla baseline: `quant_to_nope_fp8_rope_bf16_pack_triton` + `_set_k_and_s_triton` (exact production two-step fallback)
- indexer baseline: `act_quant` only (quant-only, no paged scatter); real production fallback also pays `set_index_k_scale_buffer` scatter overhead on top, so production speedup > 1.05×
- Flat speedup across all batch sizes indicates bottleneck is kernel-launch overhead + HBM intermediate tensor round-trip, not arithmetic (doesn't amortise with batch size)
- ~27 µs/call saved on flashmla × 2 calls/layer × 61 layers = ~3.3 ms/forward-pass at decode time
