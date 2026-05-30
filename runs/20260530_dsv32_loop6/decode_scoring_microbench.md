# Compact-vs-fp16 Decode-Scoring Microbench (AC-3.1)

**Question:** does the int8 compact path's dequant/scale overhead in the decode-time
selection scorer eat enough of the per-request decode budget to push DS below the
`≥ 30 TPS/req` SLO — i.e. does the compact path "fix TTFT while breaking the TPS SLO"?

**Hardware:** 1× NVIDIA H200 (RLCR dev box), single GPU, no TP. The scorer kernel
(`retrieve_topk_graph_safe` → `_logical_score_kernel`) is the same Triton path the
TP=8 served decode uses; only the per-rank shapes differ (this measures one rank's
per-layer call).

## Budget derivation
- Loop-5 DS decode throughput (p50): **33.9 tok/s/req** → `1000/33.9 = 29.50 ms/token`.
- Client SLO floor: **30 tok/s/req** → `1000/30 = 33.33 ms/token`.
- **Allowable added latency budget = 33.33 − 29.50 = 3.83 ms/token.** If the int8
  scoring overhead per token stays under this, the compact path cannot by itself
  push DS below 30 TPS/req.

## Method
- Shapes: `H_local=16`, `label_dim=16`, `head_dim=128`, `top_k=2048`, `seq=4608`
  (4096 ISL + 512 OSL), concurrency ∈ {16, 32, 64}.
- The DS selection runs once per DS layer per decode step; DeepSeek-V3.2 has
  **61 layers**, so `per-token overhead = 61 × (int8_per_call − fp16_per_call)`.
- Timing: 20-iter warmup, 100-iter measured, `torch.cuda.Event` GPU timing,
  `cuda.synchronize` around the measured window. fp16 and int8 tables written from
  matched random labels.

## Results (H200)
| conc | fp16 ms/call | int8 ms/call | Δ ms/call | int8 overhead ms/token (×61) | within 3.83 ms budget |
|---:|---:|---:|---:|---:|:--:|
| 16 | 0.13769 | 0.13817 | **+0.00048** | **+0.0292** | ✅ PASS |
| 32 | 0.17576 | 0.16505 | −0.01072 | −0.6538 | ✅ PASS |
| 64 | 0.21297 | 0.19861 | −0.01436 | −0.8761 | ✅ PASS |

## Verdict — PASS (all concurrencies)
The worst-case int8 scoring overhead is **+0.029 ms/token (conc 16)**, ~**130× under**
the 3.83 ms/token budget. At conc 32/64 the int8 path is **faster** than fp16: the
int8 signatures are half the bytes, so the `_logical_score_kernel` (memory-bandwidth-
bound on the signature load) reads less, and the one extra per-(token,head) scale load
+ multiply is cheaper than the bytes saved. The compact path therefore does **not**
trade TTFT for the TPS SLO — the decode-TPS guard is satisfied.

## Caveats
- This is the **scoring-kernel** overhead in isolation (the AC-3.1 "early microbench").
  The full per-request decode TPS on the TP=8 served model is measured by the AC-5
  client-SLO benchmark on the cluster.
- The 61-layers-per-token factor is the conservative upper bound (all layers run the
  selection); the per-call delta is sub-15 µs regardless, so the conclusion is robust
  to the exact active-layer count.
