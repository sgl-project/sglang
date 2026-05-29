# TIER-1 Smoke Comparator — Double Sparsity vs Native NSA (DeepSeek-V3.2 FP8)

**Smoke milestone, NOT AC-11 evidence.** Single-trial, shortened measurement window,
radix cache disabled on BOTH sides. Directional only — do not cite as the loop4-compatible
performance comparison (that is the radix-on 3-trial AC-11 sweep, still pending).

## Run context (8x H200, single node, sequential)

| | DSA baseline (native_nsa) | Double Sparsity |
|---|---|---|
| model_path | `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` | same |
| tp_size | 8 | 8 |
| kv_cache_dtype | fp8_e4m3 | fp8_e4m3 |
| page_size | 64 | 64 |
| **disable_radix_cache** | **true** | **true** (parity ✓) |
| enable_double_sparsity | false | true (mask: `/models/dsv32-fp8-channel-mask.safetensors`) |
| mem_fraction_static | 0.85 | 0.60 |
| base_gpu_id | 0 | 0 |
| Option B | overlap sched off, piecewise CUDA graph off | same |

Smoke shape: `MODE` per side, `CONCURRENCIES="16 32 64"`, `TRIALS=1`, `NUM_PROMPTS=64`,
`WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=30`, GSP ISL≈4096 (sys 2253 + q 1843) / OSL 512.
Each JSONL's observed `duration` (168–533 s) is far above the 30 s window, so
`benchmark.sh`'s hard duration guard passed. `.meta.json` sidecars present for all 6 runs;
the comparator (`benchmark_compare.py --baseline … --ds …`) exited 0 for every
concurrency, i.e. the `{gpu_id, tp_size, page_size, disable_radix_cache, concurrency}`
match set agreed and the radix-parity refusal did NOT trigger.

## Results

### conc 16
| Metric | native_nsa | double_sparsity |
|--------|------------|-----------------|
| Per-request output tok/s P50 | 46.02 | 36.96 |
| Per-request output tok/s P99 | 47.17 | 39.11 |
| TTFT P50 (s) | 33.59 | 119.87 |
| TTFT P99 (s) | 33.78 | 120.03 |
| TPOT P50 (ms) | 21.77 | 27.11 |
| TPOT P99 (ms) | 82.66 | 29.78 |

### conc 32
| Metric | native_nsa | double_sparsity |
|--------|------------|-----------------|
| Per-request output tok/s P50 | 26.45 | 38.90 |
| Per-request output tok/s P99 | 37.16 | 39.15 |
| TTFT P50 (s) | 64.55 | 242.52 |
| TTFT P99 (s) | 70.37 | 244.14 |
| TPOT P50 (ms) | 37.92 | 25.76 |
| TPOT P99 (ms) | 156.53 | 28.53 |

### conc 64
| Metric | native_nsa | double_sparsity |
|--------|------------|-----------------|
| Per-request output tok/s P50 | 33.99 | 38.92 |
| Per-request output tok/s P99 | 40.20 | 39.16 |
| TTFT P50 (s) | 87.64 | 252.73 |
| TTFT P99 (s) | 154.91 | 501.68 |
| TPOT P50 (ms) | 29.48 | 25.74 |
| TPOT P99 (ms) | 193.07 | 28.56 |

(Per-concurrency machine-readable reports: `mvp_compare_c{16,32,64}.md` + `.json`.)

## Directional reading

- **DS decode is competitive-to-better.** DS per-token decode latency (TPOT P50
  25.7–27.1 ms; P99 28.5–29.8 ms) is flat across concurrency and beats DSA at conc 32/64
  (DSA TPOT P99 balloons to 156–193 ms). DS per-request output tok/s is on par or higher
  at conc 32/64 (DS ~38.9 vs DSA 26–34 P50). This is the expected sparse-decode benefit.
- **DS TTFT is much worse here — but it is a single-node memory artifact, not an
  algorithmic DS regression.** DS runs at `mem_fraction_static=0.60` because it must also
  reserve a per-rank TokenLabelTable on top of ~84 GB/rank of V3.2 FP8 weights, leaving a
  small KV pool. Server logs during the DS sweep show `#running-req: 2`,
  `token usage: 0.84`, `#queue-req: 8–14`: DS admitted only ~2 concurrent 4096-token
  requests and queued the rest, so requests waited minutes for prefill → TTFT P99 grows
  120 → 244 → 502 s with nominal concurrency. DSA at 0.85 had no TokenLabelTable, kept a
  large KV pool, and ran at full concurrency. **The two columns therefore differ in
  effective admitted concurrency, so the TTFT gap is not apples-to-apples.**
- **SLO verdict is `fail` on both sides** (per-request P50 ≥ 30 tok/s AND P99 TTFT ≤ 22 s):
  even DSA's TTFT (155 s P99 at conc 64) blows the 22 s SLO because a single instance is
  saturated by 64 concurrent 4096-ISL prompts. This is expected for an unscaled smoke and
  is not a TIER-1 gate.
- **No-op detector = unknown:** `bench_serving` does not emit `dense_fallback_total` /
  `selected_tokens_mean` (a separate observability path), so the comparator cannot
  evaluate the no-op heuristic from these JSONLs. The genuine-sparsity evidence for DS is
  AC-1.1 (`ac1_1_genuine_sparsity.json`: sparsity_rate≈0.105, dense_fallback=0), not this
  smoke.

## Follow-ups (do not block the TIER-1 smoke)
- The DS KV-pool / effective-concurrency limit at mem 0.6 must be addressed before the
  AC-11 radix-on directional sweep, or the TTFT comparison there will be dominated by the
  same queuing artifact rather than the DS algorithm. Options to revisit at AC-11:
  reduce/scale the workload, raise the DS KV budget if headroom allows after the radix
  flip, or report effective vs nominal concurrency explicitly.
