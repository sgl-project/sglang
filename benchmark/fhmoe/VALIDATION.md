# DeepSeek-V4-Pro AITER FHMoE validation

## CONTINUE HERE

- AITER: `868ac1f7aa0d403a77c03e941bc5ffa28c7766d5`
- SGLang base: `3fe65e0654`
- Hardware: 8x AMD Instinct MI355X (`gfx950`), ROCm 7.2, PyTorch 2.9.1
- Status: implementation, focused tests, graph replay, full GSM8K A/B, and
  DP-attention C256 performance screening are complete.
- Result: FHMoE strict accuracy is 95.679% and is 0.607 percentage point above
  the unfused baseline. At DP8 C256, however, FHMoE reduced total throughput by
  10.860%.

## Completed checks

- AITER #4269 and #4314 are ancestors of the installed AITER commit.
- `pytest -q op_tests/test_fhmoe.py`
- `AITER_HETERO_MOE_DSV4=1 pytest -q op_tests/test_fhmoe.py`
- `python op_tests/test_moe_2stage.py --bm16-scale-boundary`
- DSV4 AOT precompile from `dsv4_fp8fp4_tuned_fmoe.csv`
- E=385/top-k=7 decode checks for C1/C2/C32/C64
- FHMoE HIP graph capture and replay
- SGLang policy, fallback, top-k, TP-sharding, native-FP8 layout/padding, and
  AITER argument-forwarding tests

## GSM8K A/B

Both runs used the same DeepSeek-V4-Pro snapshot, TP=8, EP disabled,
temperature 0, concurrency 64, 20-shot strict-match prompts, and all 1,319
questions.

| Mode | Strict | Flexible | Invalid | Latency |
|---|---:|---:|---:|---:|
| Unfused shared MLP | 1254/1319 (95.072%) | 1255/1319 (95.148%) | 0 | 242.0 s |
| AITER FHMoE | 1262/1319 (95.679%) | 1262/1319 (95.679%) | 0 | 276.5 s |

FHMoE used the expected internal `E=385`, `top-k=7` path. Serving logs showed
the tuned C1/C2/C32/C64 kernels from `dsv4_fp8fp4_tuned_fmoe.csv`, completed
without HTTP 500, E8M0-scale, worker, ROCm, or graph-replay errors.

## DP-attention C256 performance

Single fresh-server screening A/B on 8x MI355X using TP8 + DP8 attention,
TP-MoE (no EP/A2A), 8192 input / 1024 output, fixed lengths, 512 warmups, and
2,048 measured requests:

- FHMoE OFF: 32,903 total tok/s (4,113 tok/s/GPU), 3,656 output tok/s,
  median TTFT 18,098.5 ms, median TPOT 52.29 ms, median ITL 33.99 ms,
  median E2E 71,465 ms.
- FHMoE ON: 29,330 total tok/s (3,666 tok/s/GPU), 3,259 output tok/s,
  median TTFT 19,932.3 ms, median TPOT 59.05 ms, median ITL 37.85 ms,
  median E2E 78,932 ms.
- ON versus OFF: total/output throughput -10.860%, median TTFT +10.132%,
  median TPOT +12.931%, median ITL +11.368%, median E2E +10.448%.
- Both arms completed 2,048/2,048 requests without serving or ROCm errors.
  The ON server logged the expected `E=385/top-k=7` C256 tuned kernel.
- Artifacts: `/tmp/fhmoe_c256_dp_20260810_1303/{off,on}/`.

The gap is well above the 1% repeat threshold, so no counterbalanced repeat was
run for this screening decision. FHMoE should not be enabled for this DP C256
configuration without further optimization.

## TP8 concurrency sweep

Fresh-server A/B on 8x MI355X using plain TP8 (no DP attention, EP/A2A, or
TBO), 8192 input / 1024 output, fixed lengths, `conc*2` warmups, and `conc*8`
measured requests:

- Total-throughput ON gains: C2 +1.248%, C4 +1.519%, C8 +4.077%, C16 +1.673%,
  C32 +2.171%.
- Median TPOT improved at every point: C2 14.28 -> 14.09 ms, C4 14.73 ->
  14.48 ms, C8 16.21 -> 15.48 ms, C16 18.55 -> 18.18 ms, C32 23.07 ->
  22.46 ms.
- Median TTFT regressed slightly at every point (+1.3% to +2.0%), while median
  E2E improved because decode latency fell.
- All requests completed without serving or ROCm errors. Artifacts for C2-C16:
  `/tmp/fhmoe_tp8_sweep_20260810_1409/{off,on}/`.

Detailed C32 screening:

- FHMoE OFF: 10,321 total tok/s (1,290 tok/s/GPU), 1,147 output tok/s,
  median TTFT 4,923.2 ms, median TPOT 23.07 ms, median ITL 19.05 ms,
  median E2E 28,498 ms.
- FHMoE ON: 10,546 total tok/s (1,318 tok/s/GPU), 1,172 output tok/s,
  median TTFT 5,020.7 ms, median TPOT 22.46 ms, median ITL 18.20 ms,
  median E2E 27,980 ms.
- ON versus OFF: total/output throughput +2.171%, median TTFT +1.979%,
  median TPOT -2.612%, median ITL -4.482%, median E2E -1.818%.
- Both arms completed 256/256 requests without serving or ROCm errors. The ON
  server selected the expected BM16 C32 FHMoE stage-1/stage-2 kernels.
- Artifacts: `/tmp/fhmoe_c32_tp8_20260810_1344/{off,on}/`.

Unlike DP C256, plain TP8 C32 shows a modest FHMoE throughput and decode-latency
win in this screening run.

## Acceptance criteria

- FHMoE strict exact match: at least 94%.
- FHMoE versus unfused strict score: no more than 0.5 percentage point lower.
- All 1,319 requests complete successfully.
- No E8M0-scale, worker, ROCm, or graph-replay errors.

All criteria passed.
