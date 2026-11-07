# Cookbook Baselines — Atlas 800I A3 (2 x Ascend 910C die, TP=2, single node)

Environment: CANN 9.0.0 (official `9.0.0-a3-ubuntu22.04` container layout), torch_npu 2.10.0,
driver 25.5.5, SGLang 0.5.x-dev (NPU backend, ascend attention), sgl-kernel-npu nightly.
Workload: `random-ids`, ISL/OSL 1024/1024, request_rate unbounded, 3 repeats per cell,
per-cell gsm8k accuracy gate. All cells done, zero failed cells.

Use these numbers as sanity anchors when onboarding the same specs; absolute values will
shift with software stack versions — always re-provenance your own runs.

## Qwen3-8B (dense BF16) — 24-cell grid, mem-fraction 0.80/0.85 x conc 1/8/32/64

Observation: throughput is insensitive to mem-fraction 0.80 vs 0.85 (< 1%); one axis value
is enough for this spec.

| conc | output tok/s (mean of 3) | p99 TPOT ms | p99 TTFT ms | notes |
|---|---|---|---|---|
| 1  | 96   | 10.5 | ~97 (one 9.8 s spike in one repeat) | spike flagged by variance gate; rerun showed 111 ms |
| 8  | 631  | 12.7 | ~6200 | TTFT SLA fails (saturation queueing, P7) |
| 32 | 1718 | 18.9 | ~2550 | best SLA-clean cell 1720.8 tok/s (860.4 tok/s/card) |
| 64 | 2525 | 25.5 | ~3400-4100 | saturation; gate TPOT/throughput here |

Accuracy (gsm8k, per-cell gate): 0.945-0.948 across all cells.

## Qwen3.6-35B-A3B (MoE BF16, qwen3_5 hybrid GDN) — 18-cell grid, chunked-prefill 4096/8192 x conc 8/32/64

| conc | output tok/s | p99 TPOT ms | p99 TTFT ms | notes |
|---|---|---|---|---|
| 8  | ~393  | ~20 | ~1250-1430 | SLA-clean |
| 32 | ~955  | ~33 | ~3400-4700 | chunked-prefill 8192 slightly better (+3-5%) |
| 64 | ~1310 | ~42-44 | ~46000-49000 | saturation queueing (512 prompts / 64 conc) |

Accuracy (gsm8k): 0.630-0.652. p99 TPOT within 100 ms SLA at every concurrency.

## Stretch config result (compatibility matrix entry)

NEXTN speculative decoding (`--speculative-algorithm NEXTN`, steps 2, topk 1, draft tokens 4):
server never healthy — sgl-kernel-npu Triton kernel UB overflow in the MTP draft path
(see Pitfall P8). Base model without speculative flags: 18/18 cells green, so the failure
is specific to the draft path. Kept as a documented incompatible config, not a sweep failure.

## Adding a new model spec (checklist)

1. Size `tp_size` from weights + activation + KV pool vs per-die HBM (P3 for the KV term).
2. Set the CUDA-graph capture list to cover the max concurrency in the grid (P2).
3. Start from the closest existing cookbook YAML; change model path/dtype/tp only.
4. Dry-run: cell count, flag existence, grid validation.
5. Smoke cell first (small num_prompts) to validate health + accuracy gate, then the grid.
6. Record provenance; run >= 3 repeats; report with variance gate on.
