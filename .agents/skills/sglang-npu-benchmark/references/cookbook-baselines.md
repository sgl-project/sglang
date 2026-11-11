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

## Cross-machine check (16-NPU A3, CANN 9.1.0)

Re-validated on a second Atlas A3 node (8 cards x 2 dies = 16 NPU, openEuler 22.03 SP4,
CANN 9.1.0, driver 25.5.0, SGLang 0.0.0.dev18415, torch_npu 2.10.0.post6), Qwen3-8B BF16,
`random-ids`, gsm8k gate on (20 questions):

| config | output tok/s | per card | p99 TTFT | p99 TPOT | gsm8k | SLA |
|---|---|---|---|---|---|---|
| TP=2, conc 1 | 53.8 | 26.9 | 120.8 ms | 19.3 ms | 0.900 | pass |
| TP=2, conc 32 | 1099.1 | 549.6 | 6848 ms | 30.6 ms | - | FAIL (saturated-queue TTFT, expected) |
| TP=8, conc 32 | 1491.7 | 186.5 | 767.7 ms | 22.4 ms | 0.900 | pass |

TP=8 vs TP=2 at conc 32: +36% aggregate throughput and p99 TTFT 6.8 s -> 0.77 s.

Same-machine pitfall confirmations (see the pitfalls reference): with `dataset_name: random`
the bench client failed on the offline node with `LocalEntryNotFoundError` (the HF hub is still
contacted for the tokenizer); `random-ids` + `HF_HUB_OFFLINE=1` + local model path fixed it.
When the gsm8k script was not present at the expected `benchmark/gsm8k/` path, the accuracy gate
degraded to `accuracy=None` and the performance data was kept, as designed.
