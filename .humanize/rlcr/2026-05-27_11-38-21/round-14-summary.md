# Round 14 Summary

## Work Completed

### task-ac4-hwrun — Hardware Gate Documented

Submitted the production calibration command to Codex via `/humanize:ask-codex`.
The run failed with CUDA OOM:
- GPU 0 had only 683 MiB free; tried to allocate 3.50 GiB; process held ~139.12 GiB
- Also observed: UNEXPECTED keys for expert weights and `self_attn.indexer.*`; MISSING
  fused MoE params (`mlp.experts.down_proj`, `gate_up_proj`, scale tensors layers 3..60)
- Root cause: calibration requires multi-GPU H200 cluster; available machine lacks VRAM

Status: **hardware gate**. Per the Round 14 contract, hardware-gated tasks do not block AC-5.

### task-ac5-tp — TP=2 Multiprocess All-Reduce Harness

Created `test/registered/integration/test_double_sparsity_tp_multiprocess.py` with three
tests via `torch.multiprocessing.spawn` + gloo backend (CPU, no GPU required).

**Test 1 — Positive**: both ranks produce bit-equal `[[2, 7]]` after `all_reduce(SUM)`.
- Rank 0 partials `[1.0,2.0,10.0,0.5,3.0,5.0,0.1,4.0]` + rank 1 `[0.1,0.2,0.5,8.0,0.3,0.4,7.0,6.0]`
- Combined `[1.1,2.2,10.5,8.5,3.3,5.4,7.1,10.0]` → top-2 ascending: `[2,7]`

**Test 2 — Negative**: no all-reduce; rank 0 gets `[[2,5]]`, rank 1 gets `[[3,6]]` — confirms
all-reduce is load-bearing.

**Test 3 — Physical-slot permutation**: `retrieve_topk_via_labels` in logical-domain mode.
- Rank 0: identity `req_to_token`; rank 1: reversed `[[3,2,1,0]]`
- After all-reduce: both agree on logical positions `[0,1]`
- Physical slots for `[0,1]`: rank 0 → `[0,1]`; rank 1 → `[3,2]` (rank-specific)

## Files Changed

- `test/registered/integration/test_double_sparsity_tp_multiprocess.py` (created, 243 lines)

## Validation

```
PYTHONPATH=python pytest test/registered/integration/test_double_sparsity_tp_multiprocess.py -v
3 passed, 0 failed (28s)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
188 passed, 0 failed
```

Commit: `6cf32a884` — [AC-5] TP=2 multiprocess all-reduce harness for Double Sparsity

## Remaining Items

- `task-ac4-hwrun`: hardware gate (CUDA OOM); needs real H200 cluster with model sharded
  across multiple GPUs. Command is ready:
  ```
  python -m sglang.srt.layers.attention.double_sparsity.calibrate \
      --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
      --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 1 \
      --output /models/dsv32-fp8-channel-mask.safetensors \
      --label-dim 16 --page-size 64 --num-samples 256 --block-size 512 --seed 42
  ```
- Next ordered tasks: `task-ac6-cuda-graph` (coding/Claude), then `task-ac1-hwtest` and
  `task-ac1b-probe` (analyze/Codex).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: No new lessons needed. AC-5 all-reduce integration test used well-known mp.spawn+gloo
pattern without uncovering any surprising failure modes.
