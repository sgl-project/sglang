# Round 14 Contract

## Mainline Objective

Execute the two next active tasks from the plan in order:
1. `task-ac4-hwrun` (analyze/Codex) — H200 hardware calibration run
2. `task-ac5-tp` (coding/Claude) — TP=2 multiprocess all-reduce test harness

If `task-ac4-hwrun` is hardware-gated (H200 unavailable in this environment), document
the gate and proceed immediately to `task-ac5-tp`.

## Target ACs

- **AC-4** (hardware gate): generate and validate `/models/dsv32-fp8-channel-mask.safetensors`
  on H200 via `--dtype bfloat16 --kv-cache-dtype fp8_e4m3 --label-dim 16 --page-size 64
  --num-samples 256 --block-size 512 --seed 42`.
- **AC-5** (coding): TP=2 multiprocess all-reduce harness in
  `test/registered/integration/test_double_sparsity_tp_multiprocess.py`.

## Blocking Issues

None currently blocking. Both tasks are the next ordered items in the plan.

## Queued (Out of Scope This Round)

- `task-ac6-cuda-graph`, `task-ac1-hwtest`, `task-ac1b-probe`: follow AC-5 in execution order
- All hardware/analyze gates: AC-1b, AC-8, AC-9–12

## Success Criteria

### task-ac4-hwrun (analyze/Codex)
1. Route via `/humanize:ask-codex` with the exact production command.
2. If hardware is available: artifact generated, `load_channel_mask` accepts it
   with `dtype=fp8_e4m3`, `head_dim=128`, `page_size=64`, `label_dim=16`.
3. If hardware is unavailable: document as hardware gate; do not block AC-5.

### task-ac5-tp (coding/Claude)
1. `test/registered/integration/test_double_sparsity_tp_multiprocess.py` created.
2. Spawns 2 processes via `torch.multiprocessing.spawn` with gloo backend.
3. Positive test: deterministic fixture, after `all_reduce(SUM)`, both ranks produce
   bit-equal `selected_token_indices` in logical position domain.
4. Physical-slot permutation test: rank-0 and rank-1 have different physical slots for
   the same logical sequence; after all-reduce and `req_to_token`, physical indices are
   rank-specific but logical positions agree.
5. Negative test: without `all_reduce` (mock no-op), perturbed rank produces divergent
   `selected_token_indices` — confirms all-reduce is load-bearing.
6. Tests run cleanly: `PYTHONPATH=python pytest test/registered/integration/test_double_sparsity_tp_multiprocess.py -q` passes.
7. Existing unit suite: 188 passed, 0 failed.
