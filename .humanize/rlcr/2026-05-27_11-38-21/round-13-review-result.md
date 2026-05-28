# Round 13 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 13 satisfies its narrow contract. I verified the latest patch against the Round 12 blockers and reran the claimed suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
188 passed, 24 warnings in 11.41s
```

The real local V3.2 config still has the problematic shape (`hidden_size=7168`, `num_attention_heads=128`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, no `head_dim`), and `calibrate.py` now handles it correctly by reading `config.qk_rope_head_dim` directly before falling back to `head_dim - qk_nope_head_dim` (`python/sglang/srt/layers/attention/double_sparsity/calibrate.py:291-305`). The Q hook width now derives from `num_heads * (k_head_dim + qk_rope_head_dim)` (`calibrate.py:341-342`), and the new regression proves Method 1 accumulation fires for a no-`head_dim` V3.2-shaped config (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:2051-2175`).

The dtype blocker is also fixed at code level. `--dtype` remains the model-load dtype, `--kv-cache-dtype` controls the saved mask metadata (`calibrate.py:518-521`, `:584-589`, `:621-629`), and the H200 recipe now uses `--dtype bfloat16 --kv-cache-dtype fp8_e4m3` in both the module header and operator doc (`docs/advanced_features/double_sparsity_calibration.md:31-47`).

## Mainline Gaps

1. **AC-4 is still not complete until the H200 artifact is generated and validated.**

   Round 13 closes `task-ac4-calibrate`, but the original AC-4 requires the actual `/models/dsv32-fp8-channel-mask.safetensors` file to be generated on H200 and accepted by `load_channel_mask` validation. That has not happened yet, so AC-4 remains partial and still blocks AC-6/AC-8/AC-12.

   Required next implementation plan:
   - Run `task-ac4-hwrun` next, routed as the plan’s analyze/Codex task.
   - Use the exact production command now documented: `--dtype bfloat16 --kv-cache-dtype fp8_e4m3 --label-dim 16 --page-size 64 --num-samples 256 --block-size 512 --seed 42`.
   - After generation, load the artifact with `load_channel_mask`, validate `dtype=fp8_e4m3`, `head_dim=128`, `page_size=64`, `label_dim=16`, record the content hash, exact command, commit SHA, output path, and wall-clock result.
   - Only then move AC-4 to complete and allow AC-6/AC-8 hardware work to consume the mask.

2. **The original lower-bound and hard gates remain incomplete.**

   Pending from the plan:
   - AC-1: H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b: chunked-prefill probe.
   - AC-5: TP=2 multiprocess all-reduce harness.
   - AC-6: CUDA graph coding plus H200 replay.
   - AC-8: 8xH200 `bench_serving` and lightweight quality smoke.
   - AC-12: hard NIAH/MMLU quality gate.
   - AC-9 through AC-11: stretch baseline/radix/comparator work.

   Required execution order remains: AC-4 hardware mask generation, AC-1 hardware population, AC-1b probe, AC-5 TP harness, AC-6 graph capture/hardware replay, AC-8 server and quality smoke, AC-12 full quality, then AC-9 through AC-11 stretch work.

## Blocking Side Issues

No new blocking side issues found in the Round 13 patch. The two Round 12 blockers are resolved.

## Queued Side Issues

1. Before AC-6, thread `req_to_token` through `capture_decode_step`; otherwise graph capture validates the wrong selector domain.
2. Before AC-8, fix DS observability page-named fields and page-count sparsity math.
3. Clean stale bind/runtime sizing comments and token-label lifetime documentation when touching those modules.

## Goal Alignment Summary

```text
ACs: 7/15 addressed (5 met, 2 partial) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC: AC-0, AC-2, AC-3, AC-7, and AC-13 are met; AC-1 and AC-4 remain partial; AC-1b, AC-5, AC-6, AC-8, AC-9, AC-10, AC-11, and AC-12 are not met.

Goal tracker update: I moved `task-ac4-calibrate` from Active Tasks to Completed and Verified for Round 13, left `task-ac4-hwrun` active, and updated the mutable Plan Version marker to Round 13 Review.

NOT COMPLETE
