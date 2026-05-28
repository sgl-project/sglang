# Round 11 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 11 made real AC-4 progress: the old flat-slice-before-reshape bug is fixed for the 2-D fixture path, and the new K/V and Q/RoPE sentinels cover the leakage class identified in Round 10. I reran the claimed suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
185 passed, 24 warnings in 11.45s
```

AC-4 is still not closed. The implementation does not yet satisfy the Round 11 contract or the original Pile-val-256x512 calibration recipe, and the real H200 calibration run should not start from this commit.

## Mainline Gaps

1. **AC-4 real-model hooks can fail on normal 3-D projection outputs.**

   Evidence:
   - `_extract_mla_nope_prefix` sets `T = tensor.shape[0]` and reshapes to `[T, H, nope + suffix]` (`python/sglang/srt/layers/attention/double_sparsity/calibrate.py:119-120`).
   - The hooks pass the detached projection output directly into that helper (`calibrate.py:353-357`, `:388-391`).
   - The real path loads a HuggingFace `AutoModelForCausalLM` and tokenizes prompts as `[batch, seq]` (`calibrate.py:235-242`, `:417-420`). Linear projection hooks in that path normally preserve leading dimensions, so `q_b_proj` / `kv_b_proj` outputs are `[batch, seq, width]`, not just `[tokens, width]`.
   - Direct reproducer against the new helper: `_extract_mla_nope_prefix(torch.ones(1, 3, 16), 2, 4, 4)` raises `RuntimeError: shape '[1, 2, 8]' is invalid for input of size 48`.

   Consequence: the H200 AC-4 command can fail as soon as a real prompt produces a sequence length greater than one. The tests use 2-D fixed tensors, so they do not exercise this production shape.

   Required fix:
   - Change `_extract_mla_nope_prefix` to flatten all leading dimensions before the per-head reshape: `flat = tensor.reshape(-1, tensor.shape[-1])`, then reshape `flat` to `[-1, num_heads, nope_dim + suffix_dim]`.
   - Validate the last dimension before reshaping and raise or log a clear layer-specific error through the caller.
   - Add a regression where fake `kv_b_proj` and `q_b_proj` return `[1, T, width]` tensors. The test must fail on the current helper and pass after the fix.

2. **AC-4 Pile-val path still does not produce exactly 256 fixed 512-token blocks.**

   Evidence:
   - The original plan requires "Pile validation, seed=42, 256 x 512 tokens" (`development/loop4/refined_plan_v1.md:72-73`) and the task table requires "Pile-val-256x512 dataset" (`development/loop4/refined_plan_v1.md:293`).
   - The Round 11 contract was more explicit: load Pile, shuffle with seed=42, tokenize, concatenate, and yield exactly 256 blocks of 512 tokens (`round-11-contract.md:31-33`).
   - `_pile_val_blocks` ignores `block_size` except in its docstring, returns raw text examples, and stops after `num_blocks` documents (`calibrate.py:123-150`).
   - `_collect_channel_importance` later tokenizes each document with `max_length=block_size, truncation=True` but does not concatenate, split into fixed blocks, or pad/verify length (`calibrate.py:411-418`).

   Consequence: production calibration is currently "256 shuffled Pile documents truncated to at most 512 tokens." Short documents contribute fewer tokens, and the corpus is not the fixed 256x512 reference recipe required by AC-4. The resulting channel mask would not match the contract even if the H200 run completes.

   Required fix:
   - Move Pile block construction to the point where the tokenizer is available.
   - For the default production path, tokenize shuffled Pile examples with `add_special_tokens=False`, skip empty token lists, concatenate token ids, and emit exactly `num_samples` tensors of shape `[1, block_size]`.
   - Raise a clear `RuntimeError` if the split cannot provide `num_samples * block_size` tokens.
   - Keep `--dataset` as a newline-prompt override and keep `--allow-synthetic` as the only NIAH synthetic path.
   - Add a unit test with a fake tokenizer and fake Pile rows containing short examples, proving the function concatenates across document boundaries into exact fixed-size blocks.

3. **The original lower-bound and hard gates remain incomplete.**

   Pending from the plan:
   - AC-1: H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b: chunked-prefill probe.
   - AC-4: coding fixes above, then H200 mask generation, `load_channel_mask` validation, content hash, exact command, commit SHA, and result path.
   - AC-5: TP=2 multiprocess all-reduce harness.
   - AC-6: CUDA graph coding plus H200 replay.
   - AC-8: 8xH200 `bench_serving` and lightweight quality smoke.
   - AC-12: hard NIAH/MMLU quality gate.
   - AC-9 through AC-11: baseline/radix/comparator stretch work.

## Blocking Side Issues

1. **The AC-4 operator recipe is still stale and would misdirect the hardware run.**

   Evidence:
   - `docs/advanced_features/double_sparsity_calibration.md:20-22` still says `--num-samples` default is 64 and the default dataset is NIAH synthetic.
   - `docs/advanced_features/double_sparsity_calibration.md:29-38` still recommends `--num-samples 1024 --ctx-len 4096`.
   - `docs/advanced_features/double_sparsity_calibration.md:45` still describes K-only L2-squared calibration.
   - The `calibrate.py` module header has the same stale story: K L2 at `calibrate.py:3-6`, NIAH default at `:8-11`, and `--num-samples 1024` in the production recipe at `:13-23`.

   Required fix: update both the doc and module header before `task-ac4-hwrun`. They must describe Method 1 Q/K noPE, Pile-val seed=42, 256 blocks of 512 tokens, defaults `--num-samples 256 --block-size 512 --seed 42`, and synthetic usage only behind `--allow-synthetic`.

## Queued Side Issues

1. Before AC-6, thread `req_to_token` through `capture_decode_step`; otherwise graph capture validates the wrong selector domain.
2. Before AC-8, fix DS observability page-named fields and page-count sparsity math.
3. Clean stale bind/runtime sizing comments and token-label lifetime documentation when touching those modules.
4. Claude's summary says to add `BL-20260527-reshape-before-slice-mla`, but the commit only changed `calibrate.py` and the unit test file. This is non-blocking for AC-4 code correctness, but the lesson should be captured if the loop process expects that delta.

## Goal Alignment Summary

```text
ACs: 7/15 addressed (5 met, 2 partial) | Forgotten items: 0 | Unjustified deferrals: 1 rejected
```

Status by AC:

| AC | Status | Evidence / Gap |
|----|--------|----------------|
| AC-0 | MET | Completed/verified in tracker Round 2. |
| AC-1 | PARTIAL | Local hook coverage verified; H200 population and AC-8 selector-read smoke pending. |
| AC-1b | NOT MET | Chunked-prefill probe has not run. |
| AC-2 | MET | Completed/verified in tracker Round 7. |
| AC-3 | MET | Completed/verified in tracker Round 6. |
| AC-4 | PARTIAL | Method 1 and 2-D MLA sentinels advanced, but real 3-D hook shape and fixed Pile-val block recipe are still wrong; H200 mask run pending. |
| AC-5 | NOT MET | Multiprocess TP integration test still absent. |
| AC-6 | NOT MET | Graph capture coding and H200 replay pending. |
| AC-7 | MET | Completed/verified in tracker Round 9. |
| AC-8 | NOT MET | No server benchmark or lightweight quality smoke. |
| AC-9 | NOT MET | No DSA baseline JSON. |
| AC-10 | NOT MET | Radix fixture and FP8 cold/warm proof pending. |
| AC-11 | NOT MET | Comparator row pending. |
| AC-12 | NOT MET | Hard NIAH/MMLU gate pending. |
| AC-13 | MET | Unit regression suite remains green. |

Deferred item audit: `task-ac4-hwrun` remains active pending analyze work, not accepted as deferred. The doc update was declared queued in Round 11, but it is required before the hardware run and is tracked as a blocker for safe AC-4 execution.

Plan evolution audit: I rejected the tracker claim that `task-ac4-calibrate` is coding-complete. I updated the mutable tracker to reopen it with the two verified coding gaps above.

## Required Implementation Plan

1. Fix `_extract_mla_nope_prefix` to support arbitrary leading dimensions, then add a 3-D hook-output regression for both `kv_b_proj` and `q_b_proj`.

2. Refactor the production dataset path so Pile-val block construction happens after tokenizer load. The default no-`--dataset`, no-`--allow-synthetic` path must tokenize shuffled Pile rows, concatenate token ids, and run exactly `num_samples` fixed blocks of `block_size` tokens. Add a unit test that proves short documents are concatenated into fixed blocks.

3. Update `docs/advanced_features/double_sparsity_calibration.md` and the `calibrate.py` module header to the current AC-4 recipe: Method 1 Q/K noPE, Pile-val seed=42, 256x512, and synthetic only via `--allow-synthetic`.

4. Rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`. Only after steps 1-3 pass should `task-ac4-hwrun` generate `/models/dsv32-fp8-channel-mask.safetensors` on H200 and validate it with `load_channel_mask`.

5. Continue in dependency order after AC-4: AC-1 hardware population, AC-1b probe, AC-5 TP harness, AC-6 graph capture, AC-8 server/quality smoke, AC-12 full quality, then AC-9 through AC-11 stretch measurements.

NOT COMPLETE
