# Round 12 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 12 did close the three specific Round 11 findings: `_extract_mla_nope_prefix` now flattens leading dimensions, Pile-val block construction now tokenizes and concatenates into fixed blocks, and the stale K-only/NIAH operator docs were updated. I reran the claimed suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
187 passed, 24 warnings in 11.42s
```

AC-4 is still not coding-complete. The real DeepSeek-V3.2 config shape exposes a calibration-path bug that the new tests miss, and the H200 mask-generation recipe still conflicts with the FP8 serving contract.

## Mainline Gaps

1. **AC-4 real DeepSeek-V3.2 calibration still skips every Method-1 Q hook.**

   Evidence:
   - The actual local V3.2 config has `hidden_size=7168`, `num_attention_heads=128`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, and `v_head_dim=128`, but no `head_dim` field (`/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/config.json:12`, `:27`, `:33-34`, `:64`).
   - `calibrate.py` derives `head_dim` from `hidden_size // num_heads` when `config.head_dim` is absent, then derives Q RoPE width as `head_dim - qk_nope_head_dim` (`python/sglang/srt/layers/attention/double_sparsity/calibrate.py:279-285`). For V3.2 this becomes `56 - 128 = -72`.
   - Because the derived Q RoPE width is negative, `full_mla_q_width` is `None` (`calibrate.py:320-322`). A real `q_b_proj` output width is `128 * (128 + 64) = 24576`, so the Q hook falls through the width check and returns without filling `_q_buf` (`calibrate.py:416-428`).
   - I verified this with a fake model using the same config shape: `_collect_channel_importance` logs a Q projection width mismatch and raises `RuntimeError: Calibration hooks did not fire`.

   Consequence: `task-ac4-hwrun` would fail on the real H200 V3.2 calibration path even though the Round 12 unit suite passes. This directly contradicts the plan requirement that V3.2 MLA is auto-detected via `qk_nope_head_dim=128` and needs no `--model-arch` flag.

   Required fix:
   - Derive `qk_rope_head_dim` directly from `config.qk_rope_head_dim` when present.
   - Only fall back to `head_dim - qk_nope_head_dim` for configs that lack an explicit `qk_rope_head_dim`.
   - Validate the resulting MLA Q width before registering hooks, with an error message that names the derived dimensions.
   - Add a regression in `TestCalibrateMethod1` where the fake config has `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, and no `head_dim`; make `hidden_size // num_heads` intentionally different from `qk_nope + qk_rope`. The test must prove Method 1 Q/K accumulation succeeds for `q_b_proj` width `H * (nope + rope)`.

2. **The original lower-bound and hard gates remain incomplete.**

   Pending from the original plan:
   - AC-1: H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b: chunked-prefill probe.
   - AC-4: fix the real-config coding gap above, then generate `/models/dsv32-fp8-channel-mask.safetensors` on H200 and validate it with `load_channel_mask`.
   - AC-5: TP=2 multiprocess all-reduce harness.
   - AC-6: CUDA graph coding plus H200 replay.
   - AC-8: 8xH200 `bench_serving` and lightweight quality smoke.
   - AC-12: hard NIAH/MMLU quality gate.
   - AC-9 through AC-11: stretch baseline/radix/comparator work.

## Blocking Side Issues

1. **The AC-4 operator recipe writes a `bfloat16` mask that Option-B FP8 serving will reject.**

   Evidence:
   - Round 12's module header and calibration doc recommend `--dtype bfloat16` for `/models/dsv32-fp8-channel-mask.safetensors` (`calibrate.py:17-20`, `docs/advanced_features/double_sparsity_calibration.md:31-34`).
   - The channel-mask schema defines `dtype` as the server `kv_cache_dtype` (`channel_mask.py:21-22`).
   - The DS launcher defaults `KV_CACHE_DTYPE=fp8_e4m3` (`development/serve_double_sparsity.sh:21`, `:48`).
   - Startup validation rejects a mask whose metadata dtype differs from `--kv-cache-dtype` (`validator.py:197-200`).

   Consequence: following the Round 12 H200 recipe produces a content-valid mask that cannot boot the locked Option-B DS server. This blocks AC-8 and makes the AC-4 hardware artifact unusable for the stated FP8 MVP.

   Required fix: align the calibration artifact metadata with the intended serving KV dtype before hardware generation. For the Option-B path, the production artifact should be tagged `fp8_e4m3`; if the forward pass must load weights in bf16 for calibration stability, separate model-load dtype from mask/runtime dtype instead of overloading `--dtype`.

## Queued Side Issues

1. Before AC-6, thread `req_to_token` through `capture_decode_step`; otherwise graph capture validates the wrong selector domain.
2. Before AC-8, fix DS observability page-named fields and page-count sparsity math.
3. Clean stale bind/runtime sizing comments and token-label lifetime documentation when touching those modules.

## Goal Alignment Summary

```text
ACs: 7/15 addressed (5 met, 2 partial) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC:

| AC | Status | Evidence / Gap |
|----|--------|----------------|
| AC-0 | MET | Completed/verified in tracker Round 2. |
| AC-1 | PARTIAL | Local hook coverage verified; H200 population and AC-8 selector-read smoke pending. |
| AC-1b | NOT MET | Chunked-prefill probe has not run. |
| AC-2 | MET | Completed/verified in tracker Round 7. |
| AC-3 | MET | Completed/verified in tracker Round 6. |
| AC-4 | PARTIAL | Round 12 fixed 3-D outputs, token-block Pile-val, and stale docs, but real V3.2 config still breaks Q hook width derivation; H200 mask run pending. |
| AC-5 | NOT MET | Multiprocess TP integration test still absent. |
| AC-6 | NOT MET | Graph capture coding and H200 replay pending. |
| AC-7 | MET | Completed/verified in tracker Round 9. |
| AC-8 | NOT MET | No server benchmark or lightweight quality smoke. |
| AC-9 | NOT MET | No DSA baseline JSON. |
| AC-10 | NOT MET | Radix fixture and FP8 cold/warm proof pending. |
| AC-11 | NOT MET | Comparator row pending. |
| AC-12 | NOT MET | Hard NIAH/MMLU gate pending. |
| AC-13 | MET | Unit regression suite remains green. |

Deferred item audit: no tracker item is explicitly deferred. `task-ac4-hwrun` remains active analyze work, but I marked it blocked because the real-config coding path is not safe yet.

Plan evolution audit: Round 12's claim that `task-ac4-calibrate` is coding-complete is rejected. I updated the mutable tracker to reopen `task-ac4-calibrate`, block `task-ac4-hwrun`, remove the now-fixed stale K-only/NIAH docs queued issue, and add the real-config and dtype-contract blockers.

## Required Implementation Plan

1. Fix `calibrate.py` dimension derivation: use `config.qk_rope_head_dim` for MLA when present, compute `full_mla_q_width = num_heads * (qk_nope_head_dim + qk_rope_head_dim)`, and fail clearly if any derived dimension is non-positive or inconsistent with hook output widths.

2. Add the no-`head_dim` V3.2-shaped regression described above, plus a small assertion that the fake `q_b_proj` hook actually contributes Method-1 Q/K importance.

3. Resolve the dtype contract before hardware generation. For the locked FP8 serving path, update the operator recipe and generated metadata to `fp8_e4m3`, or add a separate model-load dtype argument if bf16 forward compute is required.

4. Rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`. Only after steps 1-3 pass should `task-ac4-hwrun` generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200.

NOT COMPLETE
