# Round 10 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 10 moved AC-4 in the right direction by replacing the old K-only accumulator with paired Q/K hook machinery, and the claimed local suite is green:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
183 passed, 24 warnings in 11.51s
```

AC-4 is not closed. The implementation still cannot generate a trustworthy V3.2 channel mask because it extracts MLA Q/K noPE channels with the wrong layout, and it does not implement the required Pile-val seed=42, 256x512 calibration recipe.

## Mainline Gaps

1. **AC-4 calibration still flat-slices MLA projections before reshaping, so Method 1 is computed over V/RoPE columns for later heads.**

   Evidence:
   - `python/sglang/srt/layers/attention/double_sparsity/calibrate.py:296-306` handles `kv_b_proj` by slicing `tensor[..., :prefix]` before `reshape(-1, num_heads, k_head_dim)`.
   - `python/sglang/srt/layers/attention/double_sparsity/calibrate.py:320-331` does the same for `q_b_proj`: it slices the flat first `H * nope_dim` columns before reshaping.
   - The production MLA paths reshape first, then slice per head: `forward_mha.py:292-297` reshapes `kv_b_proj` output to `[-1, H, qk_nope_head_dim + v_head_dim]` before taking `[..., :qk_nope_head_dim]`; `forward_mla.py:268-303` reshapes `q_b_proj` output to `[-1, H, qk_head_dim]` before splitting noPE/RoPE.
   - The live label writer already fixed this exact K-side bug in Round 3: `dsa_backend.py:1431-1436` reshapes `kv_proj_out` as `[T, H_local, nope_dim + v_head_dim]` before slicing K-noPE.
   - A two-head sentinel reproducer gives the failure shape: flat `[K0,V0,K1,V1]`, current `[:H*D].reshape(H,D)` yields head 1 = `V0`, not `K1`.

   Consequence: the generated mask can be content-hash-valid while ranking channels from the wrong tensor columns. This directly blocks AC-4 and then AC-6/AC-8/AC-12, because downstream quality depends on a valid channel mask.

   Required fix:
   - Add a small extraction helper in `calibrate.py` for MLA outputs: reshape to `[-1, num_heads, prefix_dim + suffix_dim]` first, then return `[..., :prefix_dim].contiguous()`.
   - Use that helper for K with `suffix_dim=v_head_dim`.
   - Use it for Q with `suffix_dim=qk_rope_head_dim`; derive `qk_rope_head_dim` from config and fail clearly if `q_b_proj` has an unexpected width.
   - Preserve the standard-attention path as a direct reshape only when `qk_nope_head_dim == 0`.
   - Add two sentinel regressions to `TestCalibrateMethod1`: one for K `[K_nope | V]` and one for Q `[Q_nope | Q_rope]`, with at least two heads and sentinel values in V/RoPE columns. The tests must fail under the current flat-slice implementation.

2. **AC-4 still lacks the required Pile-val seed=42, 256x512 recipe.**

   Evidence:
   - The refined plan requires Pile validation, seed=42, 256 x 512 tokens for AC-4 (`development/loop4/refined_plan_v1.md:73`, `:293`).
   - The current script defaults to NIAH-shaped synthetic prompts (`calibrate.py:8-11`, `:382-385`) with parser defaults `--num-samples=64` and `--ctx-len=4096` (`calibrate.py:461-465`).
   - `_read_corpus_file` only reads newline prompts and wraps lines; it does not shuffle a Pile validation split, tokenize, concatenate, or produce fixed 512-token blocks (`calibrate.py:77-100`).
   - There is no `datasets.load_dataset` path or seed argument in `calibrate.py`.

   Consequence: even after the Q/K formula is fixed, the hardware run would not match the reference calibration contract and would not satisfy AC-4.

   Required fix:
   - Make the non-synthetic, no-`--dataset` production path load Pile validation via `datasets.load_dataset("mit-han-lab/pile-val-backup", split="validation")`, shuffle with `seed=42`, tokenize, skip empty examples, concatenate tokens, and produce exactly 256 blocks of 512 tokens.
   - Keep custom `--dataset` support as an explicit override, but add metadata recording `dataset_source`, `seed`, `num_samples`, and `block_size`.
   - Keep synthetic statistics behind `--allow-synthetic`; do not let the production V3.2 path silently use NIAH prompts.
   - Update parser defaults/help and `docs/advanced_features/double_sparsity_calibration.md` before `task-ac4-hwrun`, because the doc still advertises K-only L2 and NIAH-shaped 1024x4096 calibration.

3. **The original lower-bound and hard gates remain incomplete.**

   Pending from the plan:
   - AC-1: H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b: chunked-prefill probe.
   - AC-4: fix calibration coding issues above, then run H200 mask generation and `load_channel_mask` validation.
   - AC-5: TP=2 multiprocess all-reduce harness.
   - AC-6: CUDA graph coding plus H200 replay.
   - AC-8: 8xH200 `bench_serving` and lightweight quality smoke.
   - AC-12: hard NIAH/MMLU quality gate.
   - AC-9 through AC-11: baseline/radix/comparator stretch work.

## Blocking Side Issues

None outside the mainline AC-4 gaps above. The calibration extraction and dataset recipe issues are mainline gaps, not side issues.

## Queued Side Issues

1. Before AC-6, thread `req_to_token` through `capture_decode_step`; otherwise graph capture validates the wrong selector domain.
2. Before AC-8, fix DS observability page-named fields and page-count sparsity math.
3. Before `task-ac4-hwrun`, update `docs/advanced_features/double_sparsity_calibration.md` to Method 1 + Pile-val seed=42, 256x512.
4. Clean stale bind/runtime sizing comments and token-label lifetime documentation when touching those modules.

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
| AC-4 | PARTIAL | Q/K hook structure added, but MLA extraction and Pile recipe are wrong/missing; H200 mask run pending. |
| AC-5 | NOT MET | Multiprocess TP integration test still absent. |
| AC-6 | NOT MET | Graph capture coding and H200 replay pending. |
| AC-7 | MET | Completed/verified in tracker Round 9. |
| AC-8 | NOT MET | No server benchmark or lightweight quality smoke. |
| AC-9 | NOT MET | No DSA baseline JSON. |
| AC-10 | NOT MET | Radix fixture and FP8 cold/warm proof pending. |
| AC-11 | NOT MET | Comparator row pending. |
| AC-12 | NOT MET | Hard NIAH/MMLU gate pending. |
| AC-13 | MET | Unit regression suite remains green. |

Deferred item audit: Claude's summary treated `task-ac4-hwrun` as deferred because hardware is not available locally. That is not an accepted deferral under the original plan; it is still active pending analyze work and blocks AC-4 closure.

Plan evolution audit: the Round 10 tracker entry claimed `task-ac4-calibrate` was complete. I corrected the mutable tracker to reopen `task-ac4-calibrate`, keep `task-ac4-hwrun` active, remove AC-4 from Completed/Verified, and add the stale calibration doc as a queued issue.

## Required Implementation Plan

1. Fix `calibrate.py` extraction first. Implement a helper that reshapes MLA projections per head before slicing noPE channels. K uses `v_head_dim`; Q uses `qk_rope_head_dim`. Add K/V and Q/RoPE sentinel regressions and rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`.

2. Implement the production calibration dataset path. The default real path must be Pile validation, `seed=42`, exactly 256 blocks of 512 tokens. Keep synthetic-only behavior behind `--allow-synthetic`, and record dataset/seed/block metadata in the output file.

3. Update `docs/advanced_features/double_sparsity_calibration.md` so the operator-facing H200 recipe matches AC-4: Method 1, Pile-val seed=42, 256x512, 128-d noPE channel axis from config, and no accepted `task-ac4-hwrun` deferral.

4. Run the H200 calibration command from the refined plan only after steps 1-3 are fixed. Validate `/models/dsv32-fp8-channel-mask.safetensors` with `load_channel_mask` and record the content hash, exact command, commit SHA, and result path.

5. Continue in dependency order: AC-1 hardware population, AC-1b probe, AC-5 TP harness, AC-6 graph capture, AC-8 server/quality smoke, AC-12 full quality, then AC-9 through AC-11 stretch measurements.

NOT COMPLETE
