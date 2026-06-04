# Task: UT + micro-benchmark for `_moe_lora_expand_add_kernel` (down-proj GEMM)

**[try bs 64 first, rank 16 only] [P0]**

## Goal
Mirror `bench_shrink_splitk.py` (which targets the LoRA-A shrink) with an equivalent
self-contained testbed for the LoRA-B **expand-add** kernel
(`_moe_lora_expand_add_kernel` in `python/sglang/srt/lora/trtllm_moe/specialized_expand.py`),
scoped to the **down-proj** GEMM. Confirm correctness against a torch reference, then
ship a registered UT PR with the script attached.

## Scope / kernel contract (verified from source)
- Down-proj expand-add caller (`lora_dispatch.py:207`, `moe_overlap.py`): uses
  `mul_routed_weight=True`, `fuse_sum_all_reduce=True`, `use_direct_expand_add=(rank<=64)`.
- Kernel signature: `a` = shrink intermediate `[num_tokens*top_k, R]`,
  `b` = `lora_b_virtual [num_virtual_experts, N, R]`, `c` = output `[num_tokens, N]`.
- `FUSE_SUM_ALL_REDUCE`: per-(token,expert) delta atomic-added into `c[token//top_k]`
  → output MUST be zeroed each call. `MUL_ROUTED_WEIGHT`: scale by `topk_weights`.
- `_invoke_moe_lora_expand_add` forces `BLOCK_SIZE_N = 128` when `N % 128 == 0`
  (N=2048 → 128) and hardcodes `num_stages=1`. Tunable knobs: `BLOCK_SIZE_M`,
  `GROUP_SIZE_M`, `num_warps`.
- qwen3.5-35b local-EP scope: tp=4/ep=4 → 64 local experts, N (down hidden) = 2048,
  rank = 16, top_k = 8.

## Correctness contract guarded (same class of bug as the shrink f2adddd regression)
The launcher tiles `expert_ids` with `config["BLOCK_SIZE_M"]`; routing buffers must be
aligned with the SAME block. Correctness mode sweeps block_m {16,32,64} × bs {16,64},
building routing per block_m, so a hardcoded mismatch overruns `expert_ids` → IMA.

## Plan / steps
1. [done] Sync `lora-opti-nvfp4` w/ `jybsuper/nvfp4-lora` (already up to date @ ac0fa6d3ee).
2. [done] Worktree `sglang-lora-moe-expand-down-bench`, branch `lora-moe-expand-down-bench`.
3. [in progress] Write standalone `bench_expand_add_down.py` + registered UT.
4. Launch GPU node (id `yushengsu-<date>-<time>`), run correctness + bench.
5. Open UT PR with the script and journal attached.

## Runs

### Node
- Pod `sglang-lora-expand-ut-yushengsu-20260602-174942` (id `yushengsu-20260602-174942`),
  1× **NVIDIA GB200**, image `lmsysorg/sglang:dev-cu13`. Overlaid branch
  `lora-moe-expand-down-bench` python tree onto the editable install (no model weights
  needed — pure triton kernel UT). Released after the runs.

### Correctness (`--mode correctness`) — PASS
Reference fp32 vs kernel (per-expert bf16 atomic-add accumulation), tol=5e-2 abs:
```
PASS bs=16  block_m=16 max_abs_err=3.00e-03 rel=8.6e-03
PASS bs=16  block_m=32 max_abs_err=2.46e-03 rel=7.0e-03
PASS bs=16  block_m=64 max_abs_err=2.57e-03 rel=7.4e-03
PASS bs=64  block_m=16 max_abs_err=2.46e-03 rel=6.7e-03
PASS bs=64  block_m=32 max_abs_err=3.01e-03 rel=8.2e-03
PASS bs=64  block_m=64 max_abs_err=2.38e-03 rel=6.5e-03
```
Registered UT `test/registered/lora/test_moe_lora_expand_add.py`: **2 passed, 6 subtests passed** (11.7s).

### Bench / sweep (bs=64, rank=16, N=2048, top_k=8, GB200, amortized device time)
Production default (block_m=64): **11.83 µs**.
Sweep over block_m × group_m × num_warps (BLOCK_SIZE_N forced to 128, num_stages=1):
```
block_m=16: 6.38–7.42 µs   <-- best
block_m=32: 6.94–7.69 µs
block_m=64: 10.98–11.78 µs
BEST: 6.38 µs  block_m=16 group_m=8 warps=8
```
**Finding (P0):** the down-proj expand at bs=64/rank=16 is ~1.85× faster with
`BLOCK_SIZE_M=16` than the current `_get_stage_config` fallback default of 64.
group_m / num_warps are second-order at block_m=16 (6.38–6.49 µs across 1/4/8 × 2/4/8).

## Next steps (beyond this UT PR)
- This PR ships the UT + bench only (no kernel/config change). The block_m=16 win is a
  follow-up tuning change to `_get_stage_config` / a generated config, to be validated by
  the full regression + perf-benchmark flow (Qwen + Kimi) per skill.md.
