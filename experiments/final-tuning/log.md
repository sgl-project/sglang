# Final Tuning — Implementation Log

Branch parent: `main` (commit `63e0c8da2` Merge #2 refactor_recipe).
Created: 2026-04-29 (PST).

## Goal

Find `T*(M_global)` — the optimal per-expert-load promotion threshold (or equivalently `x*(M_global)`, count of experts promoted to BF16) that minimizes total heter-MoE layer latency at each global batch size, for Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8) on A100-SXM4-80GB.

## Decisions made (no user follow-up)

- **Branch parent:** `main` (not `refactor_recipe`).
- **Routing data source:** `/data/heter-moe/routing_stats/` does not exist on this machine. Falling back to **synthetic Zipf-distributed routing** as implemented in `scripts/heter_moe_collect_routing.py:generate_synthetic_routing` (alpha=1.1 power law, per-layer permutation, seed=42). This matches the natural shape of expert routing under real serving (LLMs routinely produce Zipf-shaped distributions over experts). This is documented per-task below; the side-branch sharegpt mc-sweep can be re-run later against real captures if/when they are collected.
- **Promotion policy:** hot-frequency-rank (top-x by routing frequency). Random shuffle baseline added at extreme M_global only.
- **Tolerance for Task 3 pass:** 10%.
- **M_global sweep granularity:** original plan called for 8 values; user requested ~3× → using **24 values** spaced log-and-linearly (decode regime densely, prefill regime densely).
- **Kernel pinning for Task 3 eval:** the BF16 path in Task 3 validation MUST use the autotuned tile from Task 2.prelim (`bf16_sparse_configs.json`), not the default JSON. This is enforced by wrapping the kernel call in `with override_config(tile_dict): outplace_fused_experts(...)`. The same lookup (`hierarchical_lookup` in `_utils.py`) is used in both Task 2.bench (for the table) and Task 3 (for prediction lookup AND validation kernel). Without this, the validation would silently fall back to the default `E=128,N=768` JSON which only goes up to M=4096.

## Step log

### 2026-04-29 — Task 1 (INT4 cold path) complete

- Sharded 9 n_cold values across GPUs 0–7. Wall: ~1 min.
- Initial grid `m_pe ∈ {8..128}` was too small for the M_global sweep range; extended to {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512}. Tried 768/1024 but Marlin OOM/illegal-memory at M_global=12288+; capped at 512 (M_global=8192, matches A100 chunked-prefill cap).
- Output: `results/int4_table.csv` (108 rows).

### 2026-04-29 — Task 2.prelim (BF16 sparse-active autotune) complete

- 5 n_active × 8 m_per_expert = 40 cells; reduced search space (480 configs, dropped GROUP_SIZE_M=1, num_stages={2,5}, BLOCK_SIZE_K=256 from the production 1920) to keep wall time tractable.
- Sharded round-robin across GPUs 0–7. Cold-JIT first cell per GPU ~10 min; subsequent cells ~30s–4 min (warm Triton disk cache).
- Original 25 cells (5×5 with bse ∈ {32..512}): ~15 min wall.
- Extended 15 cells (bse ∈ {1024, 2048, 4096}): ~12 min wall.
- 1 cell killed (n=64, bse=4096) — outside relevant operating regime (M_global=65k > 8192 prefill cap); falls back to `n64_bse2048` via hierarchical lookup.
- Output: `results/bf16_sparse_configs.json` (39 cells).

### 2026-04-29 — Task 2.bench (BF16 hot path) complete

- 8 n_hot × 39 m_per_expert = 312 rows. Sharded across GPUs 0–7. Wall: ~30s.
- Hierarchical-nearest tile lookup pulls from `bf16_sparse_configs.json`; pinned via `override_config(...)`.
- Output: `results/bf16_table.csv` (312 rows).

### 2026-04-29 — GPU clock state caveat (post-K=5 rerun)

Investigating a 21% drift in the M=4096 INT4 baseline (2.51 ms → 3.03 ms) between runs led to discovering the A100 application graphics clock is capped at **1140 MHz** instead of the default 1410 MHz boost clock (`nvidia-smi -q -d CLOCK` shows `Applications Clocks Setting: Active`). Memory clock unchanged at 1593 MHz.

- This was sysadmin-set; we don't have `nvidia-smi -ac` permissions to change it.
- Effect: Marlin INT4 (compute-bound) slows ~21%; BF16 fused MoE (memory-bound on A100 with bf16 weight loads) slows only ~2%.
- Consequence: the speedup column in `x_star_curve.md` is inflated by ~15–20% on rows where the INT4 baseline is compute-bound. Honest at-default-clock estimate of the heter-MoE win:
  - M=1024: ~1.15–1.20× (reported 1.24×)
  - M=3072: ~1.10–1.15× (reported 1.31×)
  - M=4096: **~1.05–1.10×** (reported 1.27× — matches the earlier 2.51/2.33=1.08× measurement at full clock)
- The general shape (wins start at M≥1024, peak in mid-prefill, taper at heavy prefill) IS robust to the clock state.

If the cluster gets restored to default clocks, re-running Task 3 measure-all-x will give the production-clock numbers. Otherwise the current report reflects the constrained-clock reality of this hardware.

### 2026-04-29 — Task 3 (compose + find x* + validate) complete

- 24 M_global values × 9 x candidates = 216 actual paired-kernel measurements (`fused_marlin_moe` + `outplace_fused_experts` with **pinned autotuned BF16 tile** via `override_config`).
- Sharded by M_global across GPUs 0–7; wall ~5 min.
- Final deliverable: `results/x_star_curve.csv` and `results/x_star_curve.md`.
- Speedups vs pure-INT4 baseline range 1.00× (decode regime, no promotion useful) to 1.36× (mid-prefill M=3584). At heavy prefill (M=8192-9216), x*=64 (promote top half) gives ~10% speedup.
- Prediction-vs-measurement agreement is poor at small M (microbench under uniform load over-estimates kernel-launch overhead at low M_global) but excellent at large M (M=8192: 0.8% agreement at predicted x*=56, true optimum is x*=64). For deployment we use the **measured** x*, not predicted.

