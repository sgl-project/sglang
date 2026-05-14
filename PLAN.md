# Next Claude Code Session Plan: Move Double Sparsity Win Left

## Summary

Continue on `dev/double-sparsity-v2`; do not restart. The branch already proves native DS can pass at `128K / conc=32 / tb=8192`, but the next priority is to move the win left to `128K / conc=16` while preserving NIAH quality.

Primary success target: `branch_ds_on` vs `branch_ds_off` at Llama-3.1-70B TP=8, `context_len=131072`, `concurrency=16`, real calibration, `TBT ratio <= 0.90`, and `NIAH(on) - NIAH(off) >= -0.02`. Prefer passing with `tb=2048` or `tb=4096`; accept `tb=8192` only if selector/kernel work makes it clearly win at conc=16.

Start the compacted session by reading:
- `benchmark/double_sparsity/HANDOFF_NATIVE.md`
- `benchmark/double_sparsity/SESSION_REPORT_2026-05-14.md`
- `benchmark/double_sparsity/repro_session/sweep_70b_128k_tbt_win/nsys/MANIFEST.md`
- `benchmark/double_sparsity/repro_session/sweep_70b_128k_tbt_win/nsys/compare_c32.txt`

## Key Changes

- Fix profiling/repro hygiene first:
  - Update stale docs/open-work text that still says nsys was not executed.
  - Fix `run_nsys_at_winning_point.sh` defaults to match the real winning point: `CONC=32`, `OUTPUT_LEN=64`, `TOKEN_BUDGET=8192`, `MAX_SELECTED=16384`.
  - Improve `compare_nsys.py` so it can print raw kernel names and identify hidden `torch.topk`/CUB/ATen kernels instead of only buckets like `void at`.
  - Treat nsys total GPU time as structural proof, not wall-clock attribution, because DS-on has lower TBT despite higher summed GPU time.

- Implement per-step request-token caching:
  - Use `SparseCoordinator.forward_begin(forward_batch)` for DS and call it from `ModelRunner.forward_decode` before `model.forward`.
  - Add a DS algorithm hook that performs `req_to_token[req_pool_indices]` index-select once per decode step into the existing native scratch buffer.
  - Change `try_native_sparse_decode` to reuse that cached indexed table for all layers.
  - Verify CUDA graph capture records one request-token gather per decode step, not one per layer.

- Add selector backend experimentation behind a default-off internal interface:
  - Add `--double-sparsity-selector-backend` with choices: `torch` default/current path, `flashinfer_topk_page_table`, `sgl_fast_topk_transform`, `jit_fused_selector`.
  - `flashinfer_topk_page_table`: flatten scores to `[bs * h_kv, max_ctx]`, use `req_to_token_indexed` as `src_page_table`, use repeated `seq_lens - 1` lengths and `row_to_batch`, then append sink/recent with a tiny kernel.
  - `sgl_fast_topk_transform`: adapt existing SGL top-k transform code if practical; currently it is optimized for `topk=2048`, so first test `tb=2048`.
  - `jit_fused_selector`: only build after the first two selector backends are measured; use the repo’s `add-jit-kernel` guidance and avoid static loops over `top_k`.

- Improve quality at lower budgets:
  - Create `/workspace/ds_retrieval_calib_prompts.txt` with NIAH/retrieval-shaped prompts and run `scripts/double_sparsity/calibrate.py --prompts-file ...`.
  - Benchmark `tb=2048`, `tb=4096`, and `tb=8192` with both wikitext calibration and retrieval-shaped calibration.
  - If `heavy_channels=32` cannot make `tb<=4096` pass quality, try `heavy_channels=64` and measure whether the lower `tb` still wins.

- Backend spike after selector/calibration:
  - Microbench FlashInfer `BlockSparseAttentionWrapper` only after selector work, using block size `(1, 1)` first to preserve token-level sparsity.
  - Do not wire FlashInfer block-sparse attention into decode unless microbench proves it beats current native sparse attention including metadata/plan overhead.
  - Use FlashInfer `top_k_page_table_transform` aggressively for selection because it directly matches the current top-k + physical-map bottleneck. Reference: [FlashInfer top_k_page_table_transform](https://docs.flashinfer.ai/generated/flashinfer.top_k_page_table_transform.html). FlashInfer block-sparse attention reference: [FlashInfer sparse API](https://docs.flashinfer.ai/api/sparse.html).

## Test Plan

- Unit and parity:
  - Run `pytest test/registered/unit/mem_cache/sparsity/ -q`.
  - Add CUDA selector parity tests comparing each selector backend against the current torch path on small shapes; compare selected sets, not ordering.
  - Skip FlashInfer tests when FlashInfer is unavailable unless that backend is explicitly selected, in which case fail loud.

- Synthetic performance:
  - Run `PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/profile_native_decode.py` for each selector backend and `tb` in `2048,4096,8192`.
  - Add per-phase timing for selector backends so score, top-k/transform, sink/recent append, and sparse attention are visible separately.

- H200 acceptance matrix:
  - Baseline: `branch_ds_off`, `128K`, `conc=16`, `output_len=512`, NIAH n=10+.
  - DS-on sweep: `tb=2048,4096,8192`, `selector_backend=torch,flashinfer_topk_page_table,sgl_fast_topk_transform` where supported.
  - Publish a result only when both gates pass: `TBT ratio <= 0.90` and `NIAH delta >= -0.02`.
  - Recheck current headline point `conc=32/tb=8192` after changes to ensure no regression.

## Assumptions

- Current branch is the base of work: `dev/double-sparsity-v2`.
- Keep DS disabled behavior unchanged.
- Default runtime behavior remains the current torch selector unless a new selector backend is explicitly requested.
- FlashInfer may exist only on the H200 benchmarking environment; local absence should not block non-FlashInfer work.
- Custom selector kernels should be JIT first unless they need CUTLASS/AOT integration.
- Update `HANDOFF_NATIVE.md` and `SESSION_REPORT_2026-05-14.md` at the end with exact benchmark JSON paths, nsys paths, selector backend results, and the final recommendation.
