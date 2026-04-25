# HISA TODO

Living list of outstanding work on the sglang HISA integration. Maintain in-place; don't treat memory entries as source of truth.

Legend: `[ ]` pending, `[/]` in progress, `[x]` done, `[?]` needs investigation.

---

## P0 — Kernel-level perf (attack the sparse_paged 80% hotspot)

**Status 2026-04-24 update:** The 80% hotspot claim was pre-triton-port. After porting, the triton `sparse_paged_mqa_triton` is 5-20× faster than the original tilelang baseline. Rechecked kernel bench numbers on H200 (kernels.py, default config):
- B=1 ctx=65K: 0.011 ms (tilelang was 0.214 ms → 19×)
- B=8 ctx=65K: 0.012 ms (17×)
- B=32 ctx=65K: 0.022 ms (10×)
- B=64 ctx=65K: 0.036 ms (6×)

At B=64 we're at ~73% of HBM peak bandwidth already. At B=1-8 we're launch/occupancy-limited, not bandwidth-limited.

Tried optimizations (2026-04-24):
1. `[~]` **Pipeline stages (`num_stages=2/3`)** — NO WIN. Both sparse_paged and v3 kernels are 1-tile-per-CTA with no inner loop; nothing to overlap. `num_stages=3 + num_warps=4` regressed B=1 by 40%. Reverted.
2. `[~]` **Q reuse via chunked topk (A1)** — NO WIN. Tested CHUNK ∈ {1, 2, 4, 8, 16} with `tl.range(num_stages=2)`. Best case: CHUNK=2 at B=64 saved 5%; all other configs equal-or-worse. Root cause: at small B the kernel is instruction-latency-bound (Q bandwidth isn't the bottleneck); at B=64 we're near HBM peak anyway. Reverted.
3. `[ ]` **`nsys profile` on one real workload** to determine memory- vs compute-bound. Numbers we need: HBM throughput (GB/s vs H200's 4.8 TB/s peak), SM occupancy, L2 hit rate. Informs all downstream moves. Still useful to quantify.
4. `[ ]` **Warp specialization (producer/consumer warps)** — FlashMLA / DeepGEMM (`sm90_fp8_paged_mqa_logits`) use kNumTMAThreads + kNumMathThreads pattern w/ mbarrier sync. Potentially 30-50% win, but requires switching back to tilelang (triton doesn't expose `cp.async.bulk`) and 1-2 weeks of work. Hold until/unless sparse_paged shows up as the hotspot in a real profile.
5. `[ ]` **Persistent kernel (A2)** — expected win dominated by launch overhead savings. Under CUDA graph replay (production decode), launch overhead is ~0. Non-graph (prefill, warmup): could save ~60μs per call. Low ROI.
6. `[ ]` **Cluster launch for shared-Q broadcast** — H100+ cluster feature; max cluster=16. Triton support is partial/experimental. Save for when the simpler moves have been exhausted.

## P0 — Correctness (should fix before declaring Phase-2 "done")

- [/] **Prefix cache + pool_k_pages stale-data bug** — current mitigation: `_store_index_k_cache` sets `prev_seq_lens=0` on every extend, forcing `update_pool` to re-pool `[0, new_complete)`. Works (samsum stable at ~0.40 across runs) but redundant for same-request chunked prefill.
    - Proper fix: maintain a per-request "already-pooled-up-to" watermark and pool only `[watermark, new_complete)`. When `alloc_for_extend` detects a cache-hit (prefix_len > 0), set watermark=0 for the new request; otherwise carry it forward across chunks.
    - Impact of current mitigation: ~3-5× redundant mean-pool work during chunked prefill (≤ 150 ms added to TTFT for 65K input). Acceptable but ugly.
    - Test: `bash /data/sglang_scripts/eval_samsum.sh` 3× back-to-back, ROUGE should stay > 0.40 (or at least not degrade monotonically).

- [ ] **samsum ROUGE first-run floor** — first run 0.39 ± 0.02, subsequent 0.41 ± 0.04. Within ±1σ of the 0.40 threshold. Dig in only if (a) user wants a hard guarantee > 0.40 or (b) we see regressions in other tasks.

## P0 — Feature gaps (blocks shipping)

- [ ] **#9 CP path support for HisaIndexer** — `_get_topk_ragged_with_cp` currently raises NotImplementedError. vLLM's hisa prefill is CP-compatible in principle (K all-gather + Q split happen outside the kernel, same as deep_gemm.fp8_mqa_logits) — needs a CP-enabled smoke test.

- [ ] **#20 target_verify / draft_extend PAGED paths** — `_get_topk_paged` bails with NotImplementedError when `next_n > 1`. Speculative decoding workflows need this.

## P1 — Perf follow-ups (Phase-2 not yet paying off at B=10)

Each of these attacks a specific overhead identified in the stages microbench. Pick any one to restart the "is v3 actually faster than v1?" investigation. Current e2e: v3 ≈ v1 ≈ 62 ms TPOT at B=10 ctx=65K; v3 microbench is 1.15× per-call faster so we know the delta is swallowed by bookkeeping.

- [~] **Batch `update_pool × 61 layers` into one kernel launch.** SKIPPED 2026-04-24. Measured: non-graph cost 0.81 ms/step (61 × 13 μs CPU launch), but under CUDA graph replay (production decode) only 0.18 ms/step (61 × 2.9 μs GPU). Max savings from batching under graph ≈ 0.17 ms/step (0.3% of 58 ms TPOT). Architectural churn (defer hook + per-layer tensor stacking) not justified.

- [x] **Replace `get_pool_page_tables`'s fancy indexing with `index_select(out=persistent_scratch)`.** DONE 2026-04-24. Added `_scratch_pool_page_tables` in `HisaNSATokenToKVPool.__init__`; `get_pool_page_tables` now writes via `index_select(out=scratch[:B])`. Microbench (61 layers, B=10): 0.483 → 0.420 ms/step, 13% win (~63 μs). Smaller than the original 0.6 ms/step estimate — PyTorch's caching allocator is fast for small re-used allocs.

- [ ] **Bench at B=32 and B=64** to validate that v3's 1.5-2× microbench advantage materializes in e2e. Current bench (B=10) is right at v3's cross-over point — see `project_hisa_bench_v1_v3.md` memory.

## P1 — Integration polish

- [ ] **#12 Support `SGLANG_NSA_FUSE_TOPK=1` (fused topk output)** — HisaIndexer currently asserts it's off. The fused path would skip the Python `fast_topk_v2 + coord_transform` bridge. Would need `hierarchy_*_mqa_logits` to accept a fused-topk output target.

- [ ] **Scatter pool_k_pages directly from prefill's ragged mean-pool** — Right now prefill's `fp8_native_hierarchy_mqa_logits` computes its own mean-pool internally (ragged) then `_store_index_k_cache` later calls `update_pool_for_completed_blocks` for the same blocks. Redundant work. Fusing would save ~5-15 ms on long-context TTFT.

- [ ] **Pre-allocate `pool_page_tables` view for the gather kernel** — `pool_page_tables[:q_offset].contiguous()` in `_get_topk_paged` triggers an alloc (from graph pool). Trivially fixable.

## P2 — Cleanup / polish

- [ ] **Remove the `SGLANG_HISA_VERIFY` debug path** in `hierarchy_indexer.py` (env-gated v3-vs-v1 logits + topk-IoU compare). Components to delete:
    - `import os` at top of file (only this verify path uses it)
    - the `if use_pool_cache and os.environ.get("SGLANG_HISA_VERIFY") == "1" and not get_is_capture_mode():` block inside `_get_topk_paged`
    - the entire `_hisa_verify_vs_v1` method
    - It was kept in for a specific verification round. Once we trust v3's numerical parity (samsum stable + byte-equal unit tests), this can go. Zero overhead when unset, so no rush.

- [ ] Rename legacy `max_pool_blocks_per_req` mentions that leaked through after v2→v3. Grep: `grep -rn "max_pool_blocks_per_req" python/sglang/srt/layers/attention/nsa/hisa/`.

- [ ] Delete `benchmark/benchmark_indexer.py` v2b dead-code comments (already removed the fns, but some column/label names in `bench_decode` still mention v2b).

- [ ] Add a flag `SGLANG_HISA_DISABLE_POOL_CACHE=1` (already implemented in `model_runner_kv_cache_mixin.py`) to the user-facing serve scripts or docs, so A/B benching is reproducible.

- [ ] Document the pool-K page layout + allocator in the hisa dir's docstring or a short README so the next reader doesn't have to piece it together.

## How to run each version

All commands below assume you're in the repo root. Before switching, kill any running server:
```bash
pkill -9 -f "sglang.launch_server"; sleep 3
```

### **v4** — default (paged pool cache + triton hotspots)
```bash
bash /data/sglang_scripts/serve_hisa.sh
```
The script sets `SGLANG_NSA_FUSE_TOPK=0` and passes `use_hisa=true, hisa_k_block_size=128, hisa_block_topk=64` via `--json-model-override-args`. Serves as `deepseek-v32-hisa` on `127.0.0.1:30000`.

### **v3** — paged pool cache + tilelang (triton disabled)
```bash
SGLANG_HISA_DISABLE_TRITON=1 bash /data/sglang_scripts/serve_hisa.sh
```
Same pool-K-pages layout as v4, but keeps tilelang for `sparse_paged_mqa` + `batch_decode_pool_mqa_v3`. Useful for A/B comparing just the triton swap.

### **v1** — hisa indexer, NO pool cache (pre-Phase-2)
```bash
bash /data/sglang_scripts/serve_hisa_nocache.sh
```
The script sets `SGLANG_HISA_DISABLE_POOL_CACHE=1` which makes `model_runner_kv_cache_mixin.py` skip `HisaNSATokenToKVPool` and create a plain `NSATokenToKVPool`. The indexer's `isinstance` check flips to False → falls through to the v1 fresh mean-pool path. Served as `deepseek-v32-hisa-v1`.

### **Baseline** — stock sglang, no hisa
```bash
bash /data/sglang_scripts/serve_baseline.sh
```
No `use_hisa` override; plain DeepSeek-V3.2 with deep_gemm's `fp8_paged_mqa_logits`. Served as `deepseek-v32`.

### With verify overlay (any of the hisa variants)
```bash
SGLANG_HISA_VERIFY=1 bash /data/sglang_scripts/serve_hisa.sh
```
At every indexer call (outside cuda graph capture), also runs v1 fresh as reference and prints rows where:
  - logits abs diff > `SGLANG_HISA_VERIFY_LOGITS_ABS` (default 0.01)
  - logits rel diff > `SGLANG_HISA_VERIFY_LOGITS_REL` (default 0.05)
  - topk IoU < `SGLANG_HISA_VERIFY_IOU` (default 0.95)

Adds ~30-50% TTFT overhead. Remove for production.

---

## Env var reference

| Var | Effect | Production? |
|---|---|:-:|
| *(none)* | v4 default — paged pool cache + triton hotspots | ✅ |
| `SGLANG_HISA_DISABLE_TRITON=1` | v3 — same pool layout, tilelang kernels | A/B |
| `SGLANG_HISA_DISABLE_POOL_CACHE=1` | v1 — no cache, fresh mean-pool | A/B |
| `SGLANG_HISA_VERIFY=1` | Inline v1 reference compare, prints divergence | Debug |
| `SGLANG_HISA_VERIFY_LOGITS_ABS=<f>` | abs diff threshold for verify (default 0.01) | Debug |
| `SGLANG_HISA_VERIFY_LOGITS_REL=<f>` | rel diff threshold for verify (default 0.05) | Debug |
| `SGLANG_HISA_VERIFY_IOU=<f>` | topk IoU threshold for verify (default 0.95) | Debug |
| `SGLANG_NSA_FUSE_TOPK=0` | required by HisaIndexer (asserts this at init) | ✅ (set by serve_hisa.sh) |

Cannot mix `DISABLE_TRITON=1` with `DISABLE_POOL_CACHE=1` meaningfully — the latter short-circuits by choosing a non-hisa pool, making the first irrelevant.

## Bench / eval

After any server is up:
```bash
# Throughput + TTFT/TPOT/ITL — ~90s
bash /data/sglang_scripts/bench_serving.sh

# Samsum ROUGE — ~25s, auto-reads `deepseek-v32-hisa` by default
bash /data/sglang_scripts/eval_samsum.sh

# Samsum against other models
bash /data/sglang_scripts/eval_samsum.sh deepseek-v32          # baseline
bash /data/sglang_scripts/eval_samsum.sh deepseek-v32-hisa-v1  # v1
```

## Code layout

- `python/sglang/srt/layers/attention/nsa/hisa/` — tilelang kernels + pool_k_cache + hierarchy_indexer
- `python/sglang/srt/layers/attention/nsa/hisa_triton/` — triton ports: `kernels.py`, `orchestrator.py` (v4), `benchmark.py`, `test_precision.py`

## Bench history

- In memory `project_hisa_bench_v1_v3.md` — baseline vs v1 vs v3 numbers at 2026-04-22, H200×8, 10×65K.
- v4 e2e bench result (2026-04-23, same bench): TPOT median 58.6 ms, ITL median 19.3 ms — v4 closes the gap to baseline to within 5-6%.
