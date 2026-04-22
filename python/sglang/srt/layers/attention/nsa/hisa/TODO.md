# HISA TODO

Living list of outstanding work on the sglang HISA integration. Maintain in-place; don't treat memory entries as source of truth.

Legend: `[ ]` pending, `[/]` in progress, `[x]` done, `[?]` needs investigation.

---

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

- [ ] **Batch `update_pool × 61 layers` into one kernel launch.** Add a `layer_stride` dim to the kernel grid → 61 launches collapse to 1. Saves ~0.5 ms/step of launch overhead.

- [ ] **Replace `get_pool_page_tables`'s fancy indexing with `index_select(out=persistent_scratch)`.** Current per-layer `req_to_pool_page[req_pool_indices.long(), :]` triggers a gather kernel launch + allocation (from the graph pool) each layer × 61 layers = ~0.6 ms/step. Persistent `[max_running_req, max_pool_pages_per_req]` scratch + `index_select` in-place avoids the alloc.

- [ ] **Bench at B=32 and B=64** to validate that v3's 1.5-2× microbench advantage materializes in e2e. Current bench (B=10) is right at v3's cross-over point — see `project_hisa_bench_v1_v3.md` memory.

## P1 — Integration polish

- [ ] **#12 Support `SGLANG_NSA_FUSE_TOPK=1` (fused topk output)** — HisaIndexer currently asserts it's off. The fused path would skip the Python `fast_topk_v2 + coord_transform` bridge. Would need `hierarchy_*_mqa_logits` to accept a fused-topk output target.

- [ ] **Scatter pool_k_pages directly from prefill's ragged mean-pool** — Right now prefill's `fp8_native_hierarchy_mqa_logits` computes its own mean-pool internally (ragged) then `_store_index_k_cache` later calls `update_pool_for_completed_blocks` for the same blocks. Redundant work. Fusing would save ~5-15 ms on long-context TTFT.

- [ ] **Pre-allocate `pool_page_tables` view for the gather kernel** — `pool_page_tables[:q_offset].contiguous()` in `_get_topk_paged` triggers an alloc (from graph pool). Trivially fixable.

## P2 — Cleanup / polish

- [ ] Rename legacy `max_pool_blocks_per_req` mentions that leaked through after v2→v3. Grep: `grep -rn "max_pool_blocks_per_req" python/sglang/srt/layers/attention/nsa/hisa/`.

- [ ] Delete `benchmark/benchmark_indexer.py` v2b dead-code comments (already removed the fns, but some column/label names in `bench_decode` still mention v2b).

- [ ] Add a flag `SGLANG_HISA_DISABLE_POOL_CACHE=1` (already implemented in `model_runner_kv_cache_mixin.py`) to the user-facing serve scripts or docs, so A/B benching is reproducible.

- [ ] Document the pool-K page layout + allocator in the hisa dir's docstring or a short README so the next reader doesn't have to piece it together.

## Notes / reference

- Current default code path: **v3** (paged pool_k_pages, ~55 MB per 61-layer model). Env `SGLANG_HISA_DISABLE_POOL_CACHE=1` flips back to **v1** (no cache, fresh mean-pool). See commit TBD.
- Bench scripts live at `/data/sglang_scripts/` — `serve_baseline.sh`, `serve_hisa.sh` (v3 default), `serve_hisa_nocache.sh` (v1), `bench_serving.sh`, `eval_samsum.sh`.
- Bench history in memory: `project_hisa_bench_v1_v3.md` — baseline vs v1 vs v3 numbers at 2026-04-22, H200×8, 10×65K.
