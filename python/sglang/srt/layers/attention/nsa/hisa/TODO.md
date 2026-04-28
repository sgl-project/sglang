# HISA TODO

Living list of outstanding work on the sglang HISA integration. Maintain in-place; don't treat memory entries as source of truth.

Legend: `[ ]` pending, `[/]` in progress, `[x]` done, `[?]` needs investigation.

---

## Phase-3 — Small-K (k_block_size < 64) support (2026-04-27)

Goal: 让 hisa 在 `k_block_size ∈ {8, 16, 32, 64, 128}` 全范围工作，cache 默认开启；调用方只需改一个参数。

之前 tilelang 的 `fp8_native_paged_mean_pooling*` / `update_pool_for_completed_blocks` / `tail_only_v3` 都断言 `pooling_block_size % paged_block_size == 0`，导致 k<64 直接失效。本阶段把所有 cache 路径上的 tilelang kernel 都补齐了 triton 端口，按 K 分发。

**Done — kernels (`hisa_triton/kernels.py`)**
- `[x]` SK1–SK4：grouped 版 block_mean_pooling / paged_mean_pooling / sparse_paged_mqa / block_sparse_mqa，`GEMM_TILE = G·K ≥ 64`，覆盖 K∈{8,16,32,64,128}。两类 grouping：mean-pool 用连续 group，sparse 用 per-row gather。
- `[x]` SK7/9/10/11/12：各 K 下 BLOCK_N 实测最优值定下（block_sparse_mqa 256；block_mean_pooling 128 for k=128；batch_pool_mqa 自适应；sparse_paged_mqa k=128 走 grouped G=1 tile=128，B=1 −20~27%）。
- `[x]` **SK15 `update_pool_for_completed_blocks_triton`** — tilelang 同名 kernel 的 triton 端口；K∈{8,16,32,64,128}，per-row gather via ReqToToken；K≤64 → GEMM_TILE = G·K = 64，K>64 → GEMM_TILE = K。
- `[x]` **SK16 `tail_only_v3_triton`** — tilelang `fp8_native_paged_mean_pooling_tail_only_v3` 的 triton 端口；per-CTA per-req，按 `cur_size`（不是 K）做除法，paged 写回。

**Done — orchestrator / cache / indexer**
- `[x]` `orchestrator.fp8_native_hierarchy_paged_mqa_logits_with_pool_cache_v4`：tail-refresh 按 `k_block_size < paged_block_size` 分发，k<paged 走 SK16 triton，否则走 tilelang；后续 block-MQA / topk / sparse-MQA 全部 triton。
- `[x]` `orchestrator.fp8_native_hierarchy_paged_mqa_logits_triton`：no-cache 全 triton fallback orchestrator（仅在 `DISABLE_POOL_CACHE=1` 时用）。
- `[x]` `pool_k_cache.update_pool_for_completed_blocks` 内部按 K 分发：k<page_size 走 SK15 triton，否则走 tilelang。对调用方完全透明。
- `[x]` `hierarchy_indexer._get_topk_paged`：cache 默认开启的前提下，k<64 也能跑通；只有 `DISABLE_POOL_CACHE=1` 才会落到 no-cache triton orchestrator。**用户只需改 `hisa_k_block_size` 一个参数。**
- `[x]` `_store_index_k_cache` 去掉临时 SK14 guard，统一交给 `update_pool` 内部分发。

**Done — tests**
- `[x]` `hisa_triton/test_precision.py::test_cross_k_block_size`：K∈{8,16,32,64,128} 对四个 hisa 内核 fp8-strict 容差，新增 `_make_paged_kv_cache_soa` 修正之前 AoS 测试数据。
- `[x]` 与 tilelang 对比：k=128 byte_diff=1（fp8 ULP）；与 torch ref：k<64 byte_diff≤3。
- `[x]` 用户已 e2e 验证正确性（cache 默认开启路径）。

**Done — ragged prefill K<64 triton 路由 (SK17, 2026-04-28)**

针对前面讨论的 P0 问题（tilelang `fp8_native_block_mean_pooling` 在 K < block_N 时 boundary OOB），把整条 ragged prefill 也改走 triton。

- `[x]` 新增 `_ragged_pool_mqa_kernel` + `ragged_pool_mqa_triton`（`hisa_triton/kernels.py`）：tilelang `pool_mqa_attn_return_logits_fp8` 的 triton 端口，clean_logits + force_maintain 融进 GEMM 后处理（D4 嵌套 `tl.where`）。
- `[x]` 新增 `fp8_native_hierarchy_mqa_logits_triton`（`hisa_triton/orchestrator.py`）：mean-pool（SK1/SK6）+ ragged-pool-MQA（新）+ topk + sparse-MQA（SK4/SK11）。topk pad/unpad 逻辑（pad 到 GROUP_SIZE 倍数 + -1 → 内核自动 mask 成 -inf）处理 warmup 短输入边界。
- `[x]` `_get_topk_ragged` 两个分支（chunked / non-chunked）都加上 `if hisa_k_block_size < 64: → triton orchestrator` dispatch。K>=64 仍走 tilelang，零回归。
- `[x]` `test_precision.py::test_cross_k_block_size` 新增 `ragged_pool_mqa` 在 K∈{8,16,32,64,128} 的 fp8-strict torch ref 对比 + `hierarchy_triton` 全链路 smoke。
- `[x]` 用户生产配置（K=16, block_topk=512, seq_kv∈{8192, 8208, 65536}）smoke 通过——包括 8208 这个**正是 tilelang 越界条件**的非对齐长度。

**预期效果（待 e2e 验证）：**
- K=16/8/32 prefill 不再 Xid 13。warmup 不再随机 silent crash。
- prefill 速度提升 ~4×（grouped 设计消除了 tilelang 在 K<64 时的 4x 重复 K-load 浪费）。

**Pending — bench**
- `[ ]` **bench serving 性能验证（最重要的 next）**
    - `[ ]` k=128 默认路径 — 对齐 v4 baseline，无回归
    - `[ ]` k∈{32,16,8} cache 开启 — TTFT / decode 延迟、吞吐
    - `[ ]` longbench samsum 端到端 ROUGE
    - 启动后 grep `HisaNSATokenToKVPool` 确认 cache 真分配（cache 关闭时不打这行）
- `[ ]` 若 bench 表现良好，再考虑把 `hisa_k_block_size` 默认值下调
- `[ ]` 若 K≥64 也想统一走 SK15/SK16 triton（减少 tilelang 维护面），需要先确认无回归
- `[?]` SK1–SK4 的 group 维度 G 当前是按 K 静态选；后续可以做更细的 auto-tune

**Pending — kernel followup 实验（基于 2026-04-27 优化对照）**

经过对所有 hisa_triton kernel 的小 K vs 大 K 优化对照，没有发现"大 K 已经验证有效但小 K 漏掉"的优化。但有两个**大 K 没试过、小 K 也没试过**的方向值得评估：

- `[ ]` **SK17（暂编号）：`sparse_paged_mqa` K=64 路由到 grouped G=2 TILE=128**
    - 现状：K=64 命中 `_sparse_paged_mqa_kernel` legacy，grid `(B, seq, topk)`，每 CTA 处理 1 个 topk index，1 次 m=64 WGMMA，输出 64 个 logits。
    - 改动：路由到 `_sparse_paged_mqa_grouped_kernel`，`GROUP_SIZE=2, GEMM_TILE=128`。grid 变 `(B, seq, topk/2)`，每 CTA 用 per-row gather 取 2 个连续 topk index 跨的 2 个 paged page，1 次 m=128 WGMMA 输出 128 个 logits。
    - 这是 SK12 在 K=128 上证明有效（B=1 −20~27%）的同一个套路，对象换成 K=64。
    - 预期：B=1/2 small-batch −15~25%，B=10 steady-state neutral~small win，B=64 neutral。
    - 风险：legacy K=64 的 K-load 是单 phys 连续 [64, D]，编译器可向量化；grouped 多一些 per-row gather 索引运算（但 SK12 在 K=128 上已证不慢）。
    - 实现：`sparse_paged_mqa_triton` 里 dispatch 加一个 if 分支，~5 行；`test_cross_k_block_size` 已覆盖 K=64 byte-equal，加 microbench 即可。
    - 实现成本：低（半天），收益**确定性中等偏高**（最值得先试）。

- `[ ]` **SK18（暂编号）：grouped kernel `num_warps` sweep（K∈{8,16,32}）**
    - 现状：SK1–SK4 四个 grouped kernel（`_block_mean_pooling_grouped`、`_paged_mean_pooling_grouped`、`_sparse_paged_mqa_grouped`、`_block_sparse_mqa_grouped`）launch 时**没显式给 `num_warps=`**，triton 按 BLOCK shape 默认（GEMM_TILE=64 时通常是 4 warps）。
    - 改动：四个 kernel 各扫 `num_warps ∈ {4, 8}`，K∈{8,16,32}，看是否有命中。
    - 8 warps 的 trade-off：能并行 2 个 WGMMA 或 hide mem latency，但寄存器/SM 占用翻倍 → 每 SM 驻留 CTA 减半。GEMM_TILE=64 的小 tile（mean_pooling、sparse_paged grouped）可能被反超；GEMM_TILE=256 的 block_sparse 默认大概率已是 8。
    - `num_stages` 不用扫：这 4 个都是 single-tile-per-CTA 无 inner loop，num_stages 无效。
    - 预期：命中 +5~10%，没命中 0%；每 kernel 命中点可能不同。
    - 实现：launch 参数加 `num_warps`，`benchmark.py` 跑笛卡尔积。
    - 实现成本：中（1 天 sweep + 整理结果），收益**不确定**。优先级低于 SK17。
    - 前置条件：bench serving 跑完、SK17 有结论后再做。

---

## P0 — Kernel-level perf (attack the sparse_paged 80% hotspot)

**Status 2026-04-24 update:** The 80% hotspot claim was pre-triton-port. After porting, the triton `sparse_paged_mqa_triton` is 5-20× faster than the original tilelang baseline. Rechecked kernel bench numbers on H200 (kernels.py, default config):
- B=1 ctx=65K: 0.011 ms (tilelang was 0.214 ms → 19×)
- B=8 ctx=65K: 0.012 ms (17×)
- B=32 ctx=65K: 0.022 ms (10×)
- B=64 ctx=65K: 0.036 ms (6×)

At B=64 we're at ~73% of HBM peak bandwidth already. At B=1-8 we're launch/occupancy-limited, not bandwidth-limited.

Tried optimizations (2026-04-24, 2026-04-25):
1. `[~]` **Pipeline stages (`num_stages=2/3`)** — NO WIN. Both sparse_paged and v3 kernels are 1-tile-per-CTA with no inner loop; nothing to overlap. `num_stages=3 + num_warps=4` regressed B=1 by 40%. Reverted.
2. `[~]` **Q reuse via chunked topk (A1)** — NO WIN. Tested CHUNK ∈ {1, 2, 4, 8, 16} with `tl.range(num_stages=2)`. Best case: CHUNK=2 at B=64 saved 5%; all other configs equal-or-worse. Root cause: at small B the kernel is instruction-latency-bound (Q bandwidth isn't the bottleneck); at B=64 we're near HBM peak anyway. Reverted.
7. `[x]` **Merge two `tl.where` calls into nested expression (D4)** — WORKS on v3 kernel: 12-22% faster across all (B, num_pool). Same trick FAILED to help on sparse_paged (only one tl.where there) and FAILED to help on v3's post-GEMM mul/max/mul (D9: +1μs regression). Conclusion: the benefit is specific to merging sequential **control-flow selects** (where chains), not arithmetic chains.
8. `[~]` **`cache_modifier=".ca"` for read-only metadata loads (D6)** — NO WIN. v3 kernel +1-2 μs (10-25% slower); sparse_paged neutral. Triton's default LDG already routes small read-only loads through L1 read-only cache on H100; explicit .ca perturbs codegen. Reverted.
9. `[~]` **Drop `q.trans(1, 0)` (D5)** — NO WIN. Two variants: (a) hoist `tl.trans(q)` early — neutral; (b) swap GEMM direction `tl.dot(q, k.T)` with reduce axis=0 — large regression (B=64 +30%). Triton's `q.trans(1, 0)` in tl.dot is a free WGMMA descriptor hint, not a register shuffle. Reverted.
10. `[~]` **Simplify `pos_valid` mask (D8)** — counter-intuitive REGRESSION. Dropping the `(k_i >= 0)` clause (redundant given valid_page) made B=1/B=8 -18~20% SLOWER. Triton picked a different codegen path. Reverted.
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
