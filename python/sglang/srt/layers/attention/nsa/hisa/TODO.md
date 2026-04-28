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

**In Progress — 间歇性 silent crash 排查（2026-04-28）**

K=16 配置下 server 跑一段时间挂掉，无 traceback。按 OOB 几率性触发模式定位。

- `[x]` SK16 `_tail_only_v3_kernel` 漏检 `phys < num_phys` 的防御边界检查已补（命名上 SK15 的 `max_kv_blocks` = num_phys，SK16 的 `max_kv_blocks` = `block_tables.shape[1]`，含义不一致是历史遗留，已在 wrapper 拆成 `max_blocks_per_req` + `num_phys` 两个参数）。不一定是当前 crash 的根因（K=128 用 legacy 同样无此检查但 user 没出事），但这是真实 latent bug，对齐 SK15 的安全 profile。
- `[x]` **批量补齐 phys 边界检查（2026-04-28）**——把同一个防御 pattern 扩展到所有读/写 phys 的 kernel：
    - 输入侧（读 main KV phys）：`_sparse_paged_mqa_kernel`（legacy）、`_sparse_paged_mqa_grouped_kernel`、`_paged_mean_pooling_kernel`（legacy）、`_paged_mean_pooling_grouped_kernel` —— 全部加 `valid &= (phys >= 0) & (phys < num_phys)` 后 clamp。wrapper 多传一个 `num_phys_blocks` 参数。
    - pool 侧（读/写 pool_k_pages phys）：`_batch_decode_pool_mqa_v3_kernel`（read）、`_update_pool_for_completed_blocks_kernel` 输出端（SK15 的 store 之前漏了，masked-store 用 `out_valid_per_g` 而不是 `valid_per_g`）、`_tail_only_v3_kernel` 输出端（SK16 的 store 之前漏了，加 early-return）。三个 wrapper 都多传 `num_pool_phys`。
    - `test_cross_k_block_size` + 其它 3 个 precision test K∈{8,16,32,64,128} 都通过，无回归。
- `[x]` **`hisa_triton/test_oob_sanitizer.py` 写好（2026-04-28）**——专项压测：每个 phys-loading kernel 跑 clean + poisoned 两 phase。poisoned 把 BlockTables / PoolPageTables / ReqToToken ~30% 替换成 sentinel（-1 / INT_MAX / num_phys 等）。clean 应全过；poisoned 在 clamp 都到位时也应全过；任何 missing clamp → 裸跑 cudaErrorIllegalAddress / sanitizer 报 invalid global read。
- `[x]` **真实 OOB 抓到并修了（SK15 SK15 SK15）**：`_update_pool_for_completed_blocks_kernel` 在读 `req_to_token` 后做 `phys = buf_pos // 64`。当 `buf_pos < 0`（比如 stale -1）时，**Triton 的 C-style trunc-div 让 phys = 0、row = -1** —— `phys >= 0` 检查通过、`phys < num_phys` 通过，src_valid=True，但用 row=-1 算出的 K-load 字节偏移是负数 → 真实 OOB。修法：在 div 之前先 gate `buf_pos >= 0`，把 buf_pos clamp 成 0 再做 divmod。
- `[!]` **教训：compute-sanitizer 不一定能抓到所有 OOB**。第一次开 sanitizer 跑这个测试，ERROR SUMMARY 报 0 errors；裸跑（无 sanitizer）才崩。原因猜测：sanitizer memcheck 检查的是“访问是否在已分配 cudaMalloc 范围内”，OOB 字节偏移可能落到相邻 PyTorch alloc 上（也是合法 device memory）就不报；但实际执行触发了 Xid 13。**结论：用户线上间歇性 silent crash 很可能就是这一类——只有触发到未映射页时才挂，touched-but-different-alloc 时静默跑错数。修完这条路径之后用户那个 K=16 长跑 crash 应该就消失了。**

**Pending — block_topk fast-path（K=16 长上下文加速，2026-04-28）**

K=16 ctx=128K 解码 e2e 比 K=128 慢 1.55x（80μs vs 52μs，per layer）。pipeline-stage bench 定位结果：
```
                K=16 (μs)  K=128 (μs)  ratio
update_pool       10.4       8.8       1.19    （SK15 vs tilelang，差距很小）
tail-refresh       7.6       9.0       0.84    （SK16 vs tilelang，triton 反超）
block-MQA          9.6       8.0       1.20    （pool_pages 8x → grid 大）
torch.topk        38.4      13.8       2.78  ★（block_topk 512 vs 64，N=8192 vs 1024）
sparse-paged-MQA  14.1      12.0       1.17
TOTAL             80.0      51.6       1.55
```

**`torch.topk` 一项就占了 K=16/K=128 总时间差的 ~25 μs（差距 28μs 的 89%）**。kernel 实现侧已饱和（SK15/SK16 跟 tilelang 在公平对比下 17.8μs vs 17.8μs 持平），剩下唯一有大头收益的优化是替掉 `torch.topk`。

诊断已经做的事：
- `[x]` 验证 `sgl_kernel.fast_topk_v2` 写死 K=2048（deepseek-v3.2 final indexer 专用），block_topk={64,…,1024} 用不了。
- `[x]` `hisa_triton/benchmark_pipeline.py` 已支持 K-dependent block_topk（`--topk-tokens 8192`）+ `--force {auto,triton,tilelang}` 切换 SK15/SK16 vs tilelang 公平对比。
- `[x]` 验证 cudagraph capture 对 `torch.topk` 几乎无加速（90-95%），所以 40μs 是真实 GPU 工作时间，不是 launch overhead——**替换有意义，能拿真收益**。
- `[~]` 试了 3 条 triton 路全败：(1) 迭代 argmax K 次 runtime loop = 10x 慢（control-flow 不 fuse），(2) `tl.sort` packed int64 一次性排 = 8x 慢（bitonic O(N log²N)），(3) `tl.static_range` 完全 unroll K + N=16K = 编译跑 23 分钟没结束（IR 爆炸），全部丢弃。

可参考的实现（按集成成本排序）：
- `[ ]` **`sgl-kernel/csrc/elementwise/topk.cu` 的 `fast_topk_cuda_tl`** ★ 最低成本：仓库里已经有 CUB-style radix-select 的 CUDA 实现，改自 tilelang `examples/deepseek_v32/topk_selector.py`。算法对任何 K 都通用（2-pass 8-bit histogram + threshold bucket），只是源码 line 23 写死 `constexpr int TopK = 2048`。改造方案：
    - 把 `TopK` 改成 `template <int K>`，按 production K 集合（{64, 128, 256, 512, 1024} = 8192 / k_block_size）实例化 5 个 kernel。
    - 加一个新 dispatch `fast_topk_block_v2`（与 `fast_topk_v2` 并存）按运行时 K 选模板。
    - 改 `pool_k_cache` orchestrator 调用：把 `torch.topk(...)` 换成 `fast_topk_block_v2(scores, lengths=full_lens, topk=block_topk)`。
    - 加 byte-equal 单元测试 + microbench。
    - **预估工程量：1 天。**
    - **实测收益（在 K=2048 上跑现成 `fast_topk_v2`，借此推断各 K）：**
        ```
        config                          torch     sgl(radix)   speedup
        B= 1  N= 8192  K=2048           41.9μs    13.4μs       3.12x
        B=10  N= 8192  K=2048           46.9μs    13.4μs       3.49x
        B=32  N= 8192  K=2048           50.1μs    14.6μs       3.43x
        B= 1  N=16384  K=2048           73.9μs    17.4μs       4.24x
        ```
        radix-select 时间**几乎跟 K 无关**（pure O(N), 2-pass histogram）。所以推断：
        - K=16 path（top-512 of 8192）: 45 → ~13 μs，**省 32μs**
        - K=128 path（top-64 of 1024）: 14 → ~5-7 μs，**省 7μs**
        - **K=16 e2e/layer：80μs → ~50μs，跟 K=128 的 52μs 基本持平**（1.55x 慢的差距抹平）。
- `[ ]` 备选：CUB 的 `cub::DeviceRadixSelect` 直接 cudaWrap（最稳，但要新建一个 .cu 文件 + cmake 集成）。
- `[ ]` 备选：Faiss `WarpSelect`（github.com/facebookresearch/faiss/blob/main/faiss/gpu/utils/warpselect/）—— warp-level bitonic merge，也是工业级实现。
- `[ ]` 备选：RAFT `select_warpsort`（rapidsai/raft）。

工具链：
- `hisa_triton/benchmark_pipeline.py` 已经写好，新 topk kernel 接好之后用 `--Ks 8 16 32 64 128` 跑端到端对比就行。
- `/tmp/topk_check.py` 是个独立的 microbench + correctness check，可以参考扩展成正式 test。

**Pending — 结构性重构（提升可维护性，2026-04-28）**

目标：当前 dispatch 散在 4 个 wrapper 各自的 if/else 里 + 9 个 kernel 文件，结构性差。重构使代码可读性 / 可维护性提升，性能保持不变（dispatch 本来就是免费的）。

- `[ ]` **K-dispatch 统一进 kernel constexpr**（仿照 SK15 的设计）
    - 把 4 个 Category B wrapper（sparse_paged / block_sparse / paged_mean_pool / block_mean_pool）的 grouped + non-grouped 合并成单个 triton kernel + constexpr GROUP_SIZE。
    - K=64 / K=128 退化情形（GROUP_SIZE=1）需要 bench 验证不掉性能。
    - 收益：源码减少 ~50%，dispatch 集中在一处。0 runtime perf 改善（compile-time specialization）。
- `[ ]` **未实现 / 未验证分支统一 raise NotImplementedError 而不是 silent fallthrough**
    - 例如 `_get_topk_paged` 在 spec decoding（next_n>1）已经 raise；但其他可能踩坑的分支需要扫一遍。
    - `release_kv_cache` 里 SessionAwareCache 路径（`req.req_pool_idx is None` 后 return）应该显式 raise 或 log warn 提示 hisa pool pages 没被回收。
    - `decode_kvcache_offload_manager.py` 里 `req_to_token_pool.free(req)` 也没挂 hisa free hook —— 加一个 `assert not isinstance(kv_pool, HisaNSATokenToKVPool)` 或显式 raise。
- `[ ]` **dispatch 表中心化**
    - `hisa_triton/dispatch.py` 集中所有 K-dispatch 决策（kernel 选择 + tile size + group size）。各 wrapper 改成调它。
    - 类似 vLLM 的 `kernel_registry`，跨 wrapper 的 dispatch 逻辑统一。

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
