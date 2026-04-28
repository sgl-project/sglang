# HISA TODO

Living list of outstanding work on the sglang HISA integration. Maintain in-place; don't treat memory entries as source of truth.

Legend: `[ ]` pending, `[/]` in progress, `[x]` done, `[~]` skipped/superseded, `[!]` lesson worth remembering, `[?]` needs investigation.

---

## Lessons learned (don't redo)

- `[!]` **compute-sanitizer 不一定能抓所有 OOB**。memcheck 检查的是 "访问是否在已分配 cudaMalloc 范围内"，OOB 偏移落在相邻 PyTorch alloc 上是合法 device memory，sanitizer 静默放过；但实际触发未映射页时驱动会报 Xid 13 → 表现为 silent crash。**抓 OOB 必须用 poison test + 裸跑**（`hisa_triton/test_oob_sanitizer.py` 就是这种压测；2026-04-28 用它抓到了 SK15 真实 bug）。
- `[!]` **Triton 的 int 除法是 C-style trunc-div**（向 0 取整），不是 Python 的 floor-div。`(-1) // 64 == 0` 在 Triton 里，**不是** `-1`。任何对从 tensor 加载来的、可能为负的值做 divmod 之前，必须先 gate `>= 0`。SK15 `buf_pos // PAGED_BLOCK_SIZE` 就是这个坑（`buf_pos = -1` 时 `phys = 0`、`row = -1`，phys 边界检查通过、但 row=-1 让后续 K-load 字节偏移为负 → OOB）。

---

## Roadmap — 当前主线（按顺序执行）

### Stage 0 — 验证当前状态（先做这个，再开始重构！）

刚修了 SK15 的 div bug + 多个 kernel 的 phys 边界检查。重构之前必须先确认这些 fix 真的让 K=16 silent crash 消失。**没验证就重构 = 在不稳定基础上动土，万一重构后又出 crash 不知道是新引入还是老遗留**。

- `[x]` **K=16 持续跑 30 分钟 long bench**（user 自测 2026-04-28，未观察到 crash）。SK15 buf_pos fix 暂认为已让 silent crash 消失。
- `[ ]` **K=16 samsum eval**：把 K=16 的精度 floor 立出来（之前只在 K=128 上验过 ROUGE 0.40）。
  ```bash
  bash /data/sglang_scripts/eval_samsum.sh deepseek-v32-hisa
  ```
- `[ ]` **K=16 vs K=128 e2e bench 基线**：长上下文（128K）下记录当前 TTFT/TPOT 数字作为后续重构 / topk 优化的对照基线。

### Stage 1 — 把所有 hot-path tilelang 替换成 triton

**这是结构重构的前置条件**：当前 wrapper 都是 "if K<64 走 triton, else 走 tilelang" 的二分支 dispatch，这恰好是要重构掉的东西。如果不先消除 tilelang 路径，"constexpr K 统一 dispatch" 的目标根本无法干净达成——总得在某层判 K 选 backend。

基础已具备：SK15/SK16 在 kernel 层早已支持 K>=64（GROUP_SIZE=1 退化情形），只是 production dispatch 还在走 tilelang。Phase-3 的 kernel-level bench 也证明 triton 在 K=128 上跟 tilelang 持平甚至略胜（tail-refresh triton 反超 17%）。

副收益：tilelang 这个外部依赖从 hot path 完全移除——jit-compile 时间、`pooling % paged == 0` 之类古怪 assertion、tilelang 自身 bug 等都一并消失。

- `[x]` **盘点 hot path 上还在用 tilelang 的调用点**（2026-04-28）：
    - decode `pool_k_cache.update_pool_for_completed_blocks`：K>=64 走 `fp8_native_paged_mean_pooling_completed_blocks_v3_interface`（tilelang）
    - decode v4 orchestrator `tail_only`：K>=64 走 `fp8_native_paged_mean_pooling_tail_only_v3_interface`（tilelang）
    - prefill `_get_topk_ragged`：K>=64 走 `fp8_native_hierarchy_mqa_logits`（tilelang，整条 ragged pipeline）
- `[x]` **K=128 triton vs tilelang 验证**（`hisa_triton/test_k128_parity.py`，2026-04-28）：

| Kernel | 正确性 (B=1, B=10, B=32 @ ctx=65K/128K) | speed triton/tilelang | 评估 |
|---|---|---|---|
| SK15 update_pool | B=1 完全一致；B=10/32 fp8 mean=5e-6, max=4（1 fp8-bin outliers），scale ULP only | 1.03-1.07x slower | 数值 OK，速度持平 |
| SK16 tail_only | **byte-equal across all sizes** (max\|abs\|=0) | **0.87-0.90x = triton 快 12-13%** | ★ 直接 swap |
| ragged orchestrator | sq=256: 完全一致（IoU=1.0）；sq=1024 skv=65K: topk_idx 直接 IoU=0.95 mean / 0.83 min（fp8 ULP 累积导致 ~5-17% block 翻转） | sq=256: 1.30x slow; sq=1024: 0.95x | 短 sq triton 慢 30%，需诊断 |

  **关键洞察**：fp8 ULP 漂移在大 seq_kv 下不可避免（fp32 累加顺序差异 + fp8 量化边界 round-to-nearest）。kernel-level "hard byte-equal" 在 B=32 / sq=1024 不可能；唯一靠谱的判据是 **e2e samsum ROUGE**（之前 SGLANG_HISA_VERIFY 阈值就是 IoU>0.95，正好这个量级）。
- `[x]` **诊断 ragged orchestrator triton 在 sq=256 慢 30% 的来源**（2026-04-28，torch.profiler）
    - 子 kernel level：triton 都比 tilelang 快（`block_sparse_mqa` 7.3μs vs 20.4μs，`ragged_pool_mqa` 1.6μs vs 4.2μs，`block_mean_pooling` 2.6μs vs 3.2μs）。子 kernel 总和 triton 17.5μs vs tilelang 39.5μs。
    - 真正瓶颈：**host 侧 cu_seqlen 算术 + triton 单 kernel launch overhead**。
        - tilelang 路径在 `pool_mqa_attn_return_logits_fp8_interface` 内部做 `cu_seqlen // K`、`+ K - 1`，每次 call 触发 ~3 个 PyTorch elementwise launch（floor_divide × 2、add、sub）。triton 路径之前同样问题。
        - triton 自带的 Python launch overhead（每 kernel ~3μs Python-side hashing/lookup）也比 tilelang 直接 cuLaunchKernel 慢。
- `[x]` **消除 cu_seqlen host-side 算术开销**（2026-04-28）
    - `_ragged_pool_mqa_kernel` 加 `K_BLOCK_SIZE: tl.constexpr` 参数，kernel 内部做 `ks // K`、`(ke + K - 1) // K`。orchestrator 直接传 raw `cu_seqlen_ks/ke`，跳过 host 侧 floor_divide+add 共 3-4 个 PyTorch launch。
    - 实测 wall-time @ sq=32 skv=4096: 改前 triton=135μs vs tilelang=88μs（差 47μs），改后 triton=110μs vs tilelang=95μs（差 15μs）。triton 省 ~25μs，gap 缩小 2/3。
    - 剩下 ~15μs gap 来自 triton Python-side launch overhead（4-5 kernel × ~3μs），需要 kernel fusion 或 CUDA graph capture 才能进一步压。先不做。
    - `ragged_pool_mqa_triton` API：`cu_seqlen_blocked_ks/ke` → `cu_seqlen_ks/ke` + `k_block_size: int = 1`。`k_block_size=1` 默认值保 BC（已 blocked 输入等价于 K=1 identity）。`profile_ragged_stages.py` 已同步更新，`test_precision.py` 因走默认 K=1 路径无需改。
- `[x]` **改 dispatch — `_get_topk_ragged` 默认 triton**（2026-04-28）
    - `hierarchy_indexer.py:881-895` 和 `:944-952`（chunked path）：原 `K<64 → triton, else tilelang` 改成 `os.environ.get("SGLANG_HISA_DISABLE_TRITON") != "1" or K<64 → triton`。复用 decode 那个 env var，关掉就回 tilelang baseline。
    - decode (`_get_topk_paged`) 已经默认 triton (v4)，无改动。
    - **下一步**：跑 e2e bench A/B（`SGLANG_HISA_DISABLE_TRITON=1` vs 默认）+ samsum ROUGE。
- `[ ]` **e2e samsum ROUGE 验证 K=128 全 triton**（这是 swap 的 go/no-go 阈值；不能下降）。
- `[ ]` **保留 tilelang interface 作为 bench 参照**（不删 source，但从 hot path 移除）。`benchmark.py` 仍然能 A/B 跑。

### Stage 2 — 加固测试网（重构的安全垫）

当前覆盖：`test_precision.py`（4 kernel-level 测试，K∈{8..128} byte-equal）+ `test_oob_sanitizer.py`（OOB poison stress）。**缺 orchestrator 级别的测试 + 全 K 烟雾测试**。

- `[ ]` **orchestrator 级 byte-equal 测试**：v4 orchestrator vs 一个独立的 reference（pure-torch 实现 / v1 path），输入完全相同时输出 byte-equal。覆盖 K∈{8, 16, 32, 64, 128}，B∈{1, 4}，ctx∈{4K, 65K}。
    - 现状只在 K=128 用 `SGLANG_HISA_VERIFY` env 在线 verify 过；K<64 没有显式 orchestrator 测试。
- `[ ]` **全 K 烟雾测试**（`test_smoke.py`，<5 秒跑完）：把 prefill+decode、K∈{8,16,32,64,128}、cache on/off、B∈{1, 4} 各组合都 launch 一次确保不崩，不做精度对比。作为 PR merge 门槛。
- `[ ]` **回归门槛**：把上面这些测试 + 已有的 `test_precision.py` + `test_oob_sanitizer.py` 串成一个 `make hisa-test` target。

### Stage 3 — 结构性重构（终于干这个）

目标：当前 dispatch 散在 4 个 wrapper + `pool_k_cache.update_pool` + `hierarchy_indexer._get_topk_paged` + `_get_topk_ragged` 各自的 if/else 里，结构性差。重构使代码可读性 / 可维护性提升，性能保持不变（dispatch 都是 compile-time，本来就免费）。

- `[ ]` **K-dispatch 统一进 kernel constexpr**（仿照 SK15 的设计）
    - 把 4 个 Category B kernel pair（`sparse_paged_mqa` / `block_sparse_mqa` / `paged_mean_pooling` / `block_mean_pooling`）的 grouped + legacy 合并成单 triton kernel + `constexpr GROUP_SIZE`。
    - K=64 / K=128 退化情形（GROUP_SIZE=1）需要 bench 验证不掉性能（应该是无回归，因为已经是同一段 IR）。
    - 收益：源码减少 ~50%，dispatch 集中在一处。0 runtime perf 改善。
- `[ ]` **dispatch 表中心化**：`hisa_triton/dispatch.py` 集中所有 K 选择 / kernel 选择 / tile-size 决策。各 wrapper 改成调它。类似 vLLM 的 `kernel_registry`，跨 wrapper 一致。
- `[ ]` **未实现 / 未验证分支统一 raise NotImplementedError**（不再 silent fallthrough）：
    - `_get_topk_ragged_with_cp`（已经 raise，确认覆盖完整）
    - `_get_topk_paged` 当 `next_n > 1`（spec decoding，已经 raise）
    - `release_kv_cache` 在 SessionAwareCache 路径（`req.req_pool_idx is None` 后 return）应显式 raise / log warn 提示 hisa pool pages 没被回收
    - `decode_kvcache_offload_manager.py` 里 `req_to_token_pool.free(req)` 没挂 hisa free hook → 加 `assert not isinstance(kv_pool, HisaNSATokenToKVPool)` 或显式 raise
    - `SGLANG_NSA_FUSE_TOPK=1` 路径（已经 assert，确认覆盖）

### Stage 4 — 重构后再次验证（差分对比）

- `[ ]` **跑全套 test 套件**（`test_precision.py` + `test_oob_sanitizer.py` + 新加的 orchestrator + smoke test）。
- `[ ]` **K=16 长 bench 重跑**，跟 Stage 0 的基线对比 TTFT/TPOT/ITL 应在 ±2% 内。
- `[ ]` **K=16 samsum eval 重跑**，ROUGE 不能下降。
- `[ ]` **任何精度或性能回归 → 不合并，回滚定位。**

### Stage 5 — block_topk fast-path（topk 优化）

K=16 ctx=128K 解码 e2e 比 K=128 慢 1.55x（80μs vs 52μs，per layer）。pipeline-stage bench 定位结果：
```
                K=16 (μs)  K=128 (μs)  ratio
update_pool       10.4       8.8       1.19
tail-refresh       7.6       9.0       0.84
block-MQA          9.6       8.0       1.20
torch.topk        38.4      13.8       2.78  ★
sparse-paged-MQA  14.1      12.0       1.17
TOTAL             80.0      51.6       1.55
```

`torch.topk` 占 K=16/K=128 总时间差的 ~89%。kernel 实现侧已饱和（SK15/SK16 vs tilelang 在公平对比下 17.8μs vs 17.8μs 完全持平），唯一剩余的大头是 topk。

诊断已经做的事：
- `[x]` `sgl_kernel.fast_topk_v2` 写死 K=2048（deepseek-v3.2 final indexer 专用），block_topk={64,…,1024} 用不了。
- `[x]` `hisa_triton/benchmark_pipeline.py` 已支持 K-dependent block_topk + `--force {auto,triton,tilelang}` 公平对比。
- `[x]` cudagraph 对 `torch.topk` 几乎无加速（90-95%），所以 40μs 是真实 GPU 工作时间，**替换有意义**。
- `[~]` 试了 3 条 triton 路全败：(1) 迭代 argmax K 次 runtime loop = 10x 慢（control-flow 不 fuse），(2) `tl.sort` packed int64 一次性排 = 8x 慢（bitonic O(N log²N)），(3) `tl.static_range` 完全 unroll K + N=16K = 编译跑 23 分钟没结束（IR 爆炸），全部丢弃。

**实测：sgl-kernel 的 radix-select 在 K=2048 上比 torch.topk 快 3-5x**：
```
config                          torch     sgl(radix)   speedup
B= 1  N= 8192  K=2048           41.9μs    13.4μs       3.12x
B=10  N= 8192  K=2048           46.9μs    13.4μs       3.49x
B=32  N= 8192  K=2048           50.1μs    14.6μs       3.43x
B= 1  N=16384  K=2048           73.9μs    17.4μs       4.24x
```
radix-select 时间几乎跟 K 无关（pure O(N), 2-pass histogram）。所以推断我们的 production 收益：
- K=16 path（top-512 of 8192）: 45 → ~13μs，**省 32μs**
- K=128 path（top-64 of 1024）: 14 → ~5-7μs，**省 7μs**
- **K=16 e2e/layer：80μs → ~50μs，跟 K=128 的 52μs 基本持平**

**集成方案**：fork `sgl-kernel/csrc/elementwise/topk.cu` 的 `fast_topk_cuda_tl`，把 line 23 的 `constexpr int TopK = 2048` 改成 `template <int K>`，按 production K 集合（{64, 128, 256, 512, 1024}）实例化 5 个 kernel。

- `[ ]` Templatize `TopK` constant in `topk.cu`。
- `[ ]` 加 `fast_topk_block_v2` Python interface（与 `fast_topk_v2` 并存）按运行时 K 选模板。
- `[ ]` 改 `pool_k_cache` orchestrator 调用：`torch.topk(...)` → `fast_topk_block_v2(...)`。
- `[ ]` Byte-equal 测试 + microbench + e2e bench。
- `[ ]` 工程量：1 天。

---

## Backlog（重要但不在主线，按需启动）

### Feature gaps（blocks shipping）

- `[ ]` **#9 CP path support for HisaIndexer** — `_get_topk_ragged_with_cp` currently raises NotImplementedError. vLLM 的 hisa prefill is CP-compatible in principle (K all-gather + Q split happen outside the kernel)。需要 CP-enabled smoke test。
- `[ ]` **#20 target_verify / draft_extend PAGED paths** — `_get_topk_paged` bails with NotImplementedError when `next_n > 1`. Speculative decoding workflows 需要这个。
- `[ ]` **#12 Support `SGLANG_NSA_FUSE_TOPK=1`** — HisaIndexer 当前 assert 它 off。fused 路径会跳过 Python `fast_topk_v2 + coord_transform` 桥。需要 `hierarchy_*_mqa_logits` 接受 fused-topk 输出 target。

### Correctness（应该在 Stage 0 / Stage 4 顺手解决）

- `[/]` **Prefix cache + pool_k_pages stale-data bug** — 当前 mitigation：`_store_index_k_cache` 在每次 extend 都 set `prev_seq_lens=0`，强制 `update_pool` 重做 `[0, new_complete)` 的 mean-pool。Works (samsum stable at ~0.40 across runs) 但对 same-request chunked prefill 是冗余的。
    - Proper fix: maintain a per-request "already-pooled-up-to" watermark and pool only `[watermark, new_complete)`. `alloc_for_extend` 检测到 cache-hit 时 set watermark=0；否则 carry forward。
    - 当前 mitigation 影响：~3-5x 冗余 mean-pool 工作 during chunked prefill (≤ 150 ms added to TTFT for 65K input)。可接受但丑。
    - Test: `bash /data/sglang_scripts/eval_samsum.sh` 3x 连跑，ROUGE > 0.40。
- `[ ]` **samsum ROUGE first-run floor** — first run 0.39 ± 0.02, subsequent 0.41 ± 0.04。within ±1σ of 0.40 threshold。Stage 4 重新验证后看是否还要进一步 dig。

### 低优先 perf 探索（重构完且 topk 落地之后再考虑）

- `[ ]` **SK17（暂编号）：`sparse_paged_mqa` K=64 路由到 grouped G=2 TILE=128**。SK12 在 K=128 上证明有效（B=1 -20~27%），同套路换到 K=64。预期 B=1/2 small-batch −15~25%，B=10 steady-state neutral~small win。实现成本：低（半天）；收益：确定性中等偏高。
- `[ ]` **SK18：grouped kernel `num_warps` sweep（K∈{8,16,32}）**。现在四个 grouped kernel launch 不指定 num_warps，可能默认 4 不是最优。预期命中 +5~10%。1 天 sweep + 整理。优先级 < SK17。
- `[ ]` **`nsys profile` 真实 workload** 来定 memory- vs compute-bound。给后续优化方向定调。
- `[ ]` **Bench at B=32 / B=64** 验证 v3 microbench 优势在大 batch e2e 是否落地。当前 B=10 是 v3 cross-over 点。

### Integration polish

- `[ ]` **Scatter pool_k_pages directly from prefill's ragged mean-pool** — 当前 prefill 的 `fp8_native_hierarchy_mqa_logits` 内部计算 mean-pool（ragged），然后 `_store_index_k_cache` 又调 `update_pool_for_completed_blocks` 对同样 block 算一次。冗余。fuse 能省 ~5-15 ms TTFT。
- `[ ]` **Pre-allocate `pool_page_tables` view for the gather kernel** — `pool_page_tables[:q_offset].contiguous()` in `_get_topk_paged` 触发 alloc。Trivially fixable。

### Cleanup / polish（重构时一并清掉）

- `[ ]` **删 `SGLANG_HISA_VERIFY` debug 路径** in `hierarchy_indexer.py`。Stage 4 验证完之后可以删。
- `[ ]` Rename legacy `max_pool_blocks_per_req` mentions：`grep -rn max_pool_blocks_per_req python/sglang/srt/layers/attention/nsa/hisa/`。
- `[ ]` Delete `benchmark/benchmark_indexer.py` v2b dead-code references。
- `[ ]` Document the pool-K page layout + allocator in 一个简短 README，避免下一个读者再 piece together。

### Skipped / superseded（保留作 archive）

- `[~]` **Batch `update_pool × 61 layers` into one kernel launch**。SKIPPED 2026-04-24：non-graph cost 0.81 ms/step，但 cudagraph 下只有 0.18 ms/step（最大省 0.17 ms/step = 0.3% TPOT）。架构 churn 不值。
- `[~]` **sparse_paged kernel 多个 D-series micro-opts**（D5/D6/D8 失败实验）。triton 的 `q.trans(1, 0)` 是 free WGMMA hint，`cache_modifier=".ca"` 默认就走 L1 read-only，简化 `pos_valid` 反而 codegen 变差。Phase-3 优化对照已记录。
- `[~]` **Warp specialization (producer/consumer)** — 30-50% 潜在收益，但需要切回 tilelang（triton 不暴露 `cp.async.bulk`）+ 1-2 周工作量。Hold until/unless sparse_paged 真的成 hotspot。

---

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

### **K=16** variant
```bash
bash /data/sglang_scripts/serve_hisa_16.sh
```
Sets `hisa_k_block_size=16, hisa_block_topk=512` (production formula `block_topk = 8192 // k_block_size`).

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
- `python/sglang/srt/layers/attention/nsa/hisa_triton/` — triton ports: `kernels.py`, `orchestrator.py` (v4), `benchmark.py`, `benchmark_pipeline.py` (per-stage), `test_precision.py`, `test_oob_sanitizer.py`

## Bench history

- In memory `project_hisa_bench_v1_v3.md` — baseline vs v1 vs v3 numbers at 2026-04-22, H200×8, 10×65K.
- v4 e2e bench result (2026-04-23, same bench): TPOT median 58.6 ms, ITL median 19.3 ms — v4 closes the gap to baseline to within 5-6%.
- K=16 vs K=128 per-stage bench (2026-04-28, B=1 ctx=128K): see Stage 5 above.
