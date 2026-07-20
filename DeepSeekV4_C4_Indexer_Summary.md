# DeepSeekV4 C4 Indexer 计算公式总结

## 1. 总体抽象

C4 indexer 的核心作用是：对当前 token，在历史的 C4 compressed KV blocks 上做一次轻量检索，选出后续 sparse attention 要访问的 top-k compressed KV 位置。

抽象公式：

```text
I_t = TopK_m Score(q_t, k_m)
```

其中：

```text
q_t = Q_index(x_t)

k_m = C4Compress(x_{4m-7 : 4m})

Score(q_t, k_m) = Σ_h c_{t,h} · ReLU(<q_{t,h}, k_m>)
```

最终输出 `I_t` 会从逻辑 C4 block index 转换成实际 KV cache slot，写入 `c4_sparse_page_indices`，供 C4 sparse attention 使用。

## 2. 输入 `x_t`

`x_t` 是当前层、当前 token 的 hidden state。C4 indexer 不直接基于 token id 工作，而是基于 attention 层输入的 hidden activation 来计算：

- indexer query
- per-head score coefficient
- C4 compressed key

## 3. C4 compressed key：`k_m`

每 4 个 token 产生一个 C4 compressed key，但压缩不是简单平均，而是看一个带 overlap 的 8-token 局部窗口：

```text
x_{4m-7}, ..., x_{4m}
```

抽象公式：

```text
k_m = Transform(Σ_j softmax_j(r_j + b_j) · v_j)
```

各部分含义：

- `v_j`：窗口内第 `j` 个候选 token 投影出的 value。
- `r_j`：窗口内第 `j` 个候选 token 投影出的局部选择 score。
- `b_j`：模型参数中的 `ape`，可以理解为不同压缩槽位的 bias。
- `softmax_j(...)`：在局部 8-token 窗口内计算权重，决定每个 token 对 compressed key 的贡献。
- `Transform(...)`：后处理，包括 RMSNorm、RoPE、Hadamard，以及量化前的旋转变换。

所以 `C4Compress` 的语义是：

```text
把一小段连续 token 压成一个代表性 key，作为稀疏检索的 memory item。
```

## 4. Indexer query：`q_t`

当前 token 的 indexer query 抽象为：

```text
q_t = Quant(Hadamard(RoPE(W_q · f(x_t))))
```

各部分含义：

- `f(x_t)`：主 attention 里已经算出的 q-lora 表示。
- `W_q`：indexer 自己的 query projection。
- `RoPE`：给 query 注入位置信息。
- `Hadamard`：对 query 做正交旋转，利于后续量化和点积。
- `Quant`：通常量化为 FP8；开启 FP4 indexer 时走 FP4 相关路径。

## 5. Per-head score coefficient：`c_{t,h}`

当前实现中的 `weights_proj` 可以抽象成：

```text
c_t = W_c · x_t · scale
```

然后每个 head 的匹配分数会被 `c_{t,h}` 调制：

```text
Score(q_t, k_m) = Σ_h c_{t,h} · ReLU(<q_{t,h}, k_m>)
```

这里需要注意：`c_{t,h}` 不是 MoE 里常见的概率 gate。它没有 softmax 归一化，因此不保证：

```text
Σ_h c_{t,h} = 1
```

也不保证：

```text
c_{t,h} >= 0
```

因此它更准确的语义是：

```text
per-head score coefficient / per-head modulation weight
```

而不是“各 head 的贡献比例”。

## 6. Score 计算

核心打分公式：

```text
Score(t, m) = Σ_h c_{t,h} · ReLU(<q_{t,h}, k_m>)
```

各部分含义：

- `<q_{t,h}, k_m>`：当前 token 在第 `h` 个 indexer head 上，与第 `m` 个 C4 compressed key 的相似度。
- `ReLU(...)`：负相关会被截断为 0。
- `c_{t,h}`：第 `h` 个 head 的动态线性调制系数。
- `Σ_h`：把多个 indexer head 的结果汇总成一个 scalar score。

所以 `Score(t, m)` 表示：

```text
当前 token t 应该多关注第 m 个 C4 compressed block。
```

## 7. Top-k 选择

对所有历史 C4 compressed blocks 做 top-k：

```text
I_t = TopK_m Score(t, m)
```

如果当前 C4 历史长度小于等于 `topk`，实现中会直接选全部有效 C4 blocks，不再做排序。

逻辑 C4 index 会进一步通过 page table 转成实际 cache slot：

```text
c4_page_size = page_size / 4

slot = page_table[raw_index // c4_page_size] * c4_page_size
     + raw_index % c4_page_size
```

最终：

```text
c4_sparse_page_indices[t, :] = slot
```

## 8. 与 MoE gate 的区别

MoE gate 通常是概率式路由：

```text
p = softmax(W_gate x)

y = Σ_{e in TopK(p)} p_e · Expert_e(x)
```

因此 MoE gate 通常可以解释为：

```text
当前 token 分配给各 expert 的概率或贡献比例。
```

C4 indexer 的 `weights_proj` 不同：

```text
c = W_c x
```

它没有 softmax，也不做归一化，只是用于混合各 indexer head 的匹配分数：

```text
Score(t, m) = Σ_h c_{t,h} · ReLU(<q_{t,h}, k_m>)
```

所以二者共同点是：

```text
都根据当前输入动态生成权重。
```

区别是：

```text
MoE gate: 归一化路由概率，用来选择和混合 experts。
C4 indexer coefficient: 未归一化线性系数，用来混合各 indexer heads 的匹配分数。
```

## 9. 一句话总结

```text
C4 indexer = 用当前 token 的 query，在压缩后的 C4 memory 上做轻量检索，
             通过未归一化的 per-head coefficient 汇总匹配分数，
             最终选出 top-k 个 compressed KV blocks 供 sparse attention 使用。
```

## 10. DCP 拆分：哪些计算可以分片

DCP 优化需要把三类问题分开看：

1. **C4 indexer score 分片**：沿历史 C4 item 维度切分，每个 DCP rank 只计算自己拥有的历史。
2. **Attention 输出合并**：每个 rank 先对本地 KV history 得到 partial output 和 LSE，再跨 rank 合成全局 attention。
3. **C128/SWA cache ownership 与 metadata**：属于 cache/store 和索引路径，不应与独立于 DCP 的 C128 fusion 混为一项。

当前 P0 聚焦前两项。C128 fusion 已取消，不作为 DCP 优化的前置条件。

### 10.1 C4 local top-k 与全局 top-k 的等价性

把完整候选集合拆成互不相交的 DCP shards：

```text
S = S_0 union S_1 union ... union S_{D-1}
```

设最终需要全局 top-k。每个 rank 先保留：

```text
C_r = TopK_k(S_r)
```

无并列 score，或所有路径使用相同确定性 tie-break 时，有：

```text
TopK_k(S) = TopK_k(C_0 union C_1 union ... union C_{D-1})
```

证明要点：若某个候选 `x` 没进入其所在 shard 的 local top-k，则同一 shard 内至少有 `k` 个候选分数不低于 `x`。这些候选也都属于全局集合，因此 `x` 不可能严格进入全局 top-k。于是所有可能进入全局 top-k 的元素一定包含在 local top-k 的并集中。

存在相同 score 且没有统一 tie-break 时，不同实现可能选择不同 index；此时等价性应定义为 top-k score 多重集合一致，并且返回 index 都属于相应并列集合，而不是要求 index 集合逐项相等。

### 10.2 不存在跨 rank 分母不一致

C4 indexer 的 score 是直接可比较的未归一化标量：

```text
Score(t, m) = sum_h c_{t,h} * ReLU(<q_{t,h}, k_m>)
```

`c_{t,h}` 没有 softmax，历史 item 维度上也没有归一化分母。因此各 rank 对不同 history shards 算出的 score 在同一数值尺度上，可以直接 gather 后做 global top-k。

这和 attention 不同。Attention 的本地输出带有各自的 softmax 分母，必须通过全局 LSE 校准后才能合并。

### 10.3 Interleave 粒度

C4 compressor 的局部窗口包含 8 个 raw tokens，因此不能在任意 token 边界切断压缩语义。当前实现按 C4 page 做 interleave：一个 C4 page 含 64 个 compressed items，对应 256 个 raw tokens；rank `r` 处理逻辑 page `r, r+D, r+2D, ...`。该边界天然是 8 的倍数，不会拆开 compressor window。

当前 C4 shard 路径还保留以下语义：

- 短 history 不排序，按 sequential index 返回所有有效项。
- 非整页尾部通过有效长度 mask 排除。
- local candidate 使用 global raw index，global merge 后再映射到物理 C4 slot。
- packed 路径把 FP32 score 与 raw index 合成一个 `int64` candidate，只执行一次 `all_gather_into_tensor`。

## 11. DCP Attention 合并与 A2A

设 rank `r` 对本地 KV history 算出的 partial output 为 `O_r`，对应 LSE 为 `L_r`。全局结果为：

```text
L = logsumexp_r(L_r)
O = sum_r exp(L_r - L) * O_r
```

因此 attention 合并必须同时处理 output 和 LSE，不能像 C4 indexer score 一样直接排序。

### 11.1 两条通信路径

原路径：

```text
LSE all-gather -> global LSE/scale -> weighted output all-reduce -> local head shard
```

A2A 路径：

```text
按目标 head chunk pack output/LSE
-> output all-to-all + LSE all-to-all
-> 每个 rank 本地完成自己 head chunk 的 global LSE 与 weighted merge
```

A2A 避免了让所有 rank 都得到完整 merged output，但它并不把接收 payload 降到原来的 `1/D`：每个目的 rank 会从 `D` 个 source ranks 各收一个 `1/D` head chunk，合计仍是一份完整 partial-output 大小。它的收益来自通信拓扑和最终输出分片方式，而不是凭空消除 attention 数据。

AllReduce 通常具有高度优化的 ring/tree 实现且可原地工作，因此 A2A 并非在所有 DCP size 和 batch 上都更快。现有结果符合这个判断：DCP=2 为负收益，而 DCP=8、固定 B128 的 Pro 场景出现正收益。

### 11.2 128K 固定批次结果

测量条件：TP8/DCP8/DP1/EP8、global batch 128、prefix 128000、output 512、固定 expert round-robin、C4 shard 关闭、CUDA Graph。

| 变体 | Median TPOT | Throughput | 相对基线 |
|---|---:|---:|---:|
| Baseline AG/AR | 52.151 ms | 2454.39 tok/s | 1.000x |
| Attention A2A | 50.052 ms | 2557.32 tok/s | 1.0419x |

A2A 的 TPOT 改善为 `4.025%`，吞吐改善为 `4.194%`。Baseline CV 为 `0.135%`，A2A CV 为 `0.041%`，结果稳定。

该结果只证明 128K/B128/DCP8 固定路由 decode 窗口内的相对收益，不代表所有 batch、DCP size 或生产路由都应该启用。A2A 继续默认关闭，保留给 DCP8 Pro 场景验证。

另一个重要边界是：A2A 的通信张量大小主要由 `batch x heads x head_dim` 决定，并不随 context length 线性增长。128K 会放大本地 attention/C4 history 计算与 KV cache 占用，也会改变通信在整个 step 中的相对占比，但不会把 A2A output payload 本身放大 128K 倍。

## 12. 已知缺陷

### 12.1 P0：A2A CUDA Graph 显存增加

在首次 B128 graph capture 前，两条路径每张 GPU 都约有 `17.80 GB` available memory：

| 时点 | Baseline | A2A | 差值 |
|---|---:|---:|---:|
| 首次 B128 capture 后 available | 10.10 GB | 6.65 GB | -3.45 GB |
| 代表性 graph memory | 8.43 GB | 11.93 GB | +3.50 GB |

后续较小 bucket 在两边都只继续增加约 `0.4 GB`，因此差值几乎全部产生于首次 B128 graph capture。

当前 A2A 路径会物化：

- destination-major 的 `out_send`；
- 完整 `out_recv`；
- `nan_to_num` 产生的完整副本；
- FP32 weighted-output 中间量；
- `lse_send/lse_recv`、NCCL workspace 和 CUDA Graph 固定地址资源。

以 B128、H64、D512、BF16 估算，一份完整 output 为 8 MiB；显式大张量约为 `8 + 8 + 8 + 16 = 40 MiB/layer`，但这不能直接解释全部 3.5 GB。CUDA Graph 会固定跨层地址，NCCL graph/P2P 资源和 allocator reservation 也可能贡献显存。当前还没有完成显式 tensor、allocator reserve 与 NCCL graph resource 的精确拆账。

结论：这是 deployment blocker 级别的显存问题。A2A 在没有 memory snapshot 证明和显存优化前不能默认开启。

优先验证的降显存方向：

1. 跨层复用 graph-stable 的 A2A workspace。
2. fused recv-side LSE/weighted accumulation，避免 `nan_to_num` 和 FP32 full-size 中间量。
3. 让 attention 输出直接采用 A2A-ready layout，去掉 `permute().contiguous()` 副本。
4. 分离 PyTorch allocated/reserved、graph private pool 与 NCCL resource，确认 3.5 GB 的真实来源后再决定 kernel 方案。

### 12.2 P0：绝对 TPOT 过大

虽然 A2A 相对基线快约 4%，但固定 B128/128K 下绝对 TPOT 仍为 `50.052 ms`，基线为 `52.151 ms`。这相当于单请求约 20 token/s；512-token decode 窗口约需 25.6 秒，基线约 26.7 秒。

因此当前结果不能只按 speedup 判定成功。A2A 只是把一个仍然偏慢的 step 缩短了约 2.1 ms，需要先回答剩余约 50 ms 花在哪里。由于该 benchmark 在最后一次 TTFT 后统计 decode，额外的 256-token residual prefill 不应被算进这段 TPOT。

在拿到同硬件、同模型、同 batch 的已验证生产基线前，先把它记录为“高绝对 TPOT/疑似性能缺陷”，不直接断言为哪一模块的 regression。

可能贡献者包括：

- 43 层中逐层发生的 DCP attention 通信与 stream synchronization。
- C4 shard 在这次 A/B 中关闭，C4 indexer 仍在各 DCP rank 对完整 history 重复计算。
- C4/C128 FlashMLA、extra-KV 和 C128 online state 路径。
- MoE gate、DeepEP dispatch/combine 与 expert GEMM。
- output/LSE layout conversion、dtype conversion 和未融合 elementwise kernels。
- CUDA Graph 内 NCCL 与 compute 串行，缺少跨 head chunk 或跨层 overlap。
- host gap、graph bucket/padding 或额外 metadata/compaction kernel。

## 13. 下一轮 128K Timeline Profiling

下一步先做性能拆账，不立即叠加新的 runtime 优化。主目标是解释一个稳定的 B128 decode step，并区分 kernel 总耗时与真正 critical path。

### 13.1 Profiling 矩阵

| 变体 | C4 shard | Attention merge | 用途 |
|---|---:|---|---|
| A | 0 | AG/AR | 当前基线 |
| B | 0 | A2A | 定位约 2.1 ms 收益来源 |
| C | 1 | packed candidate + AG/AR | 量化 128K C4 冗余计算 |
| D | 1 | packed candidate + A2A | 观察组合后的关键路径与 overlap 空间 |

固定 TP8/DCP8/DP1/EP8、B128、prefix 128000、output 512 和固定 expert 路由。先完成 cache warmup，再只捕获 20–50 个稳定 decode steps，避免 HTTP、tokenization、prefill 和超大 trace 文件干扰。

如资源允许，再补同 shape 的 DCP2 与非 DCP/已验证生产基线，区分模型固有开销与 DCP tax。若非 DCP 128K 因 KV 显存不可行，应明确记录为不可比，而不是改变 batch 后直接对比。

### 13.2 Timeline 必须拆出的区间

1. Q/QKV projection、RoPE 与 attention metadata。
2. C4 indexer query、paged-MQA score、local top-k、candidate collective、global merge。
3. C4/C128 FlashMLA 与 extra-KV attention。
4. DCP pack、output collective、LSE collective、global LSE 和 weighted merge。
5. Attention output projection。
6. C128 online state update 与 128 边界处理，仅做归因，不在本阶段实现 fusion。
7. MoE gate、DeepEP dispatch/combine、expert GEMM。
8. Stream wait、NCCL serialization、graph replay gap 与 host idle。

先用 PyTorch/Kineto + NVTX 得到可读的 per-layer trace 和 kernel 聚合；若 NCCL 内部阶段仍不透明，再用 Nsight Systems 补一轮。每项都同时看 duration、调用次数、stream 归属和前后依赖，不能把并行 kernel 时间简单相加当成 TPOT。

### 13.3 显存采样

对 A/B 分别在以下时点采集：

1. graph capture 前；
2. 首次 B128 capture 后；
3. 所有 bucket capture 后；
4. 请求 cache 分配并完成 warmup 后。

记录 `memory_allocated`、`memory_reserved`、graph private pool，并保存 allocator memory snapshot。结合 tensor shape 标注和 NCCL 日志，回答额外 3.45–3.50 GB 中有多少来自显式 A2A tensor、有多少来自 graph pool/NCCL。

### 13.4 Profiling 必须回答的问题

- 剩余约 50 ms 的前三大 critical-path 模块是什么？
- A2A 的约 2.1 ms 收益来自 collective、layout/merge，还是减少了某个同步点？
- C4 完整 history 重复计算在 128K step 中占多少，启用 shard 后通信是否抵消计算收益？
- 两次 A2A 是否串行，output 与 LSE 能否合并或 overlap？
- NCCL 结束后是否存在可消除的 elementwise/layout kernel 链？
- 是否存在跨层 graph gap、错误 bucket、batch padding 或异常 host synchronization？
- head chunk 计算与通信是否有足够长的独立区间值得 overlap？

## 14. DCP 后续优化顺序

基于当前证据，建议顺序为：

1. **先抓 timeline 和 memory snapshot**：解释绝对 TPOT 与 3.5 GB 增量。
2. **闭环 C4 shard**：128K 时 score 计算随 history 增长，而 candidate collective 固定在 `DCP x 512`；这是最可能随长上下文放大的 DCP 特有机会。
3. **降低 A2A graph memory**：优先 workspace 复用和 recv-side fusion，同时守住 DCP8/B128 的 4% 收益。
4. **建立启用策略**：按 DCP size、batch、可用显存和实测 crossover 决定 A2A；不使用全局默认开启。
5. **最后评估 head-chunk overlap**：只有 timeline 证明 NCCL 暴露在 critical path，且后续 head chunk 有足够计算可覆盖时再实现。

Head-chunk overlap 在 DCP8 下要格外谨慎：H64 最终每 rank 只有 8 个目的 heads，继续切小会降低 attention kernel 和 collective 效率，还会增加 graph nodes、event 和双缓冲显存。它应建立在 A2A memory 已收敛且 timeline 明确显示可覆盖窗口之后。

C128 独立 fusion 继续取消。DCP 侧只保留 C128 cache ownership、metadata/compaction 和 attention 合并的 profiling；除非 trace 证明它们进入前三大 critical path，否则不扩展本轮范围。

## 15. 当前结论

```text
C4 indexer: 数学上可沿 history shard，local top-k union 与 global top-k 等价；
             128K 下应优先用 timeline 验证重复 score 的真实占比。

Attention A2A: DCP8/B128/128K 已有约 4% 正收益，但新增约 3.5 GB/GPU graph memory；
               保留实现、默认关闭，先做显存拆账与降占用。

绝对 TPOT: 50–52 ms 仍然偏大，下一步的首要任务是 profiling 全 step critical path，
           而不是继续堆叠未经归因的优化。
```

本轮原始产物位于：

```text
/data/models/dev0616/c4_c128_p0/d804277991/a2a_dcp8_fixed_batch/
```

## 16. 128K Timeline 与 C4 Shard 闭环（2026-07-18）

### 16.1 DCP8 Runtime Blocker 与修复

首次启用 C4 shard 时，B128 CUDA Graph capture 在 packed candidate merge 处失败：

```text
DCP candidate path currently requires dcp_size=2
```

根因是 fused candidate merge 只实例化了 DCP2。当前实现已扩展到 DCP2/4/8，并按 DCP size 分别实例化 merge kernel，避免 DCP2 总是承担 DCP8 的 shared-memory 配置。相关提交：

- `36d600aeb9`: DCP2/4/8 packed C4 top-k；
- `ae418cee2a`: 修复模板 kernel launcher 命名空间；
- `9d9edbea06`: 局部格式整理。

`test/registered/jit/test_dsv4_dcp_topk.py` 已在 H20/CUDA 13 容器通过，覆盖 DCP2/4/8、短 history、非整页尾部、长 history、tie 和空 shard。DCP8 B128 的全部 decode CUDA Graph bucket 也已成功 capture/replay。

### 16.2 GPU-only CUDA Graph Trace

CPU+GPU Kineto 会严重扰动 NCCL，因此定量结果只使用 GPU-only profiler。每个变体包含 8 rank，每个 rank 取 4 次稳定 graph replay；profile 后的 HTTP TPOT 不参与性能结论。

固定 shape：TP8/DCP8/DP1/EP8、B128、prefix 128000、固定 round-robin expert、online C128、FP8 KV cache。

| 变体 | C4 shard | Attention merge | Graph span (ms) | 相对 A |
|---|---:|---|---:|---:|
| A | 0 | AG/AR | 46.031 | - |
| B | 0 | A2A | 44.319 | -3.72% |
| C | 1 | packed candidate + AG/AR | 42.401 | -7.89% |
| D | 1 | packed candidate + A2A | 40.722 | -11.53% |

C4 shard 的主要拆账：

| 项目 | A (ms) | C (ms) | Delta (ms) |
|---|---:|---:|---:|
| 21 层 C4 paged-MQA score | 5.208 | 0.771 | -4.438 |
| C4 top-k/candidate/merge | 0.934 | 0.580 | -0.354 |
| 全 step NCCL AllGather | 3.719 | 4.391 | +0.672 |
| Graph span | 46.031 | 42.401 | -3.630 |

这证明 128K/DCP8 下，history shard 将 C4 score 缩短约 `85.2%`，21 次 candidate collective 没有抵消计算收益。AllGather 增量是全 step 聚合差值，可视为 candidate 通信成本的近似上界，而不是单独 NCCL event 的精确计费。

A2A 在 C4 shard 之上仍有独立收益：C 到 D 的 graph span 再下降 `1.679 ms`（`3.96%`）。它移除 43 次 FP32 attention-output AllReduce（`4.469 ms`），新增 86 次 SendRecv（`2.383 ms`），同时 FP32 reduce128 增加约 `0.317 ms`。两项优化在当前 shape 下基本可叠加。

### 16.3 无 Profiler E2E

使用 pretokenized fixed B128、prefix 128000、output 512；计时从最后一个请求收到首 token 后开始，因此 256-token page tail admission 不计入 TPOT。

| 变体 | TPOT runs (ms) | Median TPOT (ms) | Median output tok/s |
|---|---|---:|---:|
| A tail | 51.968, 52.049 | 52.009 | 2461.13 |
| C | 48.716, 48.804 | 48.760 | 2625.11 |
| D | 46.661, 46.667 | 46.664 | 2743.02 |

相对当前 commit 的 A tail：

- C4-only：TPOT `-6.247%`，吞吐 `+6.663%`，decode speedup `1.06663x`；
- C4 + A2A：TPOT `-10.277%`，吞吐 `+11.454%`，decode speedup `1.11454x`；
- D 相对 C：TPOT再下降 `4.299%`，吞吐再提高 `4.492%`。

A tail 与之前 6 次 bracketed A 的 `52.151 ms` 相差 `0.27%`，说明基线漂移较小。一次 output 64 的 C4 首轮曾得到 `62.75 ms`，但 output 512 的两轮结果和 graph trace 都稳定为正；短输出单次结果不能用于该 shape 的 go/no-go。

当前 C/D 各只有两次无 profiler 长跑，足以确认方向和量级，但尚未替代最终验收要求的三次 A-B-C-D-A 矩阵。

### 16.4 CUDA Graph 显存

代表性非 TP7 rank：

| 变体 | Graph memory (GB) | Capture 后 available (GB) |
|---|---:|---:|
| A | 8.43 | 9.34 |
| C | 8.44 | 9.34 |
| B | 11.93 | 5.84 |
| D | 11.96 | 5.82 |

C4 packed candidate 在 DCP8/B128 下只增加约 `0.01-0.03 GB/GPU`，不是显存问题来源。约 `3.5 GB/GPU` 的增量仍由 A2A 路径和其 graph/NCCL 资源主导；D 变体虽然性能最好，但 A2A 仍必须默认关闭。

### 16.5 剩余 Step 结构

D 变体 `40.722 ms` graph 的主要聚合 kernel 为：

- 4096x4096 GEMM：`5.342 ms`；
- NCCL AllGather：`3.638 ms`；
- DeepEP combine：`3.132 ms`；
- 4096x2048 GEMM：`2.954 ms`；
- A2A SendRecv：`2.383 ms`；
- sparse FlashMLA：`1.923 ms`；
- direct-copy/layout kernels：`1.658 ms`。

其中 GEMM、DeepEP 和 FlashMLA 不是这轮新增的 DCP 通信优化。DCP attention 下一批最明确的目标是 A2A direct-copy/layout 链、recv-side weighted/LSE merge，以及 graph-stable workspace 复用；它们同时关系到约 2 ms 的残余 kernel 开销和 3.5 GB 显存问题。

## 17. 更新后的下一步

1. **A2A 显存拆账与压缩**：采集 allocator snapshot，区分显式 tensor、graph private pool 和 NCCL resource；实现跨层 graph-stable workspace 复用。
2. **Recv-side fusion**：融合 unpack、global LSE、weighted accumulation，优先消除 full-size `nan_to_num`/FP32 中间量和 direct-copy 链。
3. **正式 E2E 验收**：在 128K 上补齐 A-B-C-D-A、每变体至少三次，并扩展 B8/32/64/128；随后测 16K/64K crossover，决定 C4 shard 的启用策略。
4. **C4 正确性加固**：保留 DCP2/4/8 kernel 单测，并补 full score、raw index、physical slot 的 distributed graph 测试。
5. **最后评估 head-chunk overlap**：只有 recv-side fusion 和显存压缩后，trace 仍显示 SendRecv 暴露在 critical path，才实现 chunk overlap。

C128 独立 fusion 继续不属于本轮 DCP 优化范围。

本轮产物：

```text
/data/models/dev0616/c4_c128_p0/93a5f85efa/dcp8_128k_timeline/
/data/models/dev0616/c4_c128_p0/9d9edbea06/dcp8_128k_timeline/
/data/models/dev0616/c4_c128_p0/9d9edbea06/dcp8_128k_trace_analysis/
/data/models/dev0616/c4_c128_p0/9d9edbea06/dcp8_128k_e2e/
```

## 18. A2A 定向 Profiling 与 128K TPOT/显存归因（2026-07-20）

### 18.1 公平对照与计时有效性

当前线上函数名 `cp_lse_ag_out_rs` 容易造成误解。它实际执行的是：

1. LSE all-gather；
2. FP32 weighted output all-reduce；
3. 每个 rank 本地切出自己的 head chunk。

它不是 reduce-scatter。本轮在 benchmark 中恢复了一个不进入 production 的真
`AG+RS` 候选，并将三条路径放在相同的 TP8/DCP8 communicator 下比较：

- `AG+AR`：当前 runtime reference；
- `AG+RS`：LSE all-gather + FP32 output reduce-scatter；
- `A2A`：destination-head output A2A + LSE A2A + recv-side merge。

目标 shape 为 `B=128, H=64, D=512`，一个 CUDA Graph 顺序 capture 43 次
attention merge。每条路径使用独立 `torchrun` 进程，避免前一张 graph 的内存池
污染后一条路径；CUDA Event 与全设备 synchronize wall time 的差异均小于 `0.1%`。

| 路径 | Event ms/层 | Wall ms/层 | 43 层 graph wall (ms) | Graph reserved delta |
|---|---:|---:|---:|---:|
| 当前 AG+AR | 0.197906 | 0.197918 | 8.510 | 0.039 GiB |
| 真 AG+RS | 0.164134 | 0.164159 | 7.059 | 0.055 GiB |
| A2A | 0.195437 | 0.195449 | 8.404 | 0.059 GiB |

正确性相对当前 reference：

- AG+RS output max abs diff：`4.77e-7`；
- A2A output max abs diff：`7.15e-7`；
- 两条候选的 LSE max abs diff 均为 `0`。

### 18.2 问题一：为什么 A2A 比真 AG+RS 慢

8 rank x 4 graph replay 的 GPU-only trace 中：

| 指标 | AG+RS (ms/graph) | A2A (ms/graph) | A2A delta |
|---|---:|---:|---:|
| Graph span | 7.186 | 8.479 | +1.293 (+18.0%) |
| Kernel sum | 6.698 | 5.807 | -0.890 (-13.3%) |
| NCCL AllGather | 0.596 | 0 | -0.596 |
| NCCL ReduceScatter | 2.830 | 0 | -2.830 |
| NCCL SendRecv | 0 | 2.244 | +2.244 |
| FP32 reduce128 family | 0.372 | 0.680 | +0.308 |

关键现象是 A2A 的 **kernel sum 更低，但 graph span 更长**：

- AG+RS 的 `span - kernel_sum` 约为 `0.488 ms`；
- A2A 的 `span - kernel_sum` 约为 `2.672 ms`；
- A2A 多出约 `2.184 ms` 的未覆盖间隙，抵消了 `0.890 ms` 的 kernel work 节省，
  最终反而慢 `1.293 ms`。

当前 A2A 每层发起两次 `all_to_all_single`，43 层对应 86 个 NCCL
`SendRecv` kernel。第二次 LSE A2A 每个 destination 只有约 4 KiB，是纯 latency-bound
collective；output A2A 还要求 destination-major pack，并在 recv 侧增加 FP32 weighted
sum。相比之下，reduce-scatter 把传输与求和放进一个专用 NCCL ring-LL collective，
没有 recv-side output reduction。

Profiler 中 A2A 的 `cudaGraphLaunch` API duration 也接近整个 graph span，而 AG+RS
只有约 `0.64 ms`。该 API 数字受 profiler 影响，不能单独当作性能结论，但与 P2P graph
依赖导致 launch/stream 间隙的判断一致。

因此应区分两个结论：

- 对真 AG+RS：A2A 在该 shape 下慢约 `18-19%`；
- 对当前 AG+AR：A2A 的 standalone graph 快约 `1.3%`，整模型 B128 graph 快
  `3.72%`，因为它移除了较慢的 43 次 FP32 custom all-reduce。

43 层 graph 的 batch sweep：

| Batch | AG+AR ms/层 | AG+RS ms/层 | A2A ms/层 | A2A vs AG+RS | A2A vs AG+AR |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0460 | 0.0482 | 0.0560 | +16.3% | +21.7% |
| 8 | 0.0548 | 0.0550 | 0.0598 | +8.7% | +9.2% |
| 32 | 0.0853 | 0.0787 | 0.1073 | +36.4% | +25.8% |
| 64 | 0.1184 | 0.1112 | 0.1630 | +46.6% | +37.7% |
| 128 | 0.1979 | 0.1641 | 0.1954 | +19.1% | -1.3% |

这也解释了之前 B64 加速比下降。B64 的 output chunk 约为 512 KiB/peer，P2P 固定
间隙尚未被有效摊薄；B128 达到约 1 MiB/peer 后，A2A 才能追平当前 AG+AR。A2A 不应
按 DCP size 单独启用，至少还要使用 batch bucket 与可用显存做 gating。

### 18.3 问题二：为什么 128K TPOT 大幅增加

使用无 profiler、pretokenized fixed B128、output 256 的同 server sweep。计时从最后
一个请求收到首 token 后开始，排除了 shared-prefix admission 与 residual prefill：

| Prefix | Median TPOT (ms) | 相对 3.5K delta | Output tok/s |
|---:|---:|---:|---:|
| 3.5K | 44.137 | - | 2900.04 |
| 16K | 45.058 | +0.921 (+2.1%) | 2840.79 |
| 64K | 48.739 | +4.601 (+10.4%) | 2626.25 |
| 128K | 54.589 | +10.452 (+23.7%) | 2344.78 |

已由 128K trace 直接量到的首要线性项是冗余 C4 indexer：

- 21 层 full-history C4 score：`5.208 ms`；
- 21 层 C4 top-k/merge：`0.934 ms`；
- 合计约 `6.142 ms`。

启用 C4 shard 后，score 下降到 `0.771 ms`、top-k/merge 下降到 `0.580 ms`，二者
合计节省 `4.792 ms`；candidate collective 增加后，整 graph 净节省 `3.630 ms`，
无 profiler output512 E2E 净节省约 `3.249 ms`。

因此 128K TPOT 增量可拆成：

1. **已证明的主要项**：C4 full-history score/top-k 随上下文增长，128K 时约占
   `6.14 ms`；
2. **待 paired trace 精确拆分的剩余项**：C128 compressed-history attention、相关
   metadata/compaction、page-table 访问以及 graph 间隙随历史增长；
3. **固定底座**：即使 3.5K 仍有约 `44 ms`，来自 43 层 GEMM、MoE/DeepEP、
   attention merge 和其他固定 decode kernel，绝对 TPOT 并不是全部由 128K 引入；
4. **普通 serving benchmark 的额外噪声**：rolling arrival、cache admission、batch
   retraction 和 TTFT 混入会把涨幅进一步放大。固定 batch 结果应作为上下文斜率主判据。

A2A 的 output/LSE payload 仅由 `B x H x D` 决定，与 128K context 无关，因此它不是
128K TPOT 上升的来源。要把剩余约 4 ms 继续闭环，应补 3.5K/16K/64K/128K 的 paired
GPU-only graph trace，并增加 C128 attention/metadata kernel family。

### 18.4 问题三：为什么 A2A 增加约 3.5 GB/GPU

整模型 capture 日志：

| 变体 | Graph memory | Capture 后 available | PyTorch default pool | PyTorch graph pool |
|---|---:|---:|---:|---:|
| AG+AR | 8.43 GB | 9.34 GB | 75.477 GiB | 0.486 GiB |
| A2A | 11.93 GB | 5.84 GB | 75.477 GiB | 0.504 GiB |

两边在第一个 B128 graph 前的 available 都约为 `17.80 GB`。第一个最大 graph capture
完成后：

- AG+AR available 为 `10.10 GB`；
- A2A available 为 `6.65 GB`；
- `3.45 GB` 差异已经一次性出现，后续小 bucket 只复用最大 pool。

这证明增量：

- 出现在 B128 CUDA Graph capture，而不是 128K KV cache；
- 与 C4 packed candidate 无关；
- 也不是普通 PyTorch tensor pool 的 3.5 GB 增长。8 rank runtime allocator snapshot 中，
  代表性普通 rank 的 default pool 完全相同，graph pool 只增加约 `18 MiB`；
- standalone 43 层 merge graph 中，A2A 相对 AG+AR 也只多约 `20 MiB` reserved。

当前 A2A 每层显式临时量的量纲约为：

- BF16 output send/recv：`8 + 8 MiB`；
- recv `nan_to_num`：约 `8 MiB`；
- FP32 weighted output：约 `16 MiB`；
- FP32 local output 与 LSE 临时量：约 `2 MiB`；
- 合计约 `42 MiB/层`，43 层约 `1.76 GiB`。

`1.76 GiB` 若因 NCCL side stream 生命周期、P2P staging 或 capture 双份地址资源而接近
两份，量纲正好接近观测的 3.5 GB。但 allocator snapshot 证明普通 tensor 本身并未以
这个规模常驻，所以“哪里发生双份”仍属于推断，不能写成已证明事实。

进一步排除实验：

- `NCCL_GRAPH_REGISTER=0`：整模型仍为 `11.93 GB`，capture 每个 bucket 的 available
  与默认 A2A 完全一致；
- `NCCL_GRAPH_HELPER_DISABLE=1`：standalone reserved 不变，A2A 反而从 `0.1954`
  轻微变慢到 `0.1991 ms/层`。

当前最严格的结论是：**超过 99% 的 A2A 增量位于 PyTorch allocator snapshot 之外，
由最大 B128 graph 中 86 个 NCCL SendRecv 节点及其跨 stream/capture 资源触发；它不是
NCCL user-buffer graph registration 开关能够消除的内存。** 要定位到具体 `cudaMalloc`
调用，需要下一步使用 CUDA API memory trace 或在 ProcessGroupNCCL allocator 上加计数。

### 18.5 下一步 Attention 优化顺序

1. 将真 AG+RS 接入 runtime 临时开关，跑 full-model graph memory 与 A-B E2E。当前
   standalone B128 比 A2A 快约 `19%`，也没有 A2A 的 P2P graph 间隙。
2. 保留 A2A 默认关闭。若继续优化，先把 output 与 bit-preserving LSE 合为一次 packed
   A2A，将 86 个 SendRecv 降到 43 个，再测 graph span 与外部显存。
3. 为 A2A 使用 backend-owned graph-stable workspace，显式建立跨 NCCL stream 的复用
   依赖；不能在未证明生命周期安全时直接让 43 层别名到同一 buffer。
4. 补 paired context trace，拆清 C128 compressed-history 与 metadata 的剩余斜率。
5. 只有 A2A 的 graph gap 与显存收敛后，再评估 head-chunk compute/communication overlap。

本轮产物：

```text
/data/models/dev0616/c4_c128_p0/c3e4766409/a2a_profile/
/data/models/dev0616/c4_c128_p0/c3e4766409/full_model_profile/
```
