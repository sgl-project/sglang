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
