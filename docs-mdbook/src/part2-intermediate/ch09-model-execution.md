# 第 9 章 模型执行代码走读：ForwardBatch、Attention Backend 与 CUDA Graph

> 代码来自 `python/sglang/srt/model_executor/` 与 `python/sglang/srt/layers/`。

## 9.1 三个角色的分工

GPU 侧执行涉及三个对象，先分清谁是谁：

| 对象 | 位置 | 职责 |
| --- | --- | --- |
| `ModelRunner` | `model_runner.py:246` | "GPU worker"：持有模型、KV 池、attention backend |
| `ForwardBatch` | `forward_batch_info.py:412` | 一次前向的**全部输入**（契约） |
| `ModelRunnerOutput` | `model_runner.py:238` | 一次前向的**全部输出** |

调度器负责"决定跑什么"，ModelRunner 负责"执行"，两者通过 ForwardBatch 解耦。

## 9.2 ForwardBatch：一次前向的完整描述

`ForwardBatch` 的关键字段（读的时候按分组理解）：

```python
class ForwardBatch(...):
    input_ids: torch.Tensor          # 拼接后的整批 token
    positions: torch.Tensor          # 每个 token 的绝对位置
    req_pool_indices: torch.Tensor   # 每个请求在 req_to_token_pool 的行号
    seq_lens: torch.Tensor           # 每个请求的当前长度
    extend_range: Optional[Range]    # prefill：这批覆盖的 token 区间
    output_ids: torch.Tensor         # decode：这批要算的新 token 位置
    sampling_info: SamplingBatchInfo # 采样参数（批量化）
```

prefill 和 decode 的 ForwardBatch 长得不一样：

- **prefill/extend**：`extend_range` 覆盖请求新增的整段 token，`seq_lens` 各不相同；
- **decode**：每个请求只贡献 1 个新 token，`seq_lens` 全部 +1。

这也是为什么两种阶段各有一条优化路径——它们连"输入长什么样"都不同。

## 9.3 ModelRunner.forward 主干

`forward`（第 1312 行）是入口，剥掉调试/钩子代码后主干是：

```python
def forward(self, forward_batch, ...) -> ModelRunnerOutput:
    self.forward_pass_id += 1
    ...
    with (
        step_span_ctx,
        get_global_expert_distribution_recorder().with_forward_pass(...),
    ):
        output = self._forward_raw(forward_batch, ...)
        if self.enable_elastic_ep:
            output = self._maybe_rebalance_after_rank_fault(...)
    ...
    return output
```

注意这行代码背后的含义：`forward` 同时被 prefill 和 decode 复用。真正分流在 `_forward_raw` 内部（按 `forward_batch.forward_mode` 或 CUDA graph runner 选择路径）。

`forward_split_prefill`（第 1290 行）是一个值得单独看的变体：它把 prefill 按**层数**切块跑（`split_index` 从 0 走到 `num_hidden_layers`），用来压峰值激活显存，配合 chunked prefill 使用：

```python
def forward_split_prefill(self, forward_batch, ...):
    if forward_batch.split_index == 0 or reinit_attn_backend:
        self.attn_backend.init_forward_metadata(forward_batch)
    next_split_index = min(forward_batch.split_index + forward_count,
                           self.model_config.num_hidden_layers)
    ret = self.model.forward_split_prefill(forward_batch.input_ids,
                                           forward_batch.positions,
                                           forward_batch,
                                           (forward_batch.split_index, next_split_index))
    forward_batch.split_index = next_split_index
    return ret
```

## 9.4 Attention Backend：插件化的注意力内核

注意力是推理的算力中心，SGLang 把它做成插件（`layers/attention/attention_registry.py` 注册）。`model_runner.py` 的 `init_attention_backends`（第 850 行）一次性构建：

```python
backends = build_attention_backends(model_runner=self)
self.attn_backend = backends.attn_backend
self.decode_attn_backend = backends.decode_attn_backend
self.prefill_attention_backend_str = backends.prefill_attention_backend_str
```

注意它把 prefill 和 decode 的 backend 分开选（`prefill_attention_backend_str` 与 `decode_attention_backend_str`）——两个阶段访存模式不同，最优内核可能不同。

后端清单本身就是硬件/内核演化史（`layers/attention/`）：

- FlashInfer / FlashAttention / Triton / torch_native（回退调试用）；
- `flashinfer_mla_backend` / `flashmla_backend` / `cutlass_mla_backend`：MLA 专用；
- `dsa_backend` / `nsa_backend` / `minimax_sparse_backend`：稀疏注意力；
- `intel_amx_backend` / `xpu_backend` / `wave_backend`：非 NVIDIA 硬件。

`AttentionBackend` 抽象（`base_attn_backend.py:19`）定义接口，最关键的是 `init_forward_metadata`（第 48 行）：每个 ForwardBatch 在进 attention 层前，backend 要算出"每个 token 的 KV 索引、页表、掩码"等元数据。

## 9.5 CUDA Graph：decode 的"录放机"

decode 每步的 kernel 很小，PyTorch 的 kernel launch 开销（每次几十微秒）反而成了大头。CUDA graph 的思路：**把一整套 kernel 序列录制下来，之后整图回放，CPU 只付一次启动成本**。

`model_executor/runner_backend/` 提供三种形态：

| 形态 | 文件 | 特点 |
| --- | --- | --- |
| 整图回放 | `full_cuda_graph_backend.py` | 经典方案，图内不能有动态形状 |
| 可断点图 | `breakable_cuda_graph_backend.py` | 支持在图中插入变化部分 |
| 分片+compile | `tc_piecewise_cuda_graph_backend.py` | CUDA graph 与 torch.compile 混合 |

关键工程事实：**图是按 batch 大小预录的**。`--cuda-graph-bs` 配置一组覆盖的 batch 大小（如 1,2,4,...,256），请求数不在列表里时走非图路径（慢但正确）。所以 `max_running_requests` 调大后，要同步确认 CUDA graph 覆盖到那个 batch 大小。

## 9.6 采样：从 logits 到 token

模型输出的 logits 交给 `layers/sampler.py` 的 `Sampler.forward`（第 97 行）。主干逻辑：

```python
if sampling_info.is_all_greedy:
    batch_next_token_ids = torch.argmax(logits, -1)      # 贪心：直接取最大
else:
    # 按 sampling_info 应用 temperature / top-p / top-k / min-p
    # 必要时先缓存原始 logprobs（return_logprob 场景）
    ...
    probs = torch.softmax(logits / temperatures, dim=-1)
    batch_next_token_ids = self._sample_from_probs(probs, sampling_info)
```

细节里有两件事值得记住：

1. `is_all_greedy` 是一个**整批级**的判断：如果一批请求全都是贪心采样，直接走 argmax 快路径，不分配概率张量；
2. `SamplingParams`（`sampling/sampling_params.py:45`）是 HTTP 层 dict 的落地类型，字段含 `temperature/top_p/top_k/min_p/json_schema/regex` 等——**结构化约束就挂在采样参数上**，第 11 章接着讲。

## 9.7 自己动手的实验

1. `--disable-cuda-graph` 与默认配置各跑一次 `python -m sglang.benchmark.serving`，对比 decode 吞吐。你会直观看到 CUDA graph 的价值。
2. `--attention-backend torch_native` 跑一次小模型，再换回默认，对比速度（同时观察日志里打印的 backend 选择）。
3. 发一个 64 并发请求，观察 decode 阶段日志，数一数实际 batch 大小；再对照 `--cuda-graph-bs` 配置看哪些 batch 大小走了图。

## 9.8 本章小结

- ForwardBatch 是调度与执行之间的"契约"，prefill/decode 两种形态输入完全不同。
- Attention backend 插件化，prefill/decode 分别选型，是硬件适配的入口。
- CUDA graph 把 decode 的 CPU 启动开销压到近零，但受 batch 大小预录限制。
- Sampler 有整批贪心快路径，结构化约束挂在采样参数上。

> 下一章把单 GPU 放大到多卡：并行策略的代码实现。
