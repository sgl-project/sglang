# 第 9 章 模型执行：ModelRunner、Attention Backend 与 CUDA Graph

## 9.1 执行层的三个角色

GPU 侧的执行由 `model_executor/` 承担，核心角色：

1. **ModelRunner**（`model_runner.py` 第 246 行）：调度器里的"GPU worker"，持有模型实例、KV 池引用、attention backend，提供 `forward()` 接口；
2. **ForwardBatch**（`forward_batch_info.py` 第 412 行）：描述"这一批跑什么"——input ids、位置、KV 位置、采样参数等；
3. **模型实现**（`models/` 与 `layers/`）：真正的 PyTorch 网络。

```python
class ModelRunner:
    def forward(self, forward_batch, ...) -> ModelRunnerOutput:
        ...
        output = self._forward_raw(forward_batch, ...)
```

`ModelRunnerOutput`（`model_runner.py` 第 238 行）把 logits、新 token、logprobs 等打包返回给调度器。

## 9.2 ForwardBatch：一次前向的全部输入

`ForwardBatch` 是"调度器决策"与"GPU 执行"之间的契约，关键字段包括：

- `input_ids`：拼接后的整批 token；
- `positions`：每个 token 的绝对位置；
- `req_pool_indices` / `seq_lens`：请求在 `req_to_token_pool` 中的索引与长度；
- `kv_indices` / `page_kv_indices`：KV 页索引；
- `sampling_info`：采样相关（温度、top_p、logprobs 需求）；
- `return_logprob`、`return_hidden_states` 等输出选项。

prefill 与 decode 的 ForwardBatch 形态不同：prefill 的 `extend_range` 覆盖整段 prompt，decode 每请求只推一个 token。

## 9.3 注意力 Backend：插件化设计

注意力是推理的算力中心，SGLang 把它做成插件式（`layers/attention/attention_registry.py` 注册）。后端列表本身就是一部硬件/内核演化史：

| 后端 | 说明 |
| --- | --- |
| `flashinfer_backend.py` | FlashInfer（默认主力之一） |
| `flashattention_backend.py` | FlashAttention |
| `triton_backend.py` | Triton 实现 |
| `torch_native_backend.py` | 纯 PyTorch（调试/回退用） |
| `flashinfer_mla_backend.py` / `flashmla_backend.py` | MLA 专用内核 |
| `cutlass_mla_backend.py` / `aiter_backend.py` | 针对特定硬件的 MLA/GEMM 优化 |
| `dsa_backend.py` / `nsa_backend.py` / `minimax_sparse_backend.py` | 稀疏注意力模型 |
| `intel_amx_backend.py`、`xpu_backend.py`、`wave_backend.py` | 非 NVIDIA 硬件适配 |

选择逻辑在 `model_runner.py` 的 `init_attention_backend` 与 `layers/attention/base_attn_backend.py`：按模型类型（MHA/MLA）、设备、是否量化自动挑。`--attention-backend` 参数可强制指定。

## 9.4 CUDA Graph：把调度开销压到零

PyTorch 每次 kernel launch 都有毫秒级 CPU 开销，而 decode 阶段每步 kernel 本身可能只有几十微秒——CPU 成了瓶颈。CUDA graph 把一整套 kernel 序列"录制"下来，之后整图回放，CPU 开销趋近于零。

`model_executor/runner_backend/` 提供了多种 graph 形态：

- `full_cuda_graph_backend.py`：整图回放（经典方案）；
- `breakable_cuda_graph_backend.py`：可断点图（支持在图中插入可变部分）；
- `tc_piecewise_cuda_graph_backend.py`：分片 + torch.compile 混合；
- `cuda_graph_dedup_mixin.py`：graph 去重。

图是按 batch size 预录的：`cuda_graph_bs` 配置决定覆盖哪些 batch 大小，请求数超出时走非图路径（慢一些但正确）。`--disable-cuda-graph` 可一键关闭排查问题。

## 9.5 torch.compile 与编译管线

`compilation/` 目录封装了 torch.compile 的集成：`compile.py` 定义编译流程，`pass_manager.py` / `inductor_pass.py` 挂载自定义 pass，`cuda_piecewise_backend.py` / `npu_piecewise_backend.py` / `xpu_piecewise_backend.py` 对应不同硬件的分片编译策略。这是 SGLang 对 PyTorch 2.x 编译生态的拥抱，也是新硬件移植时最需要熟悉的模块之一。

## 9.6 采样与 Logits 处理

前向输出的 logits 不会直接 argmax，而是经过 `layers/logits_processor.py`（处理 logprob 收集、范数等）和 `layers/sampler.py`：

1. 应用采样参数（temperature/top_p/top_k/min_p…）；
2. 应用约束（grammar 掩码、JSON schema、EOS 处理）；
3. 采样出下一 token；
4. 收集请求所需的 logprobs（`logprob_processor.py`）。

`sampling/sampling_params.py` 的 `SamplingParams` 就是 HTTP 层传入 dict 的落地类型。

## 9.7 MoE 与模型特化

`layers/moe/` 提供 MoE 层的 GEMM 封装（支持 DeepGEMM、Marlin 等），`layers/rotary_embedding/` 提供位置编码，`models/` 下每个模型一个文件（`models/llama.py`、`models/qwen2.py`、`models/deepseek_v3.py`…）。新模型接入的路径在第 21 章贡献指南里展开，核心是：写 `models/xxx.py` 实现 forward + 注册 config + 依赖 `layers/` 复用注意力/线性层。

## 9.8 前向的两种形态

- **Prefill/Extend**：一次处理请求新增的整段 token，走 `forward_extend` 路径，计算密集，通常配大 batch 的 GEMM；
- **Decode**：每请求每步 1 个 token，走 CUDA graph 回放，访存密集。

`model_runner.py` 里能看到这两种形态共享 `forward()` 接口，但内部对 `ForwardBatch` 的组装、注意力元数据初始化、graph 路径选择都不同。`forward_split_prefill`（第 1290 行）还支持把 prefill 拆成多层子段（与 chunked prefill 配套，降低峰值激活）。

## 9.9 本章小结

- ModelRunner 是 GPU worker，ForwardBatch 是"这步跑什么"的完整描述。
- 注意力实现是插件化的，MLA/稀疏/异构硬件各有专属后端。
- CUDA graph 把 decode 的 CPU 启动开销压到接近零，是吞吐的关键。
- torch.compile、MoE GEMM、采样器共同构成执行层的其余拼图。
- 下一章把单 GPU 的世界放大到多卡：TP/EP/DP/PP 并行。
