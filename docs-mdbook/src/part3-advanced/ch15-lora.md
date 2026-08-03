# 第 15 章 多 LoRA：参数高效微调的批量服务

## 15.1 LoRA 与多 LoRA 服务

LoRA 通过在权重旁挂低秩增量（`W + BA`）实现参数高效微调。一个服务同时承载多个 LoRA 适配器（每个适配器面向一个下游任务/租户）就是 Multi-LoRA serving——问题在于：**每个适配器都有自己的一套增量权重和前缀，怎么在同一个 batch 里高效混合跑？**

SGLang 的做法：`--lora-paths path1,path2,...` 一次性加载多个适配器，`--enable-lora` 显式开启（`server_args.py` 第 2720 行），请求通过 `lora_path` / `lora_id` 指定用哪个。

## 15.2 核心管理类

`lora/lora_manager.py` 的 `LoRAManager`（第 59 行）负责：

- **注册与加载**：`load_lora_adapter` / `load_lora_weights` 把适配器权重加载进统一的 LoRA buffer；
- **shape 管理**：`init_lora_shapes` 预分配各 rank 的 buffer；
- **批内调度**：batch 里不同请求用不同 LoRA 时，模型层按 LoRA id 取对应增量；
- **CUDA graph 适配**：`init_cuda_graph_batch_info`、`init_prefill_cuda_graph_batch_info` 为 LoRA 请求准备 graph 信息。

`lora/mem_pool.py` 管理 LoRA 权重的显存池，`lora/layers.py` 提供带 LoRA 的线性层实现，`lora/backend/` 下是不同算子的 LoRA kernel（如 Marlin 量化 LoRA：`lora_moe_runner_marlin.py`）。

## 15.3 前缀隔离：extra_key 的用武之地

第 8 章提到 `RadixKey` 的 `extra_key` 命名空间：不同 LoRA 的请求即使文本相同，也**不会共享 KV 前缀**（因为 LoRA 改变注意力行为）。`lora/lora_registry.py` 的 `LoRARef` 会为每个适配器生成唯一标识，调度器把它写进请求的 `extra_key`。这是"正确性优先于复用"的典型设计。

## 15.4 动态加载与重叠

生产场景适配器会动态增删：

- `lora_drainer.py`：把不再使用的适配器优雅下线（先停新请求，等存量跑完）；
- `lora_overlap_loader.py` + `--enable-lora-overlap-loading`：新适配器在 GPU 空闲窗口预加载，不阻塞推理；
- 在线接口：`/load_lora_adapter`、`/unload_lora_adapter`（`entrypoints/http_server.py` 中有对应端点）。

## 15.5 与 MLA / 稀疏注意力的配合

`lora/deepseek_mla_correction.py` 处理 MLA 模型的 LoRA 修正——MLA 的投影矩阵被压缩，LoRA 增量需要特殊合并；`lora/eviction_policy.py` 决定显存不足时先淘汰哪个适配器。这些文件说明：LoRA 不是"挂在模型外面"的简单插件，而是深入到了注意力与显存层的特性。

## 15.6 示例与验证

```bash
python examples/runtime/lora.py
```

`benchmark/lora/` 目录有专门的 benchmark；`test/` 下有 LoRA 的正确性测试（包括与 baseline 输出对比）。

## 15.7 本章小结

- Multi-LoRA = 一个基座模型 + 多个增量适配器 + 批内按请求取用。
- `LoRAManager` 管加载与 buffer，`extra_key` 保证前缀隔离，overlap loader 保证动态性。
- MLA/稀疏注意力/量化的适配器都有专门代码，是深度集成而非外围功能。
- 下一章转向工程形态：Rust 组件在服务中的角色。
