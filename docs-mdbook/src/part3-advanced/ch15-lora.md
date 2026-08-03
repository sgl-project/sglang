# 第 15 章 多 LoRA：显存账、动态加载与踩坑

## 15.1 先算显存账

LoRA 不复制基座模型，每个适配器只多一份低秩增量。显存公式：

```text
单适配器增量显存 ≈ Σ(输入维 × 秩 × 2) × 2（权重 + 梯度/优化器，仅训练时） × 精度系数
```

推理时更简单：`Σ 层数 × (in_dim + out_dim) × rank × 2 bytes`（fp16）。rank=64 的适配器通常只有几十 MB——所以**一个 8B 基座 + 几十个 LoRA 适配器，显存远小于"几十个独立模型"**，这就是多租户服务的经济性来源。

真正的显存大头不是 LoRA 增量本身，而是：

1. **KV 缓存不能共享**：不同 LoRA 的请求即使文本相同，注意力行为不同，KV 必须隔离（第 8 章的 `extra_key`）；
2. **CUDA graph 空间**：每个 LoRA 的 graph 缓冲（`LoRAManager.init_cuda_graph_batch_info` 按 LoRA 维度预分配）。

## 15.2 LoRAManager：适配器的一生

`lora/lora_manager.py` 的 `LoRAManager`（第 59 行）管四件事：

| 职责 | 方法 | 说明 |
| --- | --- | --- |
| 注册 | `load_lora_adapter` / `load_lora_weights` | 权重进统一 buffer，按 `LoRARef` 注册 |
| 形状管理 | `init_lora_shapes` | 预分配各 rank 的显存 |
| 批内调度 | `layers.py` | batch 里不同请求按 `lora_id` 取对应增量 |
| graph 适配 | `init_cuda_graph_batch_info` | 为 LoRA 请求准备 CUDA graph 信息 |

`lora/mem_pool.py` 管增量的显存池；`lora/backend/` 是不同算子的 LoRA kernel（含 Marlin 量化 LoRA）。

## 15.3 动态加载：上线与下线的工程

生产上适配器会持续增删，SGLang 提供了三个机制：

1. **在线加载**：`POST /load_lora_adapter`（`entrypoints/http_server.py`），不重启服务；
2. **重叠加载**（`--enable-lora-overlap-loading`）：新适配器在 GPU 空闲窗口预加载，不阻塞推理——`lora_overlap_loader.py`；
3. **优雅下线**：`lora_drainer.py` 停止接收新请求，等存量请求跑完再卸载，避免"跑到一半适配器没了"。

显存不足时的淘汰策略在 `lora/eviction_policy.py`：先淘汰最久没用、占用最大的适配器，使用中的请求优先保住。

## 15.4 与 MLA / 量化的化学反应

LoRA 和"省显存技术"叠加时经常出问题，代码里专门有修正：

- `lora/deepseek_mla_correction.py`：MLA 的投影被压缩成 latent，LoRA 增量不能直接加，需要特殊合并——这是 DeepSeek + LoRA 部署最常见的坑；
- `lora/lora_moe_runner_marlin.py`：量化（Marlin）权重 + LoRA 的融合 kernel，不是所有量化格式都支持 LoRA。

结论：**选 LoRA 方案前先确认目标模型架构（MLA？）和量化格式的组合是否被支持**，否则要么不接受，要么需要专门代码。

## 15.5 容量规划速算

```text
可承载适配器数 ≈ (可用显存 - KV 预算) / 单适配器增量显存
```

但两个修正项：

- 适配器是**按请求动态加载**的，不是全量常驻：`lora_max_num` / eviction 策略决定同时驻留多少；
- 每个运行中的 LoRA 请求都要占用一个"LoRA buffer 槽位"，并发 LoRA 请求数也有限制。

所以实际公式是：**同时活跃的 LoRA 数 × 单适配器显存 ≤ 预算**，而不是"注册总数 × 单适配器"。

## 15.6 踩坑清单

| 现象 | 原因与对策 |
| --- | --- |
| 输出与单适配器部署不一致 | KV 隔离失效（extra_key 没生效）；检查 `--enable-lora` 与 `lora_path` 传参 |
| 加载新适配器时卡顿 | 走了同步加载；开 `--enable-lora-overlap-loading` |
| 显存突增 | 并发 LoRA 请求数超过预期；调 eviction 策略 |
| DeepSeek 模型 + LoRA 结果错误 | MLA 修正未生效；确认使用 `deepseek_mla_correction` |
| 投机解码 + LoRA 接受率暴跌 | draft 不认识适配器；考虑对 LoRA 请求关闭投机 |

## 15.7 本章小结

- LoRA 的显存优势来自共享基座，但 KV 隔离与 graph 缓冲会吃掉一部分"省下的钱"。
- 生产关键是动态加载三件套：在线注册、重叠加载、优雅下线。
- MLA/量化与 LoRA 的组合需要专门支持，选型前先确认。
- 容量规划按"同时活跃数"算，不是按注册总数算。

> 下一章跳出 Python，看 Rust 组件在系统里的角色与演进逻辑。
