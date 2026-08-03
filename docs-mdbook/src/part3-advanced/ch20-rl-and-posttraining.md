# 第 20 章 RL 与后训练：推理引擎的第二战场

## 20.1 RL 需要推理引擎干什么

强化学习（RLHF / GRPO / RLVR）的每一轮：

```text
训练器更新权重 → 用新权重对一批 prompt 生成 rollout → 打分/规则校验
→ 结果喂回训练器 → 更新权重 → 循环
```

"用新权重批量生成"就是推理引擎的活。verl、AReaL、Miles 等都接 SGLang，不是因为它能"聊天"，而是因为它能**高吞吐地批量生成 + 随时换权重 + 返回训练需要的中间量**。

## 20.2 与在线服务的三个本质差异

| | 在线服务 | RL rollout |
| --- | --- | --- |
| 请求形态 | 单条、交互 | 万级 prompt 一次灌入 |
| 权重 | 固定 | **每轮都变** |
| 需要的输出 | 文本 | 文本 + logprobs + hidden states |

这三个差异决定了 SGLang 为 RL 专门做的设计。

## 20.3 形态：进程内 Engine

RL 框架用的是 `sgl.Engine`（`examples/runtime/engine/launch_engine.py`）：

```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
llm.generate("What is the capital of France?")
llm.shutdown()
```

Engine 不开 HTTP 端口，直接在调用方进程里起 Scheduler/ModelRunner 子进程。对训练框架的价值：**少一层 HTTP 序列化，且能拿到 logprobs/hidden states 等训练输入**。

## 20.4 换权重：训练与推理共享参数

每轮权重更新是 RL 最刚性的需求。SGLang 提供多条路径：

```text
HTTP：POST /update_weights_from_disk（或 _from_distributed / _from_tensor）
Engine API：update_weights_from_tensor / update_weights_from_distributed
```

实现要点（`srt/weight_sync/` + `model_runner_components/weight_updater.py`）：

- 新权重先到 rank 0，再广播到 TP 各组（避免每张卡各自拉文件）；
- 更新前要**暂停调度**（`model_update_lock`），更新后恢复——不能让"半新半旧"的权重跑请求；
- 与 CUDA graph 的耦合：graph 里引用的是权重 buffer 的地址，更新必须"原地写"或重录 graph。

## 20.5 训练要的中间量

| 中间量 | 参数 | 用途 |
| --- | --- | --- |
| logprobs | `return_logprob` | PPO/GRPO 的损失项 |
| hidden states | `return_hidden_states` | value head、KL 项 |
| 批量打分 | `/v1/score`、`EngineScoreMixin` | reward model |

`tokenizer_manager_score_mixin.py` 的 `score_prompts` 展示了打分请求的形态：返回 `scores: List[List[float]]`（每个 prompt 一组分数），可附带 pooled hidden states。

## 20.6 显存共址：sleep/wake 机制

RL 训练和 rollout 常常共享 GPU：训练器要用显存时，推理引擎先"睡"，训练完再"醒"。SGLang 提供：

```text
POST /release_memory_occupation   # 释放 KV 与权重显存（保留进程）
POST /resume_memory_occupation    # 恢复
```

配合 `--enable-memory-saver`（TorchMemorySaver）：释放显存时**保持虚拟内存地址不变**，恢复后 CUDA graph 还能直接回放，不用重录。这是"训练/推理共址"可行性的关键，也是 docs 里明确标注的 RL 专属特性。

## 20.7 确定性：训练与推理必须一致

RL 训练最大的坑之一：训练时前向和 rollout 时前向结果不一致（比如 dropout、非确定性 kernel），导致策略估计失真。SGLang 提供 `--deterministic-inference`：

- 固定采样种子（`sampling_seed`，按位置派生）；
- 使用确定性 kernel；
- 禁用有损优化路径。

代价是性能下降，**只在需要严格一致性时开**。

## 20.8 集成架构速查

```text
训练框架（verl/AReaL/...）
  ├─ 拉起 N 个 sgl.Engine（进程内）
  ├─ engine.generate(prompts, return_logprob=True, return_hidden_states=...)
  ├─ 训练器算损失、更新权重
  ├─ engine.update_weights_from_distributed(...)
  └─ 下一轮
```

长跑生产的注意点：

- watchdog（`utils/watchdog.py`）守护子进程，崩了自动重启；
- 权重更新期间的请求排队/暂停要有策略，别让 rollout 和更新互相饿死；
- 大规模 rollout 用 router 做缓存亲和（同 prompt 重复采样的场景收益明显）。

## 20.9 本章小结

- RL 推理需求 = 万级批量 + 快速换权重 + logprobs/hidden states + 显存共址。
- Engine 形态与在线权重更新 API 是训练框架选它的关键。
- memory-saver 的"释放但保地址"让共址成为可能。
- 确定性推理是训练正确性的最后一道闸。

> 最后一章：如果你想参与这个项目，从哪下手、怎么保证质量。
