# 第 20 章 RL 与后训练：SGLang 作为 Rollout 引擎

## 20.1 为什么训练框架需要推理引擎

强化学习（如 RLHF、GRPO、RLVR）每轮要：

1. 让当前策略模型对大量 prompt 生成 rollout（推理）；
2. 用 reward 模型打分或规则校验；
3. 把结果喂回训练器更新权重；
4. 循环。

这个"生成 rollout"的环节就是推理引擎的舞台。verl、AReaL、Miles、slime、Tunix 等框架都接 SGLang，正是因为它的**离线引擎（Engine）**形态和**在线权重更新**能力。

## 20.2 离线 Engine：进程内推理

训练框架最常用的是进程内 `sgl.Engine`（`examples/runtime/engine/launch_engine.py`）：

```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
llm.generate("What is the capital of France?")
llm.shutdown()
```

Engine 不监听 HTTP，直接在调用方进程里起 Scheduler/ModelRunner 子进程。对应实现：`python/sglang/srt/entrypoints/engine.py` 的 `Engine` 类与 `EngineBase.py` 的接口。典型 RL 用法是"一个训练步骤内批量 generate + 批量拿 logprobs/hidden states"。

## 20.3 权重更新：训练和推理共享参数

RL 每轮都要把新权重灌进推理引擎。`entrypoints/http_server.py` 提供 `/update_weights_from_disk`、`/update_weights_from_distributed` 等端点，`managers/tokenizer_manager.py` 里有对应的 mixin；`srt/weight_sync/` 与 `model_executor/model_runner_components/weight_updater.py` 实现分布式权重同步。

同时，Engine 也直接暴露 `update_weights_from_tensor` / `update_weights_from_distributed` 等 API（`io_struct.py` 中的 `UpdateWeightsFromTensorReqInput` 等），训练框架可以不经 HTTP 直接更新。

## 20.4 打分与奖励

- reward 模型：`examples/runtime/reward_model.py` 演示；服务端 `/v1/score` 端点（`entrypoints/openai/serving_score.py`）+ `EngineScoreMixin`（`engine_score_mixin.py`）支持批量打分；
- hidden states：`return_hidden_states` 让 rollout 拿到每 token 表示（`ReturnHiddenStatesMode`），供 KL 项或 value head 使用；
- logprobs：`return_logprob` 返回各 token 概率，RL 损失的必要输入。

`tokenizer_manager_score_mixin.py` 里的 `score_prompts` 展示了一次批量打分请求的完整形态（`scores: List[List[float]]` + 可选 pooled hidden states）。

## 20.5 针对 RL 的工程特性

`docs_new/docs/advanced_features/sglang_for_rl.mdx` 总结了 SGLang 为 RL 做的专门设计：

- **细粒度 Engine 睡眠/唤醒**：`/release_memory_occupation`、`/resume_memory_occupation` 释放并恢复显存，避免每轮重启；配合 `--enable-memory-saver`（TorchMemorySaver 保持 CUDA graph 地址）让"训练/推理共享 GPU"成为可能；
- **Refit 功能**：多样化的训练/推理共址或分离方案（`srt/multiplex/`、`session/` 相关代码）；
- **生成暂停/控制**：让训练器能控制 rollout 的推进节奏（`EngineBase` 的 pause/resume）；
- **确定性推理**：`--deterministic-inference`（`srt/configs/` 有对应配置）保证训练与推理行为一致，避免"训练时和 rollout 时结果不同"；
- **KV-aware 路由**：大规模 rollout 时用 router 把相同 prompt 路由到缓存实例（第 17 章）。

## 20.6 与训练框架的集成点速查

| 需求 | SGLang 能力 |
| --- | --- |
| 批量生成 | `Engine.generate` / `/generate`（批请求） |
| 拿 logprob | `return_logprob` / `logprob_start_len` |
| 拿 hidden states | `return_hidden_states` |
| 打分 | `/v1/score`、reward model |
| 换权重 | `/update_weights_from_*`、Engine tensor 接口 |
| 暂停/恢复 | pause/resume、memory saver |
| 确定性 | `--deterministic-inference` |
| 长时间运行 | 健康检查、watchdog（`utils/watchdog.py`） |

## 20.7 本章小结

- RL 场景的推理需求 = 高吞吐 batch 生成 + logprob/hidden states + 快速换权重 + 显存共址。
- 离线 Engine 与在线权重更新接口是 SGLang 被训练框架广泛选用的关键。
- 睡眠/唤醒、确定性推理等特性体现了"为 RL 专门设计"的深度。
- 下一章回到工程本身：如何参与这个项目。
