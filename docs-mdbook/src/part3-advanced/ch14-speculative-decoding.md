# 第 14 章 投机解码：原理、收益、工程代价

## 14.1 先算一笔账：为什么 decode 的 GPU 用不满

decode 阶段每步只处理 1 个新 token，但必须把整个模型的所有权重读一遍（访存）。假设模型权重 16GB、GPU 带宽 2TB/s，理论上一遍权重要 8ms——而真正算一个 token 的 FLOPs 可能只需要不到 1ms。**GPU 在等权重搬运，算力闲着。**

投机解码的思路：让一个小模型（draft）先快速猜 k 个 token，大模型（target）一次前向**同时验证**这 k 个位置。验证时：

- 每个位置都和 draft 的猜测一致 → 白赚，一次前向产出 k+1 个 token；
- 第一个不一致的位置 → 从那里回退，这个位置用 target 自己的输出。

每步期望产出：

```text
E[tokens per step] = 1 + α + α² + ... + αᵏ   （α = 接受率，draft 每个位置被 target 认可的概率）
```

接受率 0.7、k=4 时，期望产出约 2.7 个 token/步，比不投机多 ~2.7 倍——这就是投机解码的收益上限。

## 14.2 SGLang 支持的算法家族

`--speculative-algorithm`（`server_args.py:1996`）：

| 算法 | 草稿怎么来 | 特点 |
| --- | --- | --- |
| EAGLE / EAGLE3 | 用 target 倒数第二层特征做输入的小模型 | 质量高，接受率通常最好 |
| MTP | 训练时一起训的多 token 预测头 | DeepSeek 系模型自带 |
| NEXTN | 原生多 token 预测（如 Gemma） | 模型自带，无需额外训练 |
| NGRAM | 从 prompt 里找 n-gram 片段做草稿 | 无需草稿模型，代码在 `speculative/cpp_ngram` |
| DFlash / DSPARK | 免训练方法（2026 年的 Spec V2） | 无需额外模型，成本低 |

选型核心看三点：**有没有现成草稿模型、接受率多高、草稿前向多贵**。草稿太贵会把省下的时间吃回去。

## 14.3 工程结构：双 worker

投机解码在代码里是**两个 worker**：

```text
Scheduler
├── target worker（大模型，正常 ModelRunner）
└── draft worker（草稿模型，独立 ModelRunner / 独立 CUDA graph）
```

`scheduler.py` 的 `maybe_init_draft_worker` 拉起草稿 worker；`speculative/draft_worker_common.py` 提供公共逻辑。验证发生在 target 一次前向里：draft 序列的所有位置一起算，逐个和 target logits 的 argmax 比较。

## 14.4 耦合点：为什么它是"进阶中的进阶"

投机解码不是独立模块，它和三大系统深度耦合：

1. **CUDA graph**：draft 有自己的 graph（`eagle_draft_cuda_graph_runner.py`），target 验证步的 graph 也要按"接受/拒绝"动态调整；
2. **KV cache**：draft 序列的 KV 也走 radix cache（否则每次重新算 draft 前向就白费了）；
3. **PD 分离**：P 实例不产生 draft KV，D 实例用 sentinel 标记"这里是投机起始"——注释里明确写了这个耦合。

任何一处没对齐，投机要么不生效，要么**生成结果与 target 单独跑不一致**（这是正确性事故）。

## 14.5 自适应：EAGLE-2 的动态 top-k

接受率不是固定的：prompt 难，接受率就低。EAGLE-2 的做法（`speculative/adaptive_spec_params.py`）：

```text
接受率高 → 下次多猜几个（k 调大）
接受率低 → 下次少猜几个（k 调小）
```

运行时根据最近几轮的接受率自适应，避免"猜了 4 个只对 1 个，反而比不猜更慢"。

## 14.6 实测与调参

启动后怎么看收益：

1. 日志里的 accept rate / 每步产出 token 数（`--log-level info` 或 metrics）；
2. `--speculative-eagle-topk` 控制 EAGLE 的草稿长度上限；
3. 换模型/换 prompt 分布后，接受率会明显变化，**不要拿一个 benchmark 的结果当所有场景的结论**。

失效模式速查：

| 现象 | 可能原因 |
| --- | --- |
| 接受率极低（<0.3） | 草稿模型与 target 不匹配 / prompt 太难 / 量化降低了 target 与 draft 的一致性 |
| 吞吐反而下降 | draft 前向太慢，k 太大；调小 k 或换算法 |
| 与不加投机结果不一致 | draft/target 的采样参数不同步；版本耦合处出错 |
| 多 LoRA 场景退化 | draft 不认识 LoRA 适配器，接受率暴跌 |

## 14.7 本章小结

- 收益上限由接受率决定：α=0.7、k=4 时约 2.7x。
- 选算法看"草稿质量 vs 草稿成本"，EAGLE 质量高但需要草稿模型，NGRAM/DFlash 免训练但接受率看场景。
- 双 worker + CUDA graph + KV cache + PD 的耦合，让它成为正确性风险最高的特性。
- 上线前必须实测接受率，且按 prompt 分布单独评估。

> 下一章：多 LoRA 的显存账和动态管理。
