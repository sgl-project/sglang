# 第 14 章 投机解码：EAGLE、MTP 与 DFlash

## 14.1 为什么 decode 慢

自回归解码每步只生成一个 token，但整张模型都要过一遍。GPU 每步的算力利用率很低，时间主要花在"把权重搬一遍"上。**投机解码（Speculative Decoding）**的思路：先用一个小而快的 draft 模型一次猜出 k 个 token，再用大模型一次前向**验证**这 k 个 token——猜对的部分白赚，猜错则回退。期望上每步产出 >1 个 token，decode 吞吐显著提升。

## 14.2 SGLang 里的算法家族

`--speculative-algorithm` 参数（`server_args.py` 第 1996 行）接受的内置算法：

| 算法 | 实现目录 | 说明 |
| --- | --- | --- |
| EAGLE / EAGLE3 | `speculative/eagle_worker_v2.py` 等 | 基于"特征层"的草稿模型，EAGLE-2 引入自适应 top-k |
| NEXTN | `speculative/` 相关 worker | 原生多 token 预测头（如 Gemma 2 的 nextn） |
| MTP | `speculative/mtp_*` | DeepSeek 的多 token 预测训练配套 |
| NGRAM | `speculative/cpp_ngram` | 用 n-gram 匹配 prompt 做草稿，无需额外模型 |
| STANDALONE | `speculative/base_spec_worker.py` | 无草稿，验证路径（调试用） |
| DFlash / DSPARK | `dflash_*`、`dspark_components` | 新一代免训练投机方案（DFlash 是 2026 年 README 主打的 Spec V2） |

算法可通过 `SpeculativeAlgorithm.register` 扩展，说明这是一个开放注册表。

## 14.3 架构：草稿与验证 worker

投机解码在 SGLang 里是**双 worker**结构：

```text
Scheduler
├── target worker（大模型，ModelRunner）
└── draft worker（草稿模型，独立 ModelRunner / CUDA graph）
    └── 与 target 共享 KV 池或独立池（按算法而定）
```

- `scheduler.py` 的 `maybe_init_draft_worker` 负责拉起草稿 worker；
- `speculative/draft_worker_common.py`、`eagle_worker_common.py` 提供公共逻辑；
- 验证逻辑在 `speculative/` 下的 `draft_utils.py` 与各算法 worker 中：一次前向同时算草稿序列的所有位置，逐个对比 target logits 的 argmax。

## 14.4 与 CUDA graph / KV cache 的耦合

投机解码是 SGLang 里耦合最深的特性之一：

- 草稿序列的 KV 也走 radix cache（EAGLE 的 draft KV 有专门缓存）；
- `eagle_draft_cuda_graph_runner.py` / `eagle_draft_extend_cuda_graph_runner.py`：草稿模型也有 CUDA graph；
- `adaptive_spec_params.py` / `adaptive_runtime_state.py`：EAGLE-2 的自适应 top-k，根据接受率动态调整草稿长度；
- 与 PD 分离配合时，prefill 侧不产生草稿 KV，decode 侧用无效 sentinel 标记（`disaggregation` 代码注释中有说明）。

## 14.5 什么时候收益最大

- 草稿模型够小、够快（接受率 0.6~0.9 时收益明显）；
- decode 阶段访存受限严重（长上下文、小 batch 尤为明显）；
- 多 LoRA 场景需谨慎：草稿模型不认识 LoRA 适配器时接受率会掉。

## 14.6 基准测试

`benchmark/bench_adaptive_speculative.py` 是自适应投机解码的专用 benchmark，`examples/` 下也有 EAGLE 相关示例（`examples/runtime/speculative/`）。README 里 2026 年的 DFlash / Spec V2 博客是了解最新进展的入口。

## 14.7 本章小结

- 投机解码 = 小模型猜 + 大模型验证，白赚正确猜测。
- SGLang 支持 EAGLE/MTP/NEXTN/NGRAM/DFlash 等多种算法，双 worker 架构统一承载。
- 与 CUDA graph、KV cache、PD 分离的耦合使它成为"进阶中的进阶"话题。
- 收益取决于接受率与草稿速度，部署前务必用 benchmark 实测。
