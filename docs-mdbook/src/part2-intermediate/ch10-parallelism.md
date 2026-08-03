# 第 10 章 并行策略：TP / EP / DP / PP

## 10.1 为什么需要并行

当模型权重超过单卡显存（如 DeepSeek 671B），或者单卡吞吐不满足要求时，就要把模型/请求切到多卡。SGLang 支持的四种并行各有分工：

| 策略 | 英文 | 切分对象 | 典型场景 |
| --- | --- | --- | --- |
| 张量并行 | TP | 单层内的权重（QKV/O 矩阵按维度切） | 单机多卡放一个大模型 |
| 专家并行 | EP | MoE 层的专家路由到不同卡 | DeepSeek 等大规模 MoE |
| 数据并行 | DP | 请求分发给多份模型副本 | 提高吞吐/水平扩展 |
| 流水线并行 | PP | 网络按层切成多段，段间流水 | 超深网络、跨节点 |

## 10.2 参数如何进入代码

所有并行度都由 `server_args.py` 的 `ServerArgs` 承载：

```text
--tp-size 8 --ep-size 8 --dp-size 4 --pp-size 1 ...
```

还有一个特殊参数 `--nnodes` / `--node-rank`：跨节点时每个节点各起一个进程，通过初始化（`distributed/bootstrap.py`）用 TCP/NCCL 建立全局通信组。

## 10.3 parallel_state：一张全局拓扑图

`distributed/parallel_state.py` 是并行拓扑的心脏。它维护一个 `ProcessGroup` 层级：

```text
global group
├── tp group      (每 tp_size 个 rank 一组)
├── ep group      (MoE 专家组)
├── dp group      (数据并行副本组)
└── pp group      (流水线段)
```

模型层代码通过 `get_tp_group()` / `get_ep_group()` 等拿到对应通信组，然后调用 `communication_op.py` 里的 `tensor_model_parallel_all_reduce`、`all_gather`、`reduce_scatter` 等原语。`parallel_state_wrapper.py` 则提供了运行时动态重建拓扑的能力（弹性场景）。

## 10.4 张量并行（TP）的落地

以 `layers/model_parallel.py` 为例：`RowParallelLinear` / `ColumnParallelLinear` 把权重按维度切开，前向时配合 all-reduce / all-gather。注意力层（`layers/attention/radix_attention.py`）在 TP 下把 head 分到各卡，KV cache 也按 TP 维度分片——这是为什么 KV 池是"每 TP 副本一套"。

## 10.5 MoE 专家并行（EP）

`layers/moe/` 与 `model_executor/model_runner_components/moe_ep_setup.py` 负责专家分配。EP 的核心问题有两个：

1. **路由**：每个 token 的 top-k 专家可能在不同卡上，需要 all-to-all 通信把 token 送到专家所在卡；
2. **负载**：热门专家可能过载，`eplb/` 目录（Expert Parallel Load Balancing）与 `elastic_ep/`（弹性 EP、专家备份）就是解决这个问题的。

DeepSeek 系列的大规模部署（96 H100 等）几乎都依赖 EP + TP 的组合。

## 10.6 数据并行（DP）与 DP Controller

`--dp-size N` 时，`managers/data_parallel_controller.py` 启动一个"请求分发器"：

```text
客户端请求
  → TokenizerManager
  → DP Controller（按负载/缓存感知路由到某个 Scheduler）
  → Scheduler i（每个 Scheduler 是完整模型副本，各有独立 KV cache）
```

DP 是"无共享"的：每个副本独立调度、独立缓存，适合吞吐扩展。SGLang 的路由（第 17 章）本质上就是外部化的 DP Controller。

## 10.7 流水线并行（PP）

`managers/scheduler_pp_mixin.py` 提供 PP 支持：模型按层切段，请求在段间以微批（micro-batch）形式流水推进。`--pp-size` 配合 `--tp-size` 时形成"TP 组内张量并行、组间流水"的经典布局。PP 主要用来解决单卡放不下、TP 通信开销过大的场景（如跨节点）。

## 10.8 通信后端与初始化

- `distributed/bootstrap.py`：多进程/多机初始化，生成 rank 与 group；
- `distributed/communication_tags.py`：定义通信 tag 常量；
- `distributed/naive_distributed.py`：非 NCCL 的朴素实现（调试）；
- `distributed/communication_op.py`：对 `torch.distributed` 的封装，并针对 SGLang 场景做了流管理。

硬件方面，`hardware_backend/` 下按厂商提供初始化差异（如华为 `hccl`、AMD `rccl`、Intel `xccl`），`platforms/` 则封装设备能力探测。

## 10.9 一个典型的大模型部署组合

DeepSeek-V3 级别（671B MoE）的典型部署：TP=8（单机内张量并行）+ EP 覆盖全部专家 + DP 多副本承接流量，必要时 PP 跨机。阅读 `benchmark/deepseek_v3/` 下的部署脚本可以直观感受这些参数如何组合。

## 10.10 本章小结

- 并行策略是正交的：TP 切层内权重、EP 切专家、DP 切请求、PP 切层序列。
- `parallel_state.py` 定义拓扑，`communication_op.py` 提供原语，层代码只管调用。
- DP Controller 是"软路由"，为第 17 章的独立 router 埋下伏笔。
- 生产部署通常是多种并行的组合，参数就在 `server_args.py`。
