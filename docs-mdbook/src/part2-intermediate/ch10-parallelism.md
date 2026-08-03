# 第 10 章 并行策略代码走读：TP / EP / DP / PP

> 代码来自 `python/sglang/srt/distributed/`（通信）与 `python/sglang/srt/layers/`（并行层）。

## 10.1 四种并行，一张表说清

| 策略 | 切什么 | 通信代价 | 典型场景 |
| --- | --- | --- | --- |
| TP（张量并行） | 单层权重按维度切 | 每层一次 all-reduce | 单机多卡放不下单模型 |
| EP（专家并行） | MoE 的专家分到不同卡 | all-to-all（token 搬家） | DeepSeek 级大 MoE |
| DP（数据并行） | 请求分给多份副本 | 几乎无 | 水平扩展吞吐 |
| PP（流水线并行） | 网络按层切段 | 段间传递激活 | 超深模型跨节点 |

参数入口在 `server_args.py`：`--tp-size`、`--ep-size`、`--dp-size`、`--pp-size`。

## 10.2 parallel_state：一张全局拓扑图

`distributed/parallel_state.py` 的 `ProcessGroup` 定义层级：

```text
global group（所有 rank）
├── tp group    每 tp_size 个 rank 一组
├── ep group    MoE 专家组
├── dp group    数据并行副本组
└── pp group    流水线段
```

层代码不直接调 NCCL，而是问 `get_tp_group()` / `get_ep_group()` 拿通信组，再调 `communication_op.py` 里的原语：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

这一层抽象的价值：模型代码只写"我要 all-reduce"，至于走 NCCL 还是别的后端、在哪个组上做，由 parallel_state 决定。换硬件/换拓扑不用改模型代码。

## 10.3 TP 的实现：ColumnParallelLinear 的 forward

`layers/linear.py` 第 469 行，`ColumnParallelLinear.forward`：

```python
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    # Matrix multiply.
    assert self.quant_method is not None
    output_parallel = self.quant_method.apply(self, input_, bias)
    if self.gather_output:
        # All-gather across the partitions.
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    ...
    return output, output_bias
```

要点：

1. 每个 rank 只持有 `output_size_per_partition` 的输出（权重按输出维度切开）；
2. `gather_output=True` 时（如 attention 的 QKV 投影之前），需要 all-gather 拼回完整输出；
3. `RowParallelLinear` 则相反：每个 rank 算部分和，最后 all-reduce 求和。

一个线性层是"Column + Row"组合：前半段切着算（省显存），后半段合并（保正确），中间一次 all-reduce。这就是 TP 的基本节奏。

## 10.4 EP：专家并行的核心是 all-to-all

MoE 模型的每个 token 会被路由到 top-k 个专家。EP 把专家分到不同卡，于是发生：

```text
每张卡：门控网络算出 token → 专家映射
  → all-to-all：把 token 送到它要去的专家所在卡
  → 专家前向（本地只算自己那部分专家）
  → all-to-all：把结果送回原卡
```

代码入口：`layers/moe/` 的 MoE 层 + `distributed/communication_op.py` 的 all-to-all 原语 + `model_executor/model_runner_components/moe_ep_setup.py` 的专家分配。

EP 的两个经典问题：

1. **负载不均**：热门专家收到的 token 多，冷门专家闲着。`eplb/`（Expert Parallel Load Balancing）用启发式做专家迁移；
2. **弹性**：专家所在的卡挂了，整层不可用。`elastic_ep/` 提供专家备份（backup）与故障后重平衡——`ModelRunner.forward` 里那个 `_maybe_rebalance_after_rank_fault` 就是入口。

## 10.5 DP：引擎内的请求分发

`--dp-size N` 时，`managers/data_parallel_controller.py` 启动一个"分发器"：

```python
# 伪代码（真实实现见 data_parallel_controller.py）
while True:
    recv_reqs = ...                       # 从 TokenizerManager 收请求
    for req in recv_reqs:
        rank = policy.choose_rank(req)    # 负载感知 / 缓存感知选副本
        dispatch(req, rank)
```

每个 DP 副本是一整套独立世界（独立 Scheduler + 独立 KV cache + 独立 GPU），所以 DP 的扩展性最好，代价是缓存不共享——相同前缀的请求如果被分到不同副本，就各算各的。**这也是第 17 章"缓存感知路由"要解决的问题**。

## 10.6 PP：层的流水线

`managers/scheduler_pp_mixin.py` 提供 PP 支持。模型按层切段，请求以微批（micro-batch）在段间流水推进。PP 的代码形态和其他并行不太一样：它主要出现在**调度器与执行器的协作**上（段与段之间传递激活，调度器要协调"哪个段跑哪个微批"），而不是层内部的通信。

## 10.7 组合：大模型部署的典型姿势

DeepSeek 级别（671B MoE）的典型组合：

```text
TP=8（单机内张量并行，放得下一层）
EP=全部专家跨卡（MoE 层）
DP=多副本（承接流量）
PP=跨机（可选，单机放不下时）
```

对应到代码，就是 `parallel_state` 里同时存在多个 group，层代码各自取用。读 `benchmark/deepseek_v3/` 下的启动脚本，能看到这些参数的真实组合。

## 10.8 自己动手的实验

1. 同一个小模型分别用 `--tp-size 1` 和 `--tp-size 2` 启动，看启动日志的显存分配差异（TP 把权重和 KV 都切了）。
2. `--dp-size 2` 启动，发 20 个相同 prompt 的请求，观察日志中两个副本各收到多少（体会"缓存不共享"）。
3. 读 `distributed/bootstrap.py`，理解多机启动时 rank 是怎么分配的。

## 10.9 本章小结

- 四种并行切的东西不同，通信模式也完全不同。
- `parallel_state` 定义拓扑，`communication_op` 提供原语，模型层只写"要什么通信"。
- TP 靠 Column/Row 拆分 + all-reduce；EP 靠 all-to-all + 负载均衡；DP 靠分发器；PP 靠微批流水。
- DP 的代价是缓存不共享，这是路由层要解决的命题。

> 下一章看采样与结构化输出：从概率分布到"必须合法的 token"。
