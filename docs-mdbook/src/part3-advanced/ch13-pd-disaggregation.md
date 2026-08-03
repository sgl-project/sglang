# 第 13 章 Prefill/Decode 分离（PD Disaggregation）

## 13.1 动机：两种负载，两套最优解

prefill 是**计算密集**（大 GEMM、长序列注意力），decode 是**访存密集**（逐 token 从 KV 里捞数据）。两者混在一台机器上会互相拖累：

- 长 prefill 挤占 decode 的 GPU，导致所有在线请求 TPOT 变差；
- decode 的 KV 缓存占用巨大，压缩了 prefill 可用的显存。

PD Disaggregation（prefill/decode 分离）的解法是：**用独立的 prefill 实例和 decode 实例分别服务两种阶段**，prefill 算完的 KV Cache 通过网络转给 decode 实例，decode 只做自回归生成。

## 13.2 代码视角：两种角色

`disaggregation/utils.py` 的枚举定义了角色：

```python
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
```

启动时用 `--disaggregation-mode prefill|decode` 指定角色，调度器会相应加载 `disaggregation/prefill.py`（发送 KV）或 `disaggregation/decode.py`（接收 KV）的 mixin。

## 13.3 传输链路

一次 PD 请求的协作过程：

```text
客户端 → prefill 实例
  ├─ prefill 计算，产出 KV Cache
  └─ KV 通过传输后端送给 decode 实例
decode 实例
  ├─ 接收 KV，挂到自己的显存池（decode_kvcache_offload_manager）
  ├─ 从预填充位置继续自回归 decode
  └─ 流式返回结果给客户端
```

传输后端由 `--disaggregation-transfer-backend` 指定（`server_args.py` 列出 mooncake、nixl、mori、mooncake_tcp 等），对应 `disaggregation/mooncake/`、`nixl/`、`mori/` 目录。KV 事件的订阅/发布（`kv_events.py`）让 decode 实例能感知"KV 已就绪"并去拉取。

## 13.4 协调：bootstrap 与配对

请求进入时带 `bootstrap_host/port/room/pair_key`（`GenerateReqInput` 字段）。prefill 完成后通过 bootstrap 服务（`disaggregation/utils.py` 的 bootstrap 相关逻辑）通知配对好的 decode 实例"KV 在哪、metadata 是什么"，decode 侧再发起传输。`encode_server.py` / `encode_grpc_server.py` 还支持 encoder 分离（`--encoder-only` 路径，从 `launch_server.py` 可以看到）。

## 13.5 传输细节与优化

- **metadata 与 KV 分离传输**：hidden states / KV 走高速后端（RDMA），元数据走控制通道；
- **offload**：decode 实例的 KV 可以先进 host 内存再异步搬上 GPU（`decode_kvcache_offload_manager.py`）；
- **失败重试**：`utils.py` 里有 `_get_failure_prob` 等测试钩子，说明 PD 链路的健壮性是被显式测试的；
- **HiCache**：`disaggregation` 与 `mem_cache/hicache_storage.py` 配合，KV 可以存到远端存储系统，实现"存储级缓存"。

## 13.6 什么时候该用

适用场景：

- 长上下文、大并发在线服务（TTFT 与 TPOT 都要保）；
- DeepSeek 这类超长 prompt 的模型（如 GB200 集群部署）；
- 推理成本敏感、需要按 prefill/decode 分别伸缩容量（不同机器配比）。

不适用场景：

- 单机、低负载、短 prompt（传输开销大于收益）；
- 所有请求都一次性短回答（无 decode 瓶颈）。

## 13.7 本章小结

- PD 分离把"算前缀"和"续写"拆给不同机器，各自优化。
- 角色由 `DisaggregationMode` 表达，传输由可插拔后端承载。
- 正确性依赖 bootstrap 配对 + KV 事件通知 + 失败重试。
- 生产上通常配合 DP 副本和独立 router（第 17 章）一起使用。
