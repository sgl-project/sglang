# 第 13 章 Prefill/Decode 分离：为什么、怎么传、什么坑

> 本章不逐行贴代码，而是回答三个生产问题：什么时候该用 PD 分离、传输怎么做、哪些环节会出问题。

## 13.1 先看负载画像：两种阶段根本不是一种活

用数字感受 prefill 和 decode 的差异（以 8B 模型、8 卡 TP 为例的典型量级）：

| | prefill | decode |
| --- | --- | --- |
| 计算形态 | 大矩阵乘（长序列 × 权重） | 小矩阵（1 token × 权重） |
| 瓶颈 | 算力（FLOPs） | 访存带宽（权重/KV 搬运） |
| 单请求时长 | 毫秒级（取决于 prompt 长度） | 每个 token 几十微秒级 |
| 并发特征 | 突发、可批量 | 持续、需要稳定服务 |

混在一起时的问题：一个 10k token 的 prefill 会把 GPU 占住几百毫秒，期间所有在线请求的 TPOT 集体劣化。PD 分离的动机就是**让两种负载各占一摊资源，互不拖累**。

## 13.2 架构与角色

`python/sglang/srt/disaggregation/utils.py` 定义了两种角色：

```python
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
```

部署形态：

```text
客户端 → P 实例（prefill 集群）
          ├─ 算完整段 prompt，产出 KV Cache
          └─ KV 通过网络传给 D 实例
D 实例（decode 集群）
          ├─ 接收 KV，挂到自己显存
          └─ 从断点继续自回归，流式返回
```

容量上最大的变化：**P 和 D 可以按 1:N 配比独立伸缩**。长 prompt 多就加 P，并发高就加 D。

## 13.3 传输：KV 怎么搬家

这是 PD 最"硬"的部分。`--disaggregation-transfer-backend` 可选多个后端（`server_args.py` 中列出）：

| 后端 | 特点 | 典型场景 |
| --- | --- | --- |
| native/mooncake_tcp | 走 TCP，部署简单 | 实验、小规模 |
| mooncake | RDMA 优先、TCP 兜底 | 大规模 RDMA 集群 |
| nixl / mori | 各自硬件生态的传输栈 | 特定厂商环境 |

工程上的分工（对应 `disaggregation/` 子目录）：

- **metadata**：KV 在哪、序列多长、版本号——走控制通道（bootstrap 服务）；
- **数据**：真正的 K/V 张量——走高速传输后端，可先落 host 内存再异步搬上 GPU（`decode_kvcache_offload_manager.py`）；
- **事件**：KV 就绪通知用订阅/发布（`kv_events.py`），decode 实例"等通知再拉取"，而不是轮询。

## 13.4 协调：bootstrap 与配对

请求进入时带 `bootstrap_host/port/room/pair_key`（`GenerateReqInput` 字段）。prefill 完成后：

```text
P 实例 → bootstrap 服务：登记"这条 KV 已就绪 + 元数据"
D 实例 → bootstrap 服务：查到自己的配对请求，发起拉取
```

这套"房间 + 配对键"设计是为了多实例并发下不错配：每个请求有唯一 pair key，只有配对的 P/D 才会握手。

## 13.5 失效模式：哪里会挂，怎么处理

生产上 PD 比单实例多了一整类故障：**跨进程/跨机器的传输**。代码里甚至留了故障注入钩子（`utils.py` 的 `_get_failure_prob`），说明可靠性是被显式设计的。常见失效与对策：

| 失效 | 症状 | 对策 |
| --- | --- | --- |
| KV 传输失败 | decode 侧等不到 KV，请求挂起 | 超时 + 重传；解码端超时后把请求退回 prefill |
| P 实例崩溃 | 已登记的 KV 永远不来 | bootstrap 记录带心跳，过期清理 |
| 配对错乱 | 拿到的 KV 长度和请求对不上 | pair_key + 长度/哈希校验 |
| 版本不一致 | 模型权重更新后 P/D 行为不同 | 权重版本号随 metadata 传递，不匹配即拒绝 |
| 传输慢于 prefill | decode 侧空闲等待 | 流水化：P 边算边传（分块发送，`maybe_send_cached_prefix_chunk`） |

最后一条是性能关键：**不要把"全部算完再传"**，而是算一段传一段，让传输和计算重叠。

## 13.6 什么时候不该用

PD 分离不是银弹，以下场景别用：

- 单机低负载：传输开销（序列化 + 网络 + 二次加载）大于收益；
- 短 prompt 短输出：prefill 本来就快，拆分徒增运维复杂度；
- 无独立网络/存储：KV 传输对带宽和延迟敏感，共享网络容易成为瓶颈。

判断标准一句话：**只有当 prefill 和 decode 各自都达到瓶颈、且互相拖累时，才值得拆。**

## 13.7 容量规划速算

```text
需要的 D 实例数 ≈ 峰值并发请求数 × 平均输出长度 / 单实例 decode 吞吐
需要的 P 实例数 ≈ 每秒新增请求数 × 平均 prompt 长度 / 单实例 prefill 吞吐
```

两者互相独立，这就是 PD 最大的运营优势：**瓶颈在哪边，就只加哪边**。

## 13.8 本章小结

- PD 分离把计算密集与访存密集拆开，P/D 独立伸缩。
- KV 传输 = metadata（控制通道）+ 数据（高速后端）+ 事件通知，三者分工。
- 可靠性问题集中在传输：超时、重试、配对、版本一致性都要处理。
- 使用前提是 prefill 与 decode 真的互相拖累；短请求场景别用。

> 下一章：投机解码——用"小模型猜、大模型验证"换 decode 吞吐。
