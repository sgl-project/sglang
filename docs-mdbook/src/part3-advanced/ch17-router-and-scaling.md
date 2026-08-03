# 第 17 章 集群路由与容量规划：从单机到多实例

## 17.1 单机瓶颈在哪

单实例能扛的并发有限：显存（KV 预算）和 GPU 算力都有上限。水平扩展 = 多份实例 + 一个"分派请求"的层。分派层要回答三个问题：

1. 请求分给谁（负载均衡）；
2. 相同前缀的请求能不能落在同一实例（缓存亲和）；
3. 实例挂了怎么摘除（健康检查）。

## 17.2 方案矩阵：三条路

| 方案 | 实现 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 外部 LB（Nginx/ALB） | 轮询/最小连接 | 通用、零改造 | 不懂缓存，前缀亲和靠运气 |
| 引擎内 DP | `--dp-size N` 的 `data_parallel_controller.py` | 零额外组件 | 只在一个进程组内，不支持跨机器发现 |
| 独立 router | `experimental/sgl-router` | KV 感知 + K8s 原生 | 多一个组件要运维 |

## 17.3 sgl-router 的能力与策略

sgl-router 是一个 Rust 编写的、OpenAI 兼容的 KV-aware 路由器（README 原话："Slim, KV-aware, OpenAI-compatible router for SGLang workers"）。它暴露 `/v1/chat/completions`（含 SSE）、`/healthz`、`/readyz`、`/metrics`，worker 池来自静态 URL 或 K8s EndpointSlice 发现。

路由策略（`experimental/sgl-router/src/policies/`）从朴素到聪明：

| 策略 | 思想 | 适用 |
| --- | --- | --- |
| round-robin / random | 均匀分发 | 无缓存亲和需求的基线 |
| power-of-two | 随机抽两个 worker 选负载低的 | 通用且便宜 |
| load-based / active-load | 按实时负载 | 请求长度方差大 |
| sticky | 同一会话粘同一 worker | 多轮会话 |
| **cache-aware** | 订阅 KV 事件，把请求路由到"已缓存其前缀"的 worker | 前缀复用场景（推荐） |

## 17.4 缓存亲和的收益：一个直觉计算

假设 8 个 worker、请求前缀相同率为 50%：

- 无亲和：请求随机落点，命中缓存的概率 ≈ 1/8，约 87.5% 的请求要重复 prefill；
- 有亲和：相同前缀的请求大概率落在同一个 worker，第二次起全部命中。

对长 prompt（如 4k token）场景，这意味着 prefill 量减少约一个数量级。**在长上下文、多轮、共享 system prompt 的场景，缓存亲和比"更快的负载均衡"值钱得多。**

## 17.5 健康检查、熔断与发现

`health/mod.rs` + `circuit_breaker.rs`：

- 周期性探测 worker 的 `/healthz`；
- 连续失败 → 熔断（摘出候选池），恢复后重新加入；
- `workers/manager.rs` 维护 worker 注册表。

K8s 集成：`discovery/k8s.rs` 用 EndpointSlice 自动发现（`--service-discovery`），PD 分离场景用 `--prefill-selector` / `--decode-selector` 分别发现两组 worker。

## 17.6 集群容量规划速算

```text
实例数 = max(
    峰值并发 × 平均输出 token / (单实例 decode 吞吐 × 目标利用率),
    每秒新请求 × 平均 prompt token / 单实例 prefill 吞吐
)
```

三个容易被低估的项：

1. **利用率系数**：别按 100% 算，留 30~50% 给突发和滚动发布；
2. **KV 预算**：长上下文会吃掉大量显存，`mem_fraction_static` 决定每实例并发上限；
3. **前缀命中率**：命中率高时 prefill 需求大幅下降，容量可以更小——这也是为什么"缓存命中率"值得作为核心监控指标。

## 17.7 本章小结

- 水平扩展的钥匙是"缓存亲和"，不是单纯负载均衡。
- 三条路：外部 LB（简单但盲目）、引擎内 DP（零组件但范围小）、sgl-router（KV 感知、K8s 原生）。
- 策略从轮询到 cache-aware 可插拔；健康检查与熔断保证摘除。
- 容量规划按 prefill/decode 两项分别算，再乘利用率系数。

> 下一章把调优从"参数表"变成"方法论 + 案例"。
