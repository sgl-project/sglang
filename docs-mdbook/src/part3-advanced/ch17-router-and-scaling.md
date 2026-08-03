# 第 17 章 路由与集群扩展：sgl-router 与多实例部署

## 17.1 从单机到集群的跳跃

单机 SGLang 能扛的并发有限；集群化要回答三个问题：

1. **请求分给谁**（负载均衡）；
2. **怎么让相同前缀落在同一实例**（缓存亲和）；
3. **实例挂了怎么办**（健康检查与摘除）。

`experimental/sgl-router/` 是 SGLang 对这些问题的回答：一个 Rust 编写的、KV-aware、OpenAI 兼容的路由器。README 开篇即定义身份：

> "Slim, KV-aware, OpenAI-compatible router for SGLang workers. Serves a single model and routes across its workers."

## 17.2 sgl-router 的能力

- 暴露 `/v1/tokenize`、`/v1/detokenize`、`/v1/models`、`/v1/chat/completions`（缓冲与 SSE）、`/healthz`、`/readyz`、`/metrics`；
- worker 池来自静态 URL 列表或 Kubernetes EndpointSlice 自动发现；
- 单模型（`--model-id` 必填），可选本地 `tokenizer.json`，否则从 HuggingFace 下载。

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

PD 分离场景用 `--prefill-selector` / `--decode-selector` 分别发现 prefill 和 decode 组。

## 17.3 路由策略：从轮询到 KV 感知

`experimental/sgl-router/src/policies/` 是算法仓库：

| 策略 | 文件 | 思想 |
| --- | --- | --- |
| Round Robin / Random | `round_robin.rs` / `random.rs` | 基线策略 |
| Load Based | `load_based.rs` | 按 worker 当前负载 |
| Power of Two | `power_of_two.rs` | 抽样两节点选负载低的 |
| Sticky | `sticky.rs` | 相同会话/请求粘到同一 worker |
| Cache Aware | `cache_aware_zmq.rs` | 订阅 KV 事件，命中缓存前缀的 worker 优先（第 8 章的 RadixCache 分布式化） |

KV-aware 是核心卖点：router 通过 `kv_events/`（ZMQ 订阅 worker 的缓存事件，维护一棵前缀树索引）知道每个 worker 缓存了什么，把请求路由到"已缓存其前缀"的实例，减少跨实例重复 prefill。

## 17.4 健康检查与熔断

`health/mod.rs` 与 `circuit_breaker.rs` 实现探活与熔断：持续失败的 worker 被摘出候选池，恢复后重新加入。`workers/manager.rs` 管理 worker 注册表与状态。

## 17.5 另一种形态：DP Controller 与 Engine 内路由

不引入独立 router 时，SGLang 自带两种集群能力：

- `--dp-size N` 的 `data_parallel_controller.py`：进程内的数据并行分发；
- 多节点 `--nnodes N --node-rank i`：每个节点一个服务进程，前置负载均衡器（Nginx/LB）分发。

router 的定位是**独立于引擎、跨实例**，适合 K8s 环境；DP Controller 是引擎内部能力，适合单机多卡或小集群。

## 17.6 生产部署建议（结合代码）

- 实例间共享 tokenizer 路径，router 配置 `--tokenizer-path` 保证路由层与引擎层分词一致；
- 启用 KV 事件订阅时注意带宽（`kv_events` 事件流在超大集群上是额外负载，`subscriber.rs` 做了过滤）；
- 与 PD 分离组合：router 同时面向 prefill 与 decode 组，配 `--prefill-selector`/`--decode-selector`；
- 监控用 `/metrics`（Prometheus 格式）接入现有监控体系。

## 17.7 本章小结

- sgl-router 是 KV-aware 的 OpenAI 兼容路由层，解决"请求分给谁"与"缓存亲和"。
- 策略可插拔，从轮询到 cache-aware 一应俱全，带健康检查与熔断。
- 集群扩展的三条路：独立 router、引擎内 DP Controller、外部 LB + 多实例。
- 下一章把视角收回到单实例：如何系统性地做性能调优。
