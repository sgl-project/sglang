# 附录 G：SGLang 生产监控（Prometheus + Grafana）设计文档

> 用途：给已有 Prometheus/Grafana 平台的管理员对照检查接入项。SGLang 原生暴露 Prometheus 格式 `/metrics`（需启动参数 `--enable-metrics`，脚本默认已带），无需额外 exporter。

## 1. 架构

```
客户端 ─► 代理 8080（限并发+priority） ─► SGLang 8000（/metrics）
                                              │
                                              ▼
                       ┌─── scrape ────┐
                       │  Prometheus   │
                       └──────┬────────┘
                              ▼
                          Grafana（面板 + 告警）
```

采集路径二选一（都从 pod 外）：

| 路径 | 抓取地址 | 说明 |
|------|---------|------|
| K8s Service + pod 注解 | `http://<service-ip>:8000/metrics` | 推荐；`scrape` 到 SGLang 端口（8000），不是代理端口（8080 也能透传 `/metrics`，但建议直连后端） |
| 直接抓 Service | `http://<service-ip>:8000/metrics` | 最简；需 Service 暴露 8000 |

> 直方图指标名以实际 `/metrics` 输出为准（不同版本 bucket 定义可能微调）。上线后先 `curl http://<service-ip>:8000/metrics | grep '^sglang:'` 核对一遍再配面板。

## 2. 关键指标清单

### 2.1 容量与负载（Gauge）

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `sglang:num_queue_reqs` | 排队请求数（带 `priority="10"` / `priority="0"` 标签拆分；`priority=""` 为总量） | **核心**：>0 持续上涨 = 饱和前兆 |
| `sglang:num_running_reqs` | 运行中请求数 | 对比 `max-running-requests × DP`（36） |
| `sglang:token_usage` | KV 池占用率（0~1） | ≥0.92 触发驱逐风暴 |
| `sglang:kv_available_tokens` | 可分配 KV token | 逼近 0 = 缓存压力大 |
| `sglang:gen_throughput` | 生成吞吐 tok/s | 对比不同配置（如 723 vs 112） |
| `sglang:cache_hit_rate` | 前缀缓存命中率 | 基线 ~43%；下降说明缓存被挤爆 |

### 2.2 延迟（Histogram，`stream` 标签区分流式/非流式）

| 指标 | 含义 | 面板建议 |
|------|------|----------|
| `sglang:time_to_first_token_seconds` | TTFT（首 token 延迟） | p50 / p90 / p99，`stream="true"` |
| `sglang:inter_token_latency_seconds` | ITL（token 间延迟） | p50 / p90，配合 MTP `spec_accept_length` 看 |
| `sglang:e2e_request_latency_seconds` | 端到端延迟 | p50 / p90 |
| `sglang:queue_time_seconds` | 排队时间（不含执行） | TTFT 变差时先看它分清排队 vs prefill |

> **限制**：TTFT/ITL/E2E 直方图**不按 priority 拆分**（只有 model/tp_rank/stream 等标签），无法直接区分 tool call 与 thinking 的延迟。近似手段：代理按类型限并发，`num_queue_reqs{priority="10"}` 反映 tool call 排队；tool call 普遍走流式，看 `stream="true"` 即可。

### 2.3 MTP 效果（Gauge）

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `sglang:spec_accept_rate` | 投机接受率 | <70% 说明 draft 质量差或 batch 干扰 |
| `sglang:spec_accept_length` | 平均接受长度 | ~3.4 对应约 3.4x decode 加速 |
| `sglang:spec_verify_calls_total` | verify 调用次数（Counter） | 观察 verify 开销占比 |

### 2.4 质量（Counter）

| 指标 | 含义 | 告警建议 |
|------|------|----------|
| `sglang:num_aborted_requests_total` | 中止请求数 | 增速快 = KV 紧张/TTFT 超时 |
| `sglang:num_requests_total` | 总请求数 | 用于计算 abort 率 |

## 3. Prometheus 抓取配置

### 3.1 ServiceMonitor / 注解方式（推荐）

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sglang
  namespace: your-ns
spec:
  selector:
    matchLabels:
      app: sglang          # 按你 Service 的实际标签改
  endpoints:
    - port: http           # Service 中暴露 8000 的端口名
      path: /metrics
      interval: 15s
      scrapeTimeout: 5s
      honorLabels: true
```

### 3.2 手动抓取配置（无 Operator）

```yaml
scrape_configs:
  - job_name: sglang
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["10.0.0.5:8000"]   # Service IP:8000
        labels:
          instance: sglang-6g
```

> 抓 SGLang 后端（8000）而非代理（8080）：代理对 `/metrics` 是透传，但直连后端少一跳、少一层故障面。若集群只暴露 8080，抓 `http://<ip>:8080/metrics` 同样有效（已透传验证）。

### 3.3 采集自检

```bash
# 确认目标 UP
curl -s 'http://<prometheus>/api/v1/targets' | jq '.data.activeTargets[] | {labels: .labels.job, health}'
# 确认指标存在
curl -s 'http://<prometheus>/api/v1/query' --data-urlencode 'query=sglang:num_queue_reqs' | jq '.data.result[0].value'
```

## 4. Grafana 面板设计（按评估目标分组）

### 4.1 面板一：容量水位（一眼看饱和）

```promql
# 排队（核心）
sum(sglang:num_queue_reqs{priority=""})                       # 总量
sum(sglang:num_queue_reqs{priority="10"})                     # tool call 排队
# 运行与容量
sglang:num_running_reqs{priority=""}
sglang:token_usage
sglang:kv_available_tokens
```

判定：`token_usage` 长时间 ≥0.92 + `kv_available` 贴 0 = KV 瓶颈；`queue` 持续 >5 = 限流/容量瓶颈。

### 4.2 面板二：TTFT 分位数（开 adaptive 前后对比核心）

```promql
histogram_quantile(0.50, sum(rate(sglang:time_to_first_token_seconds_bucket{stream="true"}[5m])) by (le))
histogram_quantile(0.90, sum(rate(sglang:time_to_first_token_seconds_bucket{stream="true"}[5m])) by (le))
histogram_quantile(0.99, sum(rate(sglang:time_to_first_token_seconds_bucket{stream="true"}[5m])) by (le))
```

同时叠 `sglang:queue_time_seconds`（分位数）区分"排队慢" vs "执行慢"。

### 4.3 面板三：MTP 健康

```promql
sglang:spec_accept_rate
sglang:spec_accept_length
rate(sglang:spec_verify_calls_total[5m])
```

### 4.4 面板四：吞吐与缓存

```promql
sglang:gen_throughput
sglang:cache_hit_rate
rate(sglang:num_aborted_requests_total[5m]) / rate(sglang:num_requests_total[5m])   # abort 率
```

## 5. 告警规则建议

```yaml
groups:
  - name: sglang
    rules:
      - alert: SGLangQueueBuildUp
        expr: sum(sglang:num_queue_reqs{priority=""}) > 10
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "排队请求数持续 >10，检查限流/容量"
      - alert: SGLangKVBottleneck
        expr: sglang:token_usage > 0.92
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "KV 池占用 >92%，可能触发驱逐风暴"
      - alert: SGLangTTFTSlow
        expr: histogram_quantile(0.90, sum(rate(sglang:time_to_first_token_seconds_bucket{stream="true"}[5m])) by (le)) > 10
        for: 10m
        labels: { severity: critical }
        annotations:
          summary: "TTFT p90 >10s"
      - alert: SGLangAbortRate
        expr: rate(sglang:num_aborted_requests_total[5m]) / rate(sglang:num_requests_total[5m]) > 0.05
        for: 10m
        labels: { severity: warning }
        annotations:
          summary: "中止率 >5%"
      - alert: SGLangMTPDegraded
        expr: sglang:spec_accept_rate < 0.65
        for: 15m
        labels: { severity: info }
        annotations:
          summary: "MTP 接受率偏低，投机收益下降"
```

> 阈值按你实际基线调：排队 10、TTFT p90 10s、abort 5% 为初始值，跑 1~2 天基线后校准。

## 6. 基线对比与 adaptive 效果评估流程

**阶段一（不加 `--adaptive-limit`，1~2 天）**：

- 记录四个面板的典型值（尤其 queue、TTFT p50/p90、abort 率）；
- 标记一天中的高峰/低谷时段（与业务请求曲线对齐）；
- 用 Grafana annotation 标注发版/重启时间点。

**阶段二（加 `--adaptive-limit`）**：

- 同一 Grafana 时间范围切片对比（同星期同时段最公平）；
- 关注：queue 是否收敛（不再持续堆积）、TTFT p90 是否稳定、tool_call 排队（`priority="10"`）是否下降；
- 代理侧看 pod 日志 `[adaptive-limits] 调整 limits:` 的行，与 Grafana 曲线对照——每次调整后 15~30s 内 queue/TTFT 应出现预期方向的变化；
- 若 tool_call 频繁震荡（放宽→收紧→放宽），调大 `--adaptive-interval`（如 30s）或 `--step` 减到 2。

**判定标准**：

| 结果 | 结论 |
|------|------|
| queue 不再持续 >5，TTFT p90 稳，无震荡 | 可以固化 `ADAPTIVE_LIMIT=true` 到脚本默认 |
| queue 仍堆积，TTFT 仍升 | 后端容量确实到顶，调大 `--adaptive-max-tool-call` 无意义，考虑 HPA 扩副本 |
| 频繁震荡 | 收敛太激进，加 interval / 减 step |

## 7. 补充：轻量 CSV 采集（可选，无 Prometheus 时）

`sglang_start.sh --monitor` 会拉起 [sglang_monitor.py](appendix/sglang_monitor.py)，每 15s 抓 `/metrics` 关键指标（queue、token_usage、TTFT/ITL/E2E 的 p50/p90/p99、spec accept、abort）追加写 CSV：

```bash
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 --monitor
# 输出: /tmp/sglang_monitor/sglang_monitor.csv（pod 内；要持久化请挂载卷）
```

适合：临时压测对比、无 Prometheus 的环境。生产长期监控仍建议走 Prometheus（本附录）。

## 8. 上线检查清单

- [ ] Service 暴露 8000（或 8080），pod 外 `curl <ip>:8000/metrics` 有 `sglang:` 前缀指标
- [ ] Prometheus target UP，`scrape_interval=15s`
- [ ] 四个面板建好：容量水位 / TTFT 分位数 / MTP 健康 / 吞吐缓存
- [ ] 告警规则导入，阈值按基线校准
- [ ] 记录 1~2 天 baseline（阶段一），再开启 `--adaptive-limit`（阶段二）
- [ ] adaptive 震荡时按第 6 节调参；稳定后固化默认值
