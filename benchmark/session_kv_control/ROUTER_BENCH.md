# Router B2/B3 实验

`router_bench.py` 比较相同准备目标 B 下的直连恢复与 Router 自由选路。两臂都使用 Router 的 `task_started → request_observed → tool_wait → resume_expected`；B2 正式请求直达 B，B3 正式请求带 `kv_context` 发到 Router。`worker_url=B` 只指定准备节点，正式请求不带 worker 指定字段。原 `session_kv_bench.py --router` 继续只采集能力。

脚本使用现有服务。Router 必须配置 P6 orchestration、普通 `cache_aware` 路由与两个健康 Engine；通过 `/metrics` 的 `sgl_router_worker_health` 核验实际 worker URL 集合恰为 A/B。Engine 模型、layout、namespace、storage backend namespace、page size、容量和负载配置要一致。可执行能力、配置及 Router cost slope 写入 manifest 和原始控制响应。

运行前，在环境中设置 Router 配置对应的 control bearer。脚本参数只接受环境变量名，raw 不记录 Authorization 请求头，不在请求体中指定 Router caller。普通 Router API 凭证与 Engine API 凭证分别使用可选的 `--router-api-key-env`、`--engine-key-env`。

```bash
/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/.venv/bin/python \
  benchmark/session_kv_control/router_bench.py \
  --router-mode b2-b3 --router http://127.0.0.1:12910 \
  --router-control-key-env SGL_ROUTER_CONTROL_BEARER \
  --engine-a http://127.0.0.1:51801 \
  --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --output /tmp/sglang-rfc-build/bench-router-smoke \
  --prefix-lengths 2048 --tool-wait-s 3 \
  --repetitions 1 --output-tokens 8 --negative route_change
```

扩大长度矩阵时可用 `--prefix-lengths 2048,8192,16384 --repetitions 6`。`--trace-jsonl FILE` 复用已有本地轨迹 schema 和来源哈希，记录中的 `tool_wait_s` 覆盖命令行等待。该入口由 Router 的 cost prior/EMA 选择准备提前量，不采用 JSONL 中的 `lead_s`。默认每种输入在 A/B 单独 warmup 一次，warmup 使用新 salt 并排除出 trial 统计。输出目录内不能已有 `router-raw.jsonl`。

每条 trial 从冷态 A 生成准确的源输入，确认完整 L3 发布后再次调用普通 `/flush_cache` 清 A 本地，并刷新 A 的 epoch/context。该普通 flush 保留共享 Mooncake；脚本不调用清空 storage 的接口。准备前逐个核验 A/B 上目标的 GPU/host residency 均为空。`request_observed` 报告刚才已实际生成的 input_ids、原始 salt 和源 URL，不补发一次推理来建立关联。这样，Router 自由选 A 时也必须证明真实 L3 恢复，不会把 A 的暖 GPU 命中包装为预取收益。

`tool_wait` 不提供压力证据，跳过 DEMOTE；源 L3 发布已由两臂共同完成。`resume_expected` 指定冻结的 main branch revision 1 和准备 B，发送剩余工具等待毫秒。独立线程查询 intent，核验冻结范围、B 的 Engine ID/epoch 和 `engine_scoped_query` 的 host 就绪范围。主线程在同一个工具 deadline 发出正式请求，不 join 准备线程、不延长等待。准备不足时仍发推理并保留 fallback/late 结果。正式发送迟到量和每个 RPC 时间写入 raw。

B3 强制要求响应携带 `x-sgl-worker-url`，值必须属于已核验的 A/B。通过该实际 owner 的 `cached_tokens_details` 校验 tier：准备 B 已就绪时需证明 host 复用，选择冷 A 时需证明普通 storage 复用。两种情况都拒绝 GPU 命中和前缀重算。缺失/未知归属、错误 epoch、冻结范围不匹配和准备查询错误会保留完整响应并标为 invalid。每个 input_case 的输出 IDs/文本在各条件间比较。

默认 `route_change` 负例在每种输入的第 0 个配对中加入一条：同样准备 B，正式恢复直达冷 A。它独立于 B3 的自然选 A，不计入 B2/B3 配对差值。只把实际观察到的准备范围计为 `unused_prepared_tokens_observed`；未观察到完成范围时不声称发生了未使用传输。正式请求到 B 时准备尚未完成，使用/未使用归因留空。字节与物理 I/O 不由 token 数冒充。

正式响应后显式 `task_finished`，对已接收的稳定 `prepare-b` intent 请求 cancel、query 到 quiescent，再 ack。提交响应丢失时仍沿同一 message ID 查询，无法确认 drain 时保留 unknown，不会报告 cleanup 成功。生命周期调用有稳定 event ID，传输响应丢失可重试一次。Router 的任务 tombstone 和生命周期 replay 仍使用其有界预算；每条 trial 通常有五条 lifecycle receipts，运行配置应容纳全部计划条目。

`router-summary.json/.md` 保留全部完成响应、invalid 原因、准备就绪数量、fallback、路由变化、观察到的 unused 范围及配对 mean delta 的 bootstrap CI。差值为 B3 − B2；这两个组都准备 B，均不代表 B0。`router-schedule.json` 记录固定 seed 的组内随机顺序。按长度/等待/组的统计在 `conditions`，原始请求/响应/SSE/header 在 `router-raw.jsonl`。

Router cost EMA 是有状态的，前后两臂可能拿到不同估计。直连 B2 的正式流量不会触发 Router 的正式调度学习屏障；若准备尚未完成，Router 对该轮成本学习可能混入按需竞争。raw 保存每次 decision 和时间，summary 明确此限制。可使用单独实验 Router 与明确 prior 控制初始条件；当前脚本不会修改 Router 配置、重置 EMA 或固定最终路由。

本地验证：

```bash
cd benchmark/session_kv_control
/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/.venv/bin/python \
  -W error::ResourceWarning -m unittest -v test_router_bench test_session_kv_bench
```

Router 测试使用 loopback HTTP fixture，覆盖冻结源输入、实际选 A/B、普通 L3/准备 host 反馈、header 丢失/错误、child epoch 错误、late/fallback、未 drain cleanup、凭证不入日志和整次 JSONL 配对。fixture 不启动真实 Router/SGLang，也不占用 GPU。真实 B3 性能由运行该命令的调用方验证。
