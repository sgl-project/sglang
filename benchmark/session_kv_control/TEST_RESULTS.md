# 验证记录

日期：2026-11-04。Python：`/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/.venv/bin/python`。

执行目录：`/tmp/sglang-rfc-build/benchmark-work`。

```text
$ python -W error::ResourceWarning -m unittest -v test_session_kv_bench
test_accounting_and_scope_union ... ok
test_ema_separates_contended_costs_and_paired_bootstrap ... ok
test_failed_trials_remain_in_primary_statistics ... ok
test_local_trace_provenance_and_validation ... ok
test_pin_protection_is_not_transfer_readiness ... ok
test_replay_recorded_real_sse ... ok
test_sse_truncation_and_comments ... ok
test_two_http_engines_and_negative_cases ... ok

Ran 8 tests
OK
```

HTTP 集成验证覆盖：源发布→等待→恢复、public prefetch 路径、legacy src/dst 回退、backend_namespace、DEMOTE/PIN/release、kind=claim 终态查询、缺失、取消、未恢复、路由变化、拒绝 GPU 命中的基线。延迟 job 测试证明在途准备归因为 ambiguous；取消 drain 超过第二个窗口时保留 unknown 与 operation ref。完整 `run()` 路径还验证 manifest/schedule/raw/summary 文件、独立 warmup、随机配对、JSONL 输入来源、历史成本 EMA、背景并发、host 压力和带宽声明记录。全部 HTTP 集成使用本地 mock，没有请求 live A/B。

真实 SGLang HTTP/SSE 验证：`127.0.0.1:51800`，Qwen3-8B，单 A100，page64，HiCache + Mooncake v0.3.14；单次 16 input / 3 output 短请求解析成功。记录：`real-parser-smoke.jsonl`、`real-parser-smoke-result.json`。冷请求返回 cached_tokens=0、cached_tokens_details=null，脚本正确按零命中处理。

父任务已独立运行真实 feature Engine 对照：`/tmp/sglang-rfc-build/bench-p2-matrix-v1`，36/36 trial 通过路径和输出校验。该批 raw 只读分析结果在 `p2-matrix-analysis.md/.json`，Engine B 日志快照在 `p2-engine-b-evidence.log`。2K mean TTFT 336.2→138.0ms，8K 804.2→755.0ms，16K 1469.1→2502.7ms；每种长度 N=6 配对。扩展轨迹、EMA 与背景负载尚未在 GPU 上运行，不能用 mock 结果声称性能收益。

最终结果更新：完整实现、最终快照测试、真实双 Engine/Router 和性能证据见 [FINAL_ACCEPTANCE.md](results/2026-11-04/FINAL_ACCEPTANCE.md)。本文件中较早的测试记录保持原样。
