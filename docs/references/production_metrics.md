# Production Metrics

SGLang exposes the following metrics via Prometheus. You can enable it by adding `--enable-metrics` when you launch the server.

An example of the monitoring dashboard is available in [examples/monitoring/grafana.json](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json).

Here is an example of the metrics:

```
$ curl http://localhost:30000/metrics
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.128902e+06
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.557572e+06
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.007507552643049313
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.3518979474117756e+06
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.0
sglang:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.25",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 27.0
sglang:time_to_first_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
sglang:time_to_first_token_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 314.0
sglang:time_to_first_token_seconds_bucket{le="7.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 941.0
sglang:time_to_first_token_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1330.0
sglang:time_to_first_token_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1970.0
sglang:time_to_first_token_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2326.0
sglang:time_to_first_token_seconds_bucket{le="25.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2417.0
sglang:time_to_first_token_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2513.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.116093850019932e+06
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.0
sglang:e2e_request_latency_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 10.0
sglang:e2e_request_latency_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11.0
sglang:e2e_request_latency_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 14.0
sglang:e2e_request_latency_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 247.0
sglang:e2e_request_latency_seconds_bucket{le="40.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 486.0
sglang:e2e_request_latency_seconds_bucket{le="50.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 845.0
sglang:e2e_request_latency_seconds_bucket{le="60.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1513.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
# HELP sglang:inter_token_latency_seconds Histogram of inter-token latency in seconds.
# TYPE sglang:inter_token_latency_seconds histogram
sglang:inter_token_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:inter_token_latency_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:inter_token_latency_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:inter_token_latency_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:inter_token_latency_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 593.0
sglang:inter_token_latency_seconds_bucket{le="0.025",model_name="meta-llama/Llama-3.1-8B-Instruct"} 855.0
sglang:inter_token_latency_seconds_bucket{le="0.03",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1035.0
sglang:inter_token_latency_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1815.0
sglang:inter_token_latency_seconds_bucket{le="0.05",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11685.0
sglang:inter_token_latency_seconds_bucket{le="0.075",model_name="meta-llama/Llama-3.1-8B-Instruct"} 433413.0
sglang:inter_token_latency_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.950195e+06
sglang:inter_token_latency_seconds_bucket{le="0.15",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.039435e+06
sglang:inter_token_latency_seconds_bucket{le="0.2",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.171662e+06
sglang:inter_token_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.266055e+06
sglang:inter_token_latency_seconds_bucket{le="0.4",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.296752e+06
sglang:inter_token_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.312226e+06
sglang:inter_token_latency_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.339675e+06
sglang:inter_token_latency_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.357747e+06
sglang:inter_token_latency_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.389414e+06
sglang:inter_token_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:inter_token_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 4.514771912145079
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.253125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.3796875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.56953125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.8542968750000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.2814453125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.9221679687500002",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="2.8832519531250003",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="4.3248779296875",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="6.487316894531251",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="9.730975341796876",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="14.596463012695313",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="21.89469451904297",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="32.84204177856446",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="49.26306266784668",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 14007.0
sglang:func_latency_seconds_count{name="generate_request"} 14007.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 162.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 123859.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 86.50814177726902
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2826.0
```

The snippet above is only a small illustrative sample. The complete list of
metrics exposed by the server is documented in the
[Metrics Reference](#metrics-reference) below.

## Metrics Reference

This section lists every metric exposed on the `/metrics` endpoint, grouped by
area. The **Type** column uses Prometheus metric types (`Counter`, `Gauge`,
`Histogram`, `Summary`). Any labels listed in **Extra labels** are attached in
addition to the common labels described below.

> **Metric activity.** Most metrics below are actively populated during normal
> operation. A handful are currently **inactive** — they are registered on the
> endpoint but the code path that would populate them is not wired up, so they
> always report their default value (or are never emitted). Inactive metrics are
> marked with a dagger (`†`) in the tables and listed together under
> [Currently inactive metrics](#currently-inactive-metrics). Feature-gated
> metrics (see [Enablement flags](#enablement-flags)) are *not* inactive — they
> are populated with real values once their feature is turned on.

### Labels

Two collectors emit metrics, each with its own set of common labels applied to
every metric it exports:

- **Request metrics** (emitted from the tokenizer/API-server process) carry
  `model_name`.
- **Scheduler / engine metrics** carry `model_name`, `engine_type`, `tp_rank`,
  `pp_rank`, and `moe_ep_rank`. `dp_rank` is added when data parallelism is
  enabled.

Additional common labels can appear depending on server configuration:

- `priority` — added to all metrics when `--enable-priority-scheduling` is set.
- Custom labels from `--extra-metric-labels` (both collectors) and
  `--tokenizer-metrics-allowed-custom-labels` (request metrics).

### Enablement flags

Some metrics are only created when the corresponding feature is turned on:

| Feature | Flag / condition | Metrics |
| --- | --- | --- |
| MFU estimation | `--enable-mfu-metrics` | `estimated_flops_per_gpu_total`, `estimated_read_bytes_per_gpu_total`, `estimated_write_bytes_per_gpu_total` |
| LoRA | LoRA serving enabled | `lora_pool_slots_used`, `lora_pool_slots_total`, `lora_pool_utilization` |
| Hierarchical (Hi)Cache | `--enable-hierarchical-cache` | `hicache_host_used_tokens`, `hicache_host_total_tokens`, `evicted_tokens_total`, `eviction_duration_seconds`, `load_back_tokens_total`, `load_back_duration_seconds` |
| L3 / storage tier | storage backend enabled | `prefetched_tokens_total`, `backuped_tokens_total`, `prefetch_pgs`, `backup_pgs`, `prefetch_bandwidth`, `backup_bandwidth` |
| Prefill/Decode disaggregation | disaggregation mode enabled | queue-depth, `kv_transfer_*`, `num_bootstrap_failed_reqs_total`, `num_transfer_failed_reqs_total` |
| Per-device GPU timing | `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` | `gpu_execution_seconds_total`, `gpu_overlap_wait_seconds_total`, `dp_cooperation_gpu_execution_seconds_total` |
| Expert parallelism load balancing | `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC=1` (MoE, `moe_ep_rank == 0`) | `eplb_balancedness` |
| EPLB heatmap | `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL > 0` | `eplb_gpu_physical_count` |
| Function latency timer | metrics enabled | `func_latency_seconds` |

### Currently inactive metrics

The following metrics are registered on the `/metrics` endpoint but are **not
currently populated** by any code path — they always report their default value
(e.g. `0`) or are never emitted at all. They are kept here for completeness and
so operators do not build alerts on top of them expecting live data. Each is
also marked with `†` in the tables below.

| Metric | Reason it is inactive |
| --- | --- |
| `sglang:utilization` | Only computed when `max_running_requests_under_SLO` is set, which never happens; stays `0` (or `-1` in prefill-disaggregation mode). |
| `sglang:max_running_requests_under_SLO` | Source value is never assigned anywhere, so the gauge is never emitted. |
| `sglang:pending_prealloc_token_usage` | Backing `SchedulerStats` field is never populated; always `0`. |
| `sglang:engine_startup_time` | Backing `SchedulerStats` field is never populated; always `0`. |
| `sglang:engine_load_weights_time` | Backing `SchedulerStats` field is never populated; always `0`. |
| `sglang:is_cuda_graph` | Backing `SchedulerStats` field is never populated; superseded by `cuda_graph_passes_total`. |
| `sglang:num_prefill_retries_total` | `increment_prefill_retries()` has no callers. |
| `sglang:num_grammar_total` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:num_grammar_cache_hit_total` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:num_grammar_aborted_total` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:num_grammar_timeout_total` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:grammar_compilation_time_seconds` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:grammar_schema_count` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:grammar_ebnf_size` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:grammar_tree_traversal_time_avg` | Populated only by `log_grammar_stats()`, which has no callers. |
| `sglang:grammar_tree_traversal_time_max` | Populated only by `log_grammar_stats()`, which has no callers. |

> Note: `sglang:num_grammar_queue_reqs` is **active** — it is a separate gauge
> sourced from the scheduler's grammar queue length, not from `log_grammar_stats()`.

### Request metrics

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:prompt_tokens_total` | Counter | | Number of prefill tokens processed. |
| `sglang:generation_tokens_total` | Counter | | Number of generation tokens processed. |
| `sglang:cached_tokens_total` | Counter | `cache_source` | Number of cached prompt tokens by source (device/host/storage). |
| `sglang:num_requests_total` | Counter | | Number of requests processed. |
| `sglang:num_so_requests_total` | Counter | | Number of structured output requests processed. |
| `sglang:num_aborted_requests_total` | Counter | | Number of requests aborted. |
| `sglang:prompt_tokens_histogram` | Histogram | | Histogram of prompt token length. |
| `sglang:generation_tokens_histogram` | Histogram | | Histogram of generation token length. |
| `sglang:time_to_first_token_seconds` | Histogram | | Histogram of time to first token in seconds. |
| `sglang:inter_token_latency_seconds` | Histogram | | Histogram of inter-token latency in seconds. |
| `sglang:e2e_request_latency_seconds` | Histogram | | Histogram of end-to-end request latency in seconds. |
| `sglang:per_stage_req_latency_seconds` | Histogram | `stage` | Latency of each stage of a request's lifecycle. |
| `sglang:queue_time_seconds` | Histogram | | Histogram of queueing time in seconds. |

### Scheduler state

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:num_running_reqs` | Gauge | | The number of running requests. |
| `sglang:num_running_reqs_offline_batch` | Gauge | | The number of running low-priority offline batch requests. |
| `sglang:num_queue_reqs` | Gauge | | The number of requests in the waiting queue. |
| `sglang:num_grammar_queue_reqs` | Gauge | | The number of requests in the grammar waiting queue. |
| `sglang:num_paused_reqs` | Gauge | | The number of paused requests by async weight sync. |
| `sglang:num_used_tokens` | Gauge | | The number of used tokens. |
| `sglang:max_total_num_tokens` | Gauge | | Maximum total number of tokens in the KV cache pool. |
| `sglang:token_usage` | Gauge | | The token usage. |
| `sglang:full_token_usage` | Gauge | | The token usage for full attention layers. |
| `sglang:swa_token_usage` | Gauge | | The token usage for SWA (sliding window attention) layers. |
| `sglang:mamba_usage` | Gauge | | The token usage for Mamba layers. |
| `sglang:pending_prealloc_token_usage` † | Gauge | | The token usage for pending preallocated tokens (not preallocated yet). |
| `sglang:gen_throughput` | Gauge | | The generation throughput (token/s). |
| `sglang:cache_hit_rate` | Gauge | | The prefix cache hit rate. |
| `sglang:new_token_ratio` | Gauge | | The new token ratio. |
| `sglang:decode_sum_seq_lens` | Gauge | | The sum of all sequence lengths in decode. |
| `sglang:utilization` † | Gauge | | The utilization. |
| `sglang:max_running_requests_under_SLO` † | Gauge | | The maximum number of running requests under SLO. |
| `sglang:cache_config_info` | Gauge | `page_size`, `num_pages` | Cache configuration information. |

### Retraction and retries

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:num_retracted_reqs` | Gauge | | The number of retracted requests (point-in-time). |
| `sglang:num_retracted_requests_total` | Counter | | Total number of retracted requests. |
| `sglang:num_retracted_input_tokens_total` | Counter | | Total number of retracted input tokens. |
| `sglang:num_retracted_output_tokens_total` | Counter | | Total number of retracted output tokens. |
| `sglang:num_retractions` | Histogram | | Histogram of retraction counts per request. |
| `sglang:num_prefill_retries_total` † | Counter | | Total number of prefill retries. |

### Speculative decoding

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:spec_accept_length` | Gauge | | The average acceptance length of speculative decoding. |
| `sglang:spec_accept_rate` | Gauge | | The average acceptance rate of speculative decoding (`accepted tokens / total draft tokens` in batch). |

### Grammar / structured output

> All metrics in this table are currently **inactive** (`†`): they are populated
> only by `log_grammar_stats()`, which has no callers. The live signal for
> structured-output load is `sglang:num_grammar_queue_reqs` (see
> [Scheduler state](#scheduler-state)) and `sglang:num_so_requests_total` (see
> [Request metrics](#request-metrics)).

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:num_grammar_total` † | Counter | | Number of total grammar requests. |
| `sglang:num_grammar_cache_hit_total` † | Counter | | Number of grammar cache hits. |
| `sglang:num_grammar_aborted_total` † | Counter | | Number of grammar aborted requests. |
| `sglang:num_grammar_timeout_total` † | Counter | | Number of grammar timeouts. |
| `sglang:grammar_compilation_time_seconds` † | Histogram | | Histogram of grammar compilation time in seconds. |
| `sglang:grammar_schema_count` † | Histogram | | Histogram of grammar schema count. |
| `sglang:grammar_ebnf_size` † | Histogram | | Histogram of grammar EBNF size. |
| `sglang:grammar_tree_traversal_time_avg` † | Histogram | | Histogram of average grammar tree traversal time in seconds. |
| `sglang:grammar_tree_traversal_time_max` † | Histogram | | Histogram of max grammar tree traversal time in seconds. |

### Prefill/Decode disaggregation and KV transfer

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:num_prefill_prealloc_queue_reqs` | Gauge | | The number of requests in the prefill prealloc queue. |
| `sglang:num_prefill_inflight_queue_reqs` | Gauge | | The number of requests in the prefill inflight queue. |
| `sglang:num_decode_prealloc_queue_reqs` | Gauge | | The number of requests in the decode prealloc queue. |
| `sglang:num_decode_transfer_queue_reqs` | Gauge | | The number of requests in the decode transfer queue. |
| `sglang:num_bootstrap_failed_reqs_total` | Counter | | The number of bootstrap failed requests. |
| `sglang:num_transfer_failed_reqs_total` | Counter | | The number of transfer failed requests. |
| `sglang:kv_transfer_speed_gb_s` | Histogram | | Histogram of KV cache transfer speed in GB/s. |
| `sglang:kv_transfer_latency_ms` | Histogram | | Histogram of KV cache transfer latency in ms. |
| `sglang:kv_transfer_bootstrap_ms` | Histogram | | Histogram of KV transfer bootstrap time in ms. |
| `sglang:kv_transfer_alloc_ms` | Histogram | | Histogram of KV transfer allocation waiting time in ms. |
| `sglang:kv_transfer_total_mb` | Histogram | | Histogram of KV cache transfer size in MB. |

### Performance, MFU, and CUDA graph

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:realtime_tokens_total` | Counter | `mode` | Total number of tokens processed (updated each log interval). `mode`: `prefill_compute`, `prefill_cache`, `decode`. |
| `sglang:gpu_execution_seconds_total` | Counter | `category` | Total time that GPU is busy executing a workload. See `ForwardMode` for category labels. |
| `sglang:gpu_overlap_wait_seconds_total` | Counter | `category` | Total time the GPU forward stream was idle waiting for the CPU schedule stream (overlap bubble). |
| `sglang:estimated_flops_per_gpu_total` | Counter | | Estimated floating-point operations per GPU (for MFU). |
| `sglang:estimated_read_bytes_per_gpu_total` | Counter | | Estimated bytes read from memory per GPU (for MFU). |
| `sglang:estimated_write_bytes_per_gpu_total` | Counter | | Estimated bytes written to memory per GPU (for MFU). |
| `sglang:cuda_graph_passes_total` | Counter | `mode` | Total number of forward passes categorized by CUDA graph. |
| `sglang:is_cuda_graph` † | Gauge | | Whether the batch is using CUDA graph. |
| `sglang:dp_cooperation_realtime_tokens_total` | Counter | `mode`, `num_prefill_ranks` | Total tokens processed, labeled with DP cooperation info. |
| `sglang:dp_cooperation_gpu_execution_seconds_total` | Counter | `category`, `num_prefill_ranks` | GPU busy time, labeled with DP cooperation info. |

### Hierarchical cache, radix cache, and storage

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:hicache_host_used_tokens` | Gauge | | Number of tokens currently used in the host KV cache. |
| `sglang:hicache_host_total_tokens` | Gauge | | Total capacity of the host KV cache in tokens. |
| `sglang:evicted_tokens_total` | Counter | | The number of tokens evicted from GPU to CPU. |
| `sglang:eviction_duration_seconds` | Histogram | | Time taken to evict memory from GPU to CPU in seconds. |
| `sglang:load_back_tokens_total` | Counter | | The number of tokens loaded from CPU to GPU. |
| `sglang:load_back_duration_seconds` | Histogram | | Time taken to load memory from CPU to GPU in seconds. |
| `sglang:prefetched_tokens_total` | Counter | | Number of prefetched prompt tokens. |
| `sglang:backuped_tokens_total` | Counter | | Number of backuped tokens. |
| `sglang:prefetch_bandwidth` | Histogram | | Histogram of prefetch bandwidth in GB/s. |
| `sglang:prefetch_pgs` | Histogram | | Histogram of prefetch pages of batches. |
| `sglang:backup_bandwidth` | Histogram | | Histogram of backup bandwidth in GB/s. |
| `sglang:backup_pgs` | Histogram | | Histogram of backup pages of batches. |

### LoRA

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:lora_pool_slots_used` | Gauge | | Number of LoRA adapter slots currently occupied in GPU memory. |
| `sglang:lora_pool_slots_total` | Gauge | | Total number of LoRA adapter slots available (`max_loras_per_batch`). |
| `sglang:lora_pool_utilization` | Gauge | | LoRA pool utilization ratio (used/total); `1.0` means the pool is full. |

### Mixture-of-Experts (EPLB)

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:eplb_balancedness` | Summary | `forward_mode` | Balancedness of MoE in expert parallelism. |
| `sglang:eplb_gpu_physical_count` | Histogram | `layer` | The selected count of physical experts on each layer and GPU rank. |

### Routing keys (multi-tenant)

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:num_unique_running_routing_keys` | Gauge | | Number of unique routing keys in the running batch. |
| `sglang:routing_key_running_req_count` | GaugeHistogram | | Distribution of routing keys by running request count (`gt < count <= le`). |
| `sglang:routing_key_all_req_count` | GaugeHistogram | | Distribution of routing keys by running+waiting request count (`gt < count <= le`). |

### Prefill delayer

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:prefill_delayer_wait_forward_passes` | Histogram | | Histogram of forward passes waited by the prefill delayer. |
| `sglang:prefill_delayer_wait_seconds` | Histogram | | Histogram of wait time in seconds by the prefill delayer. |
| `sglang:prefill_delayer_outcomes_total` | Counter | `input_estimation`, `output_allow`, `output_reason`, `actual_execution` | Prefill delayer outcome counts. |

### Engine and process

| Metric | Type | Extra labels | Description |
| --- | --- | --- | --- |
| `sglang:engine_startup_time` † | Gauge | | The time taken for the engine to start up. |
| `sglang:engine_load_weights_time` † | Gauge | | The time taken for the engine to load weights. |
| `sglang:process_cpu_seconds_total` | Counter | `component` | Total CPU time consumed by this process (user + system). |
| `sglang:func_latency_seconds` | Histogram | `name` | Function latency in seconds (per instrumented function). |

## Setup Guide

This section describes how to set up the monitoring stack (Prometheus + Grafana) provided in the `examples/monitoring` directory.

### Prerequisites

- Docker and Docker Compose installed
- SGLang server running with metrics enabled

### Usage

1.  **Start your SGLang server with metrics enabled:**

    ```bash
    python -m sglang.launch_server \
      --model-path <your_model_path> \
      --port 30000 \
      --enable-metrics \
      --enable-mfu-metrics
    ```
    Replace `<your_model_path>` with the actual path to your model (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). Ensure the server is accessible from the monitoring stack (you might need `--host 0.0.0.0` if running in Docker). By default, the metrics endpoint will be available at `http://<sglang_server_host>:30000/metrics`.

2.  **Navigate to the monitoring example directory:**
    ```bash
    cd examples/monitoring
    ```

3.  **Start the monitoring stack:**
    ```bash
    docker compose up -d
    ```
    This command will start Prometheus and Grafana in the background.

4.  **Access the monitoring interfaces:**
    *   **Grafana:** Open your web browser and go to [http://localhost:3000](http://localhost:3000).
    *   **Prometheus:** Open your web browser and go to [http://localhost:9090](http://localhost:9090).

5.  **Log in to Grafana:**
    *   Default Username: `admin`
    *   Default Password: `admin`
    You will be prompted to change the password upon your first login.

6.  **View the Dashboard:**
    The SGLang dashboard is pre-configured and should be available automatically. Navigate to `Dashboards` -> `Browse` -> `SGLang Monitoring` folder -> `SGLang Dashboard`.

### Troubleshooting

*   **Port Conflicts:** If you encounter errors like "port is already allocated," check if other services (including previous instances of Prometheus/Grafana) are using ports `9090` or `3000`. Use `docker ps` to find running containers and `docker stop <container_id>` to stop them, or use `lsof -i :<port>` to find other processes using the ports. You might need to adjust the ports in the `docker-compose.yaml` file if they permanently conflict with other essential services on your system.

To modify Grafana's port to the other one(like 3090) in your Docker Compose file, you need to explicitly specify the port mapping under the grafana service.

    Option 1: Add GF_SERVER_HTTP_PORT to the environment section:
    ```
      environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_SERVER_HTTP_PORT=3090  # <-- Add this line
    ```
    Option 2: Use port mapping:
    ```
    grafana:
      image: grafana/grafana:latest
      container_name: grafana
      ports:
      - "3090:3000"  # <-- Host:Container port mapping
    ```
*   **Connection Issues:**
    *   Ensure both Prometheus and Grafana containers are running (`docker ps`).
    *   Verify the Prometheus data source configuration in Grafana (usually auto-configured via `grafana/datasources/datasource.yaml`). Go to `Connections` -> `Data sources` -> `Prometheus`. The URL should point to the Prometheus service (e.g., `http://prometheus:9090`).
    *   Confirm that your SGLang server is running and the metrics endpoint (`http://<sglang_server_host>:30000/metrics`) is accessible *from the Prometheus container*. If SGLang is running on your host machine and Prometheus is in Docker, use `host.docker.internal` (on Docker Desktop) or your machine's network IP instead of `localhost` in the `prometheus.yaml` scrape configuration.
*   **No Data on Dashboard:**
    *   Generate some traffic to your SGLang server to produce metrics. For example, run a benchmark:
        ```bash
        python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 100 --random-input 128 --random-output 128
        ```
    *   Check the Prometheus UI (`http://localhost:9090`) under `Status` -> `Targets` to see if the SGLang endpoint is being scraped successfully.
    *   Verify the `model_name` and `instance` labels in your Prometheus metrics match the variables used in the Grafana dashboard. You might need to adjust the Grafana dashboard variables or the labels in your Prometheus configuration.

### Configuration Files

The monitoring setup is defined by the following files within the `examples/monitoring` directory:

*   `docker-compose.yaml`: Defines the Prometheus and Grafana services.
*   `prometheus.yaml`: Prometheus configuration, including scrape targets.
*   `grafana/datasources/datasource.yaml`: Configures the Prometheus data source for Grafana.
*   `grafana/dashboards/config/dashboard.yaml`: Tells Grafana to load dashboards from the specified path.
*   `grafana/dashboards/json/sglang-dashboard.json`: The actual Grafana dashboard definition in JSON format.

You can customize the setup by modifying these files. For instance, you might need to update the `static_configs` target in `prometheus.yaml` if your SGLang server runs on a different host or port.

#### Check if the metrics are being collected

Run:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024 \
  --random-range-ratio 0.5
```

to generate some requests.

Then you should be able to see the metrics in the Grafana dashboard.

## Estimated Performance Metrics (MFU-related)

SGLang exports the following estimated per-GPU counters that can be used to derive
Model FLOPs Utilization (MFU)-related signals:

- `sglang:estimated_flops_per_gpu_total`: Estimated floating-point operations.
- `sglang:estimated_read_bytes_per_gpu_total`: Estimated bytes read from memory.
- `sglang:estimated_write_bytes_per_gpu_total`: Estimated bytes written to memory.

These metrics are available when both `--enable-metrics` and
`--enable-mfu-metrics` are enabled.

These are cumulative counters. Use Prometheus `rate(...)` to get per-second values.

### PromQL examples

Average TFLOPS per GPU:

```promql
rate(sglang:estimated_flops_per_gpu_total[1m]) / 1e12
```

Average estimated memory bandwidth in GB/s:

```promql
(rate(sglang:estimated_read_bytes_per_gpu_total[1m]) +
 rate(sglang:estimated_write_bytes_per_gpu_total[1m])) / 1e9
```

### Notes

- These metrics are estimates intended for observability and trend analysis.
- Estimated memory bytes reflect modeled traffic and are not a direct hardware
  counter from GPU profilers.
