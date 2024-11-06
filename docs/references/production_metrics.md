# Production Metrics

sglang exposes the following metrics via Prometheus. The metrics are namespaced by `$name` (the model name).

An example of the monitoring dashboard is available in [examples/monitoring/grafana.json](../examples/monitoring/grafana.json).

Here is an example of the metrics:

```
# HELP sglang:max_total_num_tokens Maximum total number of tokens
# TYPE sglang:max_total_num_tokens gauge
sglang:max_total_num_tokens{name="google/gemma-2-9b-it"} 161721.0
# HELP sglang:max_prefill_tokens Maximum prefill tokens
# TYPE sglang:max_prefill_tokens gauge
sglang:max_prefill_tokens{name="google/gemma-2-9b-it"} 16384.0
# HELP sglang:max_running_requests Maximum running requests
# TYPE sglang:max_running_requests gauge
sglang:max_running_requests{name="google/gemma-2-9b-it"} 4097.0
# HELP sglang:context_len Context length
# TYPE sglang:context_len gauge
sglang:context_len{name="google/gemma-2-9b-it"} 8192.0
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{name="google/gemma-2-9b-it"} 506780.0
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{name="google/gemma-2-9b-it"} 424549.0
# HELP sglang:num_requests_running Number of requests currently running on GPU
# TYPE sglang:num_requests_running gauge
sglang:num_requests_running{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:num_requests_waiting Number of requests waiting to be processed.
# TYPE sglang:num_requests_waiting gauge
sglang:num_requests_waiting{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:gen_throughput Gen token throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:token_usage Total token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{name="google/gemma-2-9b-it"} 0.01
# HELP sglang:new_seq Number of new sequences
# TYPE sglang:new_seq gauge
sglang:new_seq{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:new_token Number of new token
# TYPE sglang:new_token gauge
sglang:new_token{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:cached_token Number of cached token
# TYPE sglang:cached_token gauge
sglang:cached_token{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:cache_hit_rate Cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{name="google/gemma-2-9b-it"} 10.61
# HELP sglang:queue_req Number of queued requests
# TYPE sglang:queue_req gauge
sglang:queue_req{name="google/gemma-2-9b-it"} 0.0
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{name="google/gemma-2-9b-it"} 656.0780844688416
sglang:time_to_first_token_seconds_bucket{le="0.001",name="google/gemma-2-9b-it"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",name="google/gemma-2-9b-it"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",name="google/gemma-2-9b-it"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.02",name="google/gemma-2-9b-it"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.04",name="google/gemma-2-9b-it"} 207.0
sglang:time_to_first_token_seconds_bucket{le="0.06",name="google/gemma-2-9b-it"} 456.0
sglang:time_to_first_token_seconds_bucket{le="0.08",name="google/gemma-2-9b-it"} 598.0
sglang:time_to_first_token_seconds_bucket{le="0.1",name="google/gemma-2-9b-it"} 707.0
sglang:time_to_first_token_seconds_bucket{le="0.25",name="google/gemma-2-9b-it"} 1187.0
sglang:time_to_first_token_seconds_bucket{le="0.5",name="google/gemma-2-9b-it"} 1350.0
sglang:time_to_first_token_seconds_bucket{le="0.75",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="1.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="2.5",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="5.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="7.5",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="10.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="15.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="20.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="25.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="30.0",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",name="google/gemma-2-9b-it"} 2124.0
sglang:time_to_first_token_seconds_count{name="google/gemma-2-9b-it"} 2124.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{name="google/gemma-2-9b-it"} 29846.5393948555
sglang:time_per_output_token_seconds_bucket{le="0.005",name="google/gemma-2-9b-it"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.01",name="google/gemma-2-9b-it"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.015",name="google/gemma-2-9b-it"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.02",name="google/gemma-2-9b-it"} 9602.0
sglang:time_per_output_token_seconds_bucket{le="0.025",name="google/gemma-2-9b-it"} 30060.0
sglang:time_per_output_token_seconds_bucket{le="0.03",name="google/gemma-2-9b-it"} 39184.0
sglang:time_per_output_token_seconds_bucket{le="0.04",name="google/gemma-2-9b-it"} 61387.0
sglang:time_per_output_token_seconds_bucket{le="0.05",name="google/gemma-2-9b-it"} 78835.0
sglang:time_per_output_token_seconds_bucket{le="0.075",name="google/gemma-2-9b-it"} 139394.0
sglang:time_per_output_token_seconds_bucket{le="0.1",name="google/gemma-2-9b-it"} 422029.0
sglang:time_per_output_token_seconds_bucket{le="0.15",name="google/gemma-2-9b-it"} 422029.0
sglang:time_per_output_token_seconds_bucket{le="0.2",name="google/gemma-2-9b-it"} 422029.0
sglang:time_per_output_token_seconds_bucket{le="0.3",name="google/gemma-2-9b-it"} 422424.0
sglang:time_per_output_token_seconds_bucket{le="0.4",name="google/gemma-2-9b-it"} 422424.0
sglang:time_per_output_token_seconds_bucket{le="0.5",name="google/gemma-2-9b-it"} 422425.0
sglang:time_per_output_token_seconds_bucket{le="0.75",name="google/gemma-2-9b-it"} 422425.0
sglang:time_per_output_token_seconds_bucket{le="1.0",name="google/gemma-2-9b-it"} 422425.0
sglang:time_per_output_token_seconds_bucket{le="2.5",name="google/gemma-2-9b-it"} 422425.0
sglang:time_per_output_token_seconds_bucket{le="+Inf",name="google/gemma-2-9b-it"} 422425.0
sglang:time_per_output_token_seconds_count{name="google/gemma-2-9b-it"} 422425.0
# HELP sglang:request_prompt_tokens Number of prefill tokens processed
# TYPE sglang:request_prompt_tokens histogram
sglang:request_prompt_tokens_sum{name="google/gemma-2-9b-it"} 500552.0
sglang:request_prompt_tokens_bucket{le="1.0",name="google/gemma-2-9b-it"} 0.0
sglang:request_prompt_tokens_bucket{le="2.0",name="google/gemma-2-9b-it"} 0.0
sglang:request_prompt_tokens_bucket{le="5.0",name="google/gemma-2-9b-it"} 22.0
sglang:request_prompt_tokens_bucket{le="10.0",name="google/gemma-2-9b-it"} 191.0
sglang:request_prompt_tokens_bucket{le="20.0",name="google/gemma-2-9b-it"} 511.0
sglang:request_prompt_tokens_bucket{le="50.0",name="google/gemma-2-9b-it"} 825.0
sglang:request_prompt_tokens_bucket{le="100.0",name="google/gemma-2-9b-it"} 997.0
sglang:request_prompt_tokens_bucket{le="200.0",name="google/gemma-2-9b-it"} 1182.0
sglang:request_prompt_tokens_bucket{le="500.0",name="google/gemma-2-9b-it"} 1748.0
sglang:request_prompt_tokens_bucket{le="1000.0",name="google/gemma-2-9b-it"} 2102.0
sglang:request_prompt_tokens_bucket{le="2000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="5000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="10000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="20000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="50000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="100000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_bucket{le="+Inf",name="google/gemma-2-9b-it"} 2104.0
sglang:request_prompt_tokens_count{name="google/gemma-2-9b-it"} 2104.0
# HELP sglang:request_generation_tokens Number of generation tokens processed.
# TYPE sglang:request_generation_tokens histogram
sglang:request_generation_tokens_sum{name="google/gemma-2-9b-it"} 424529.0
sglang:request_generation_tokens_bucket{le="1.0",name="google/gemma-2-9b-it"} 0.0
sglang:request_generation_tokens_bucket{le="2.0",name="google/gemma-2-9b-it"} 0.0
sglang:request_generation_tokens_bucket{le="5.0",name="google/gemma-2-9b-it"} 49.0
sglang:request_generation_tokens_bucket{le="10.0",name="google/gemma-2-9b-it"} 202.0
sglang:request_generation_tokens_bucket{le="20.0",name="google/gemma-2-9b-it"} 448.0
sglang:request_generation_tokens_bucket{le="50.0",name="google/gemma-2-9b-it"} 814.0
sglang:request_generation_tokens_bucket{le="100.0",name="google/gemma-2-9b-it"} 979.0
sglang:request_generation_tokens_bucket{le="200.0",name="google/gemma-2-9b-it"} 1266.0
sglang:request_generation_tokens_bucket{le="500.0",name="google/gemma-2-9b-it"} 1883.0
sglang:request_generation_tokens_bucket{le="1000.0",name="google/gemma-2-9b-it"} 2095.0
sglang:request_generation_tokens_bucket{le="2000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="5000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="10000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="20000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="50000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="100000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_bucket{le="+Inf",name="google/gemma-2-9b-it"} 2104.0
sglang:request_generation_tokens_count{name="google/gemma-2-9b-it"} 2104.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{name="google/gemma-2-9b-it"} 70517.99934530258
sglang:e2e_request_latency_seconds_bucket{le="1.0",name="google/gemma-2-9b-it"} 2.0
sglang:e2e_request_latency_seconds_bucket{le="2.0",name="google/gemma-2-9b-it"} 21.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",name="google/gemma-2-9b-it"} 54.0
sglang:e2e_request_latency_seconds_bucket{le="10.0",name="google/gemma-2-9b-it"} 311.0
sglang:e2e_request_latency_seconds_bucket{le="20.0",name="google/gemma-2-9b-it"} 733.0
sglang:e2e_request_latency_seconds_bucket{le="50.0",name="google/gemma-2-9b-it"} 1563.0
sglang:e2e_request_latency_seconds_bucket{le="100.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="200.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="500.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="1000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="2000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="5000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="10000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="20000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="50000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="100000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",name="google/gemma-2-9b-it"} 2104.0
sglang:e2e_request_latency_seconds_count{name="google/gemma-2-9b-it"} 2104.0
# HELP sglang:waiting_request_latency_seconds Histogram of request waiting time in seconds
# TYPE sglang:waiting_request_latency_seconds histogram
sglang:waiting_request_latency_seconds_sum{name="google/gemma-2-9b-it"} 24885.007263183594
sglang:waiting_request_latency_seconds_bucket{le="1.0",name="google/gemma-2-9b-it"} 421.0
sglang:waiting_request_latency_seconds_bucket{le="2.0",name="google/gemma-2-9b-it"} 563.0
sglang:waiting_request_latency_seconds_bucket{le="5.0",name="google/gemma-2-9b-it"} 900.0
sglang:waiting_request_latency_seconds_bucket{le="10.0",name="google/gemma-2-9b-it"} 1270.0
sglang:waiting_request_latency_seconds_bucket{le="20.0",name="google/gemma-2-9b-it"} 1623.0
sglang:waiting_request_latency_seconds_bucket{le="50.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="100.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="200.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="500.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="1000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="2000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="5000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="10000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="20000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="50000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="100000.0",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_bucket{le="+Inf",name="google/gemma-2-9b-it"} 2104.0
sglang:waiting_request_latency_seconds_count{name="google/gemma-2-9b-it"} 2104.0
```

## Setup Guide

To setup a monitoring dashboard, you can use the following docker compose file: [examples/monitoring/docker-compose.yaml](../examples/monitoring/docker-compose.yaml).

Assume you have sglang server running at `localhost:30000`.

To start the monitoring dashboard (prometheus + grafana), cd to `examples/monitoring` and run:

```bash
docker compose -f compose.yaml -p monitoring up
```

Then you can access the Grafana dashboard at http://localhost:3000.

### Grafana Dashboard

To import the Grafana dashboard, click `+` -> `Import` -> `Upload JSON file` -> `Upload` and select [grafana.json](../examples/monitoring/grafana.json).
