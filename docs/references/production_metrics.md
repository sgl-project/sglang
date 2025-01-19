# Production Metrics

SGLang exposes the following metrics via Prometheus. The metrics are namespaced by `$name` (the model name).

An example of the monitoring dashboard is available in [examples/monitoring/grafana.json](../examples/monitoring/grafana.json).

Here is an example of the metrics:

```
$ curl http://localhost:30000/metrics

# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.0
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.30457592010498047
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.25",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="7.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="25.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.30521273612976074
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="1.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="2.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="40.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="50.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="60.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0381757915019989
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.025",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.03",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_per_output_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.05",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.075",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.15",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.2",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.4",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 0.3061351010110229
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 0.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 0.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 0.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 0.0
sglang:func_latency_seconds_bucket{le="0.253125",name="generate_request"} 0.0
sglang:func_latency_seconds_bucket{le="0.3796875",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="0.56953125",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="0.8542968750000001",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="1.2814453125",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="1.9221679687500002",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="2.8832519531250003",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="4.3248779296875",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="6.487316894531251",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="9.730975341796876",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="14.596463012695313",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="21.89469451904297",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="32.84204177856446",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="49.26306266784668",name="generate_request"} 1.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 1.0
sglang:func_latency_seconds_count{name="generate_request"} 1.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
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
