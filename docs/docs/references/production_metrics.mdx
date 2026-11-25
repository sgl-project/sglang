---
title: "Production Metrics"
metatags:
    description: "SGLang Prometheus metrics: TTFT, TPOT, throughput, cache hit rate. Grafana dashboard setup guide."
---
SGLang exposes the following metrics via Prometheus. You can enable it by adding `--enable-metrics` when you launch the server.

An example of the monitoring dashboard is available in [examples/monitoring/grafana.json](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json).

## Language model metrics

`sglang:time_to_first_token_seconds` measures the time from request creation in the API server to the first token. For streaming requests it ends when the first output arrives after detokenization, which may contain no printable text. Non-streaming clients only see the final response, so for them it ends when the scheduler produced the first token.

Here is an example of the metrics:

```text Output
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
# HELP sglang:spec_num_steps Currently active speculative_num_steps.
# TYPE sglang:spec_num_steps gauge
sglang:spec_num_steps{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.0
# HELP sglang:spec_num_draft_tokens Currently active speculative_num_draft_tokens (decouples from steps under topk>1).
# TYPE sglang:spec_num_draft_tokens gauge
sglang:spec_num_draft_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.0
```

## Request-level time per output token (TPOT)

`sglang:request_time_per_output_token_seconds` records TPOT once per successfully finished request:

```text
TPOT = (finished_time - first_token_time) / (completion_tokens - 1)
     = 1 / meta_info["decode_throughput"]
```

The first token is excluded because its latency belongs to TTFT. Requests with fewer than two output tokens, streaming requests that finish in their first output, aborted requests, and PD prefill workers do not contribute. The histogram is split by `is_streaming` and reuses the `--bucket-inter-token-latency` buckets.

Unlike the token-weighted `sglang:inter_token_latency_seconds`, this metric gives every qualifying request equal weight.

p95 TPOT per model:

```promql
histogram_quantile(
  0.95,
  sum by (model_name, is_streaming, le) (
    rate(sglang:request_time_per_output_token_seconds_bucket[5m])
  )
)
```

Mean TPOT is `rate(..._sum[5m]) / rate(..._count[5m])`. Per-request decode tokens/s is its reciprocal, so p5 TPS is `1 / p95 TPOT`; the reciprocal of mean TPOT is not mean TPS.

## Rust frontend metrics

The Rust frontend (`sglang-server`) exports the frontend metrics above under the same names, plus these families of its own. They carry the same labels as the request metrics (`model_name`, `engine_type`, ...), plus the one listed.

| Metric | Type | Label | Meaning |
|---|---|---|---|
| `sglang:frontend_queue_depth` | gauge | `stage` | Requests waiting in a frontend stage queue, read at each scrape: `intake` (the request loop's inbox), `tokenizer` and `mm` (the tokenizer and multimodal worker pools), `scheduler` (the scheduler's request queue). |
| `sglang:frontend_queue_full_total` | counter | `stage` | Generations failed as overloaded because a stage queue was full: a 503, or a 503 error item in a stream or batch. `stage` is the full queue: `tokenizer`, `mm` or `scheduler`, as in `sglang:frontend_queue_depth` (intake waits for room instead of refusing). |
| `sglang:frontend_media_fetch_slot_wait_seconds` | histogram | | Time a media fetch waited for one of the process's concurrent fetch slots. |

The gauge is summed over processes.

## Diffusion metrics

SGLang Diffusion exposes request, queue, stage and LoRA metrics with
`--enable-metrics`. See [Diffusion production metrics](/docs/sglang-diffusion/production_metrics)
for the metric reference, counting semantics and disaggregated scraping setup.

## Setup Guide

This section describes how to set up the monitoring stack (Prometheus + Grafana) provided in the `examples/monitoring` directory.

### Prerequisites

- Docker and Docker Compose installed
- SGLang server running with metrics enabled

### Usage

1.  **Start your SGLang server with metrics enabled:**

    ```bash Command
    python -m sglang.launch_server \
      --model-path <your_model_path> \
      --port 30000 \
      --enable-metrics \
      --enable-mfu-metrics
    ```
    Replace `<your_model_path>` with the actual path to your model (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). The example stack uses host networking, so the default `--host 127.0.0.1` is reachable from it. If you bind SGLang to `0.0.0.0` instead, also set `--api-key` (see [Server Arguments](../advanced_features/server_arguments#http-server)). By default, the metrics endpoint will be available at `http://<sglang_server_host>:30000/metrics`.

2.  **Navigate to the monitoring example directory:**
    ```bash Command
    cd examples/monitoring
    ```

3.  **Start the monitoring stack, choosing a Grafana admin password:**
    ```bash Command
    (umask 077; echo "GRAFANA_ADMIN_PASSWORD='<a strong password>'" > .env)
    docker compose up -d
    ```
    This starts Prometheus and Grafana in the background. Docker Compose reads the password from `.env` (excluded by the repository's `.gitignore`) for every later command too, such as `docker compose ps` or `docker compose down`; exporting `GRAFANA_ADMIN_PASSWORD` in your shell works as well. Every `docker compose` command for this stack fails while the variable is unset.

    Grafana applies the password only when it first creates its database. To change it later, change it in Grafana under your profile, or run `docker compose exec grafana grafana cli admin reset-admin-password '<new password>'`, then update `.env`. Do not use `docker compose down` and a restart to change it: the example does not persist Grafana's database or Prometheus's data, so that discards dashboards, users and alert rules created in Grafana and Prometheus's metric history.

4.  **Access the monitoring interfaces:**
    *   **Grafana:** Open your web browser and go to [http://localhost:3000](http://localhost:3000).
    *   **Prometheus:** Open your web browser and go to [http://localhost:9090](http://localhost:9090).

5.  **Log in to Grafana:** as `admin`, with the password you set. Grafana and Prometheus listen on `127.0.0.1` only; to view them from another machine, use an SSH tunnel (`ssh -L 3000:127.0.0.1:3000 <host>`) instead of exposing the ports.

6.  **View the Dashboard:**
    The SGLang dashboard is pre-configured and should be available automatically. Navigate to `Dashboards` -> `Browse` -> `SGLang Monitoring` folder -> `SGLang Dashboard`.

### Troubleshooting

*   **Port Conflicts:** If you encounter errors like "port is already allocated," check if other services (including previous instances of Prometheus/Grafana) are using ports `9090` or `3000`. Use `docker ps` to find running containers and `docker stop <container_id>` to stop them, or use `lsof -i :<port>` to find other processes using the ports. You might need to adjust the ports in the `docker-compose.yaml` file if they permanently conflict with other essential services on your system.

    To move Grafana to another port (like 3090), append `GF_SERVER_HTTP_PORT` to the existing `environment` list of the `grafana` service. Both services use host networking, so a `ports:` mapping has no effect.

    ```yaml
      grafana:
        environment:
          # ...keep the existing entries...
          - GF_SERVER_HTTP_PORT=3090  # <-- Add this line
    ```
*   **Connection Issues:**
    *   Ensure both Prometheus and Grafana containers are running (`docker ps`).
    *   Verify the Prometheus data source configuration in Grafana (usually auto-configured via `grafana/datasources/datasource.yaml`). Go to `Connections` -> `Data sources` -> `Prometheus`. The URL should point to the Prometheus service (e.g., `http://prometheus:9090`).
    *   Confirm that your SGLang server is running and the metrics endpoint (`http://<sglang_server_host>:30000/metrics`) is accessible *from the Prometheus container*. If SGLang is running on your host machine and Prometheus is in Docker, use `host.docker.internal` (on Docker Desktop) or your machine's network IP instead of `localhost` in the `prometheus.yaml` scrape configuration.
*   **No Data on Dashboard:**
    *   Generate some traffic to your SGLang server to produce metrics. For example, run a benchmark:
        ```bash Command
        python3 -m sglang.benchmark.serving --backend sglang --dataset-name random --num-prompts 100 --random-input 128 --random-output 128
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
```text Output
python3 -m sglang.benchmark.serving \
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
