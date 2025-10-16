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
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:time_per_output_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 593.0
sglang:time_per_output_token_seconds_bucket{le="0.025",model_name="meta-llama/Llama-3.1-8B-Instruct"} 855.0
sglang:time_per_output_token_seconds_bucket{le="0.03",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1035.0
sglang:time_per_output_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1815.0
sglang:time_per_output_token_seconds_bucket{le="0.05",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11685.0
sglang:time_per_output_token_seconds_bucket{le="0.075",model_name="meta-llama/Llama-3.1-8B-Instruct"} 433413.0
sglang:time_per_output_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.950195e+06
sglang:time_per_output_token_seconds_bucket{le="0.15",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.039435e+06
sglang:time_per_output_token_seconds_bucket{le="0.2",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.171662e+06
sglang:time_per_output_token_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.266055e+06
sglang:time_per_output_token_seconds_bucket{le="0.4",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.296752e+06
sglang:time_per_output_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.312226e+06
sglang:time_per_output_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.339675e+06
sglang:time_per_output_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.357747e+06
sglang:time_per_output_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.389414e+06
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
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
      --enable-metrics
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
