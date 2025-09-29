# SGLang Auto-Tune Tool

Auto-tune tool for finding optimal SGLang server configurations through systematic parameter search.

## Overview

This tool automatically searches through different server configurations to find the optimal settings for your specific model and hardware. It tests various combinations of server parameters and benchmarks each configuration to identify the best performing setup.

## Usage

### Basic Example

```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-8B \
  --tp 2 \
  --search-server-args '{"attention_backend": ["flashinfer", "triton"]}' \
  --request-rates 1 4 8 16
```

### Testing Multiple Parameters

```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-70B \
  --tp 4 \
  --search-server-args '{
    "attention_backend": ["flashinfer", "triton"],
    "schedule_policy": ["lpm", "random"],
    "chunked_prefill_size": [512, 1024, 2048]
  }' \
  --request-rates 1 2 4 8 \
  --num-prompts 512 \
  --random-input-len 1024 \
  --random-output-len 512
```

### Using Max Concurrency Mode

Instead of testing different request rates, you can test different max concurrency levels:

```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-8B \
  --tp 2 \
  --search-server-args '{"attention_backend": ["flashinfer"]}' \
  --max-concurrency-list 1 2 4 8 16
```

### Using Configuration File

```bash
python -m benchmark.autotune.tune_server_args --config config.json
```

Example `config.json`:
```json
{
  "model_path": "meta-llama/Meta-Llama-3-8B",
  "static_server_args": {
    "tp": 2,
    "port": 30000
  },
  "search_server_args": {
    "attention_backend": ["flashinfer", "triton"],
    "schedule_policy": ["lpm", "random"]
  },
  "request_rates": [1, 4, 8, 16],
  "num_prompts": 256,
  "random_input_len": 512,
  "random_output_len": 256
}
```

## Key Parameters

### Required Parameters
- `--model-path`: Path to the model to test
- `--search-server-args`: JSON string of server parameters to search (each parameter should be a list of values to test)
- Either `--request-rates` or `--max-concurrency-list` (mutually exclusive)

### Server Configuration
- `--tp`: Tensor parallelism degree (default: 1)
- `--dp`: Data parallelism degree (default: 1)
- `--port`: Port for the server (default: 30000)

### Benchmark Configuration
- `--num-prompts`: Number of prompts per benchmark (default: 512)
- `--dataset-name`: Dataset to use (default: "random")
- `--random-input-len`: Input length for random dataset (default: 1024)
- `--random-output-len`: Output length for random dataset (default: 1024)

### Load Testing Modes (mutually exclusive)
- `--request-rates`: List of request rates to test
- `--max-concurrency-list`: List of max concurrency levels to test

### Output Configuration
- `--output-dir`: Directory to save results (default: "autotune_results")
- `--save-server-logs`: Save server stdout/stderr logs
- `--server-log-dir`: Directory for server logs (default: "server_logs")
- `--tail-server-logs`: Show real-time server logs during execution

### Timeout Configuration
- `--server-timeout`: Server startup timeout in seconds (default: 120)
- `--benchmark-timeout`: Benchmark run timeout in seconds (default: 600)

## Output

The tool generates:
1. **CSV file** with detailed results for each configuration and request rate
2. **JSON summary** with the best configurations for each metric
3. **Console output** showing progress and results in real-time

### Metrics Tracked
- **Token Throughput**: Input, output, and total tokens per second
- **TTFT (Time to First Token)**: Mean, median, and P99 latencies
- **ITL (Inter-Token Latency)**: Mean, median, and P99 latencies
- **E2E Latency**: Mean and median end-to-end latencies
- **Success Rate**: Percentage of successful requests

## Advanced Features

### Parallel Testing
The tool automatically tests configurations in sequence but benchmarks multiple request rates for each configuration.

### Server Log Management
- Use `--save-server-logs` to save all server logs for debugging
- Use `--tail-server-logs` to see real-time server output during testing
- Logs are organized by configuration and timestamp

### Custom Datasets
Besides random data, you can use:
- `--dataset-name sharegpt` with `--dataset-path /path/to/sharegpt.json`
- Other supported datasets: random-ids, generated-shared-prefix, mmmu, random-image, mooncake

## Examples

### Find Best Attention Backend
```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-8B \
  --tp 2 \
  --search-server-args '{"attention_backend": ["flashinfer", "triton", "trtllm_mha"]}' \
  --request-rates 1 4 8 16 \
  --num-prompts 256
```

### Optimize for High Throughput
```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-70B \
  --tp 8 \
  --search-server-args '{
    "schedule_policy": ["lpm", "random", "fcfs"],
    "schedule_conservativeness": [0.3, 0.5, 1.0]
  }' \
  --request-rates 8 16 32 64 \
  --num-prompts 1024
```

### Debug Configuration Issues
```bash
python -m benchmark.autotune.tune_server_args \
  --model-path meta-llama/Meta-Llama-3-8B \
  --tp 2 \
  --search-server-args '{"attention_backend": ["flashinfer"]}' \
  --request-rates 1 \
  --num-prompts 10 \
  --save-server-logs \
  --tail-server-logs \
  --verbose-benchmark
```