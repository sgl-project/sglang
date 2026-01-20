# Support Features on Ascend NPU

This section describes the basic functions and features supported by the Ascend NPU.If you encounter issues or have any
questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

If you want to know the meaning and usage of each parameter,
click [Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html).

## Model and tokenizer

| Argument                               | Defaults | Options                               |                    A2                    |                    A3                    |
|----------------------------------------|----------|---------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--model-path`<br/>`--model`           | `None`   | Type: str                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tokenizer-path`                     | `None`   | Type: str                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tokenizer-mode`                     | `auto`   | `auto`, `slow`                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tokenizer-worker-num`               | `1`      | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--skip-tokenizer-init`                | `False`  | bool flag (set to enable)             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--load-format`                        | `auto`   | `auto`, `safetensors`                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--model-loader-` <br/> `extra-config` | {}       | Type: str                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--trust-remote-code`                  | `False`  | bool flag (set to enable)             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--context-length`                     | `None`   | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--is-embedding`                       | `False`  | bool flag (set to enable)             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-multimodal`                  | `None`   | bool flag (set to enable)             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--revision`                           | `None`   | Type: str                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--model-impl`                         | `auto`   | `auto`, `sglang`,<br/> `transformers` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## HTTP server

| Argument                                                          | Defaults    | Options                        |                    A2                    |                    A3                    |
|-------------------------------------------------------------------|-------------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--host`                                                          | `127.0.0.1` | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--port`                                                          | `30000`     | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--skip-server-warmup`                                            | `False`     | bool flag <br/>(set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--warmups`                                                       | `None`      | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--nccl-port`                                                     | `None`      | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--fastapi-root-path`                                             | `None`      | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--grpc-mode`                                                     | `False`     | bool flag <br/>(set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--checkpoint-engine-` <br/> `wait-weights-` <br/> `before-ready` | `False`     | bool flag <br/>(set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Quantization and data type

| Argument                                    | Defaults | Options                                 |                    A2                    |                    A3                    |
|---------------------------------------------|----------|-----------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--dtype`                                   | `auto`   | `auto`,<br/> `float16`,<br/> `bfloat16` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--quantization`                            | `None`   | `modelslim`                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--quantization-param-path`                 | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--kv-cache-dtype`                          | `auto`   | `auto`                                  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-fp32-lm-head`                     | `False`  | bool flag <br/> (set to enable)         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--modelopt-quant`                          | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--modelopt-checkpoint-`<br/>`restore-path` | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--modelopt-checkpoint-`<br/>`save-path`    | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--modelopt-export-path`                    | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--quantize-and-serve`                      | `False`  | bool flag <br/> (set to enable)         |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--rl-quant-profile`                        | `None`   | Type: str                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Memory and scheduling

| Argument                                            | Defaults | Options                        |                    A2                    |                    A3                    |
|-----------------------------------------------------|----------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--mem-fraction-static`                             | `None`   | Type: float                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-running-requests`                            | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--prefill-max-requests`                            | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-queued-requests`                             | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-total-tokens`                                | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--chunked-prefill-size`                            | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-prefill-tokens`                              | `16384`  | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--schedule-policy`                                 | `fcfs`   | `lpm`, `fcfs`                  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-priority-`<br/>`scheduling`               | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--schedule-low-priority-`<br/>`values-first`       | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--priority-scheduling-`<br/>`preemption-threshold` | `10`     | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--schedule-conservativeness`                       | `1.0`    | Type: float                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--page-size`                                       | `128`    | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hybrid-kvcache-ratio`                            | `None`   | Optional[float]                |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--swa-full-tokens-ratio`                           | `0.8`    | Type: float                    |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-hybrid-swa-memory`                       | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--abort-on-priority-`<br/>`when-disabled`          | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-dynamic-chunking`                         | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Runtime options

| Argument                                           | Defaults | Options                   |                    A2                    |                    A3                    |
|----------------------------------------------------|----------|---------------------------|:----------------------------------------:|:----------------------------------------:|
| `--device`                                         | `None`   | Type: str                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tensor-parallel-size`<br/>`--tp-size`           | `1`      | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--pipeline-parallel-size`<br/>`--pp-size`         | `1`      | Type: int                 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--pp-max-micro-batch-size`                        | `None`   | Type: int                 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--pp-async-batch-depth`                           | `None`   | Type: int                 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--stream-interval`                                | `1`      | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--stream-output`                                  | `False`  | bool flag (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--random-seed`                                    | `None`   | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--constrained-json-`<br/>`whitespace-pattern`     | `None`   | Type: str                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--constrained-json-`<br/>`disable-any-whitespace` | `False`  | bool flag (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--watchdog-timeout`                               | `300`    | Type: float               | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--soft-watchdog-timeout`                          | `300`    | Type: float               | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--dist-timeout`                                   | `None`   | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--base-gpu-id`                                    | `0`      | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--gpu-id-step`                                    | `1`      | Type: int                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--sleep-on-idle`                                  | `False`  | bool flag (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--custom-sigquit-handler`                         | `None`   | Optional[Callable]        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Logging

| Argument                                           | Defaults          | Options                        |                    A2                    |                    A3                    |
|----------------------------------------------------|-------------------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--log-level`                                      | `info`            | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--log-level-http`                                 | `None`            | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--log-requests`                                   | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--log-requests-level`                             | `2`               | `0`, `1`, `2`, `3`             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--log-requests-format`                            | text              | text, json                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--crash-dump-folder`                              | `None`            | Type: str                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--crash-on-nan`                                   | `False`           | Type: str                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-metrics`                                 | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-metrics-for-`<br/>`all-schedulers`       | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tokenizer-metrics-`<br/>`custom-labels-header`  | `x-custom-labels` | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tokenizer-metrics-`<br/>`allowed-custom-labels` | `None`            | List[str]                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--bucket-time-to-`<br/>`first-token`              | `None`            | List[float]                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--bucket-inter-token-`<br/>`latency`              | `None`            | List[float]                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--bucket-e2e-request-`<br/>`latency`              | `None`            | List[float]                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--collect-tokens-`<br/>`histogram`                | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--prompt-tokens-buckets`                          | `None`            | List[str]                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--generation-tokens-buckets`                      | `None`            | List[str]                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--gc-warning-threshold-secs`                      | `0.0`             | Type: float                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--decode-log-interval`                            | `40`              | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-request-time-`<br/>`stats-logging`       | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--kv-events-config`                               | `None`            | Type: str                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-trace`                                   | `False`           | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--oltp-traces-endpoint`                           | `localhost:4317`  | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## RequestMetricsExporter configuration

| Argument                              | Defaults | Options                        |                    A2                    |                    A3                    |
|---------------------------------------|----------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--export-metrics-to-`<br/>`file`     | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--export-metrics-to-`<br/>`file-dir` | `None`   | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## API related

| Argument                | Defaults  | Options                        |                    A2                    |                    A3                    |
|-------------------------|-----------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--api-key`             | `None`    | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--admin-api-key`       | `None`    | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--served-model-name`   | `None`    | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--weight-version`      | `default` | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--chat-template`       | `None`    | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--completion-template` | `None`    | Type: str                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-cache-report` | `False`   | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--reasoning-parser`    | `None`    | `deepseek-r1`                  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tool-call-parser`    | `None`    | `llama`,`pythonic`             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--sampling-defaults`   | `model`   | `openai`, `model`              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--tool-server`         | `None`    | Type: str                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Data parallelism

| Argument                               | Defaults      | Options                                                   |                    A2                    |                    A3                    |
|----------------------------------------|---------------|-----------------------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--data-parallel-size`<br/>`--dp-size` | `1`           | Type: int                                                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--load-balance-method`                | `round_robin` | `round_robin`,<br/> `total_requests`,<br/> `total_tokens` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--prefill-round-robin-balance`        | `False`       | bool flag<br/> (set to enable)                            | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Multi-node distributed serving

| Argument                                  | Defaults | Options   |                    A2                    |                    A3                    |
|-------------------------------------------|----------|-----------|:----------------------------------------:|:----------------------------------------:|
| `--dist-init-addr`<br/>`--nccl-init-addr` | `None`   | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--nnodes`                                | `1`      | Type: int | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--node-rank`                             | `0`      | Type: int | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Model override args

| Argument                             | Defaults | Options   |                    A2                    |                    A3                    |
|--------------------------------------|----------|-----------|:----------------------------------------:|:----------------------------------------:|
| `--json-model-override-`<br/>`args`  | `{}`     | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--preferred-sampling-`<br/>`params` | `None`   | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## LoRA

| Argument                 | Defaults | Options                             |                    A2                    |                    A3                    |
|--------------------------|----------|-------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--enable-lora`          | `False`  | Bool flag <br/>(set to enable)      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-lora-rank`        | `None`   | Type: int                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--lora-target-modules`  | `None`   | `all`                               | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--lora-paths`           | `None`   | Type: List[str] /<br/> JSON objects | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-loras-per-batch`  | `8`      | Type: int                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-loaded-loras`     | `None`   | Type: int                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--lora-eviction-policy` | `lru`    | `lru`,<br/> `fifo`                  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--lora-backend`         | `triton` | `triton`                            | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--max-lora-chunk-size`  | `16`     | `16`, `32`,<br/> `64`, `128`        |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Kernel Backends (Attention, Sampling, Grammar, GEMM)

| Argument                               | Defaults          | Options                                                                                         |                    A2                    |                    A3                    |
|----------------------------------------|-------------------|-------------------------------------------------------------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--attention-backend`                  | `None`            | `ascend`                                                                                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--prefill-attention-backend`          | `None`            | `ascend`                                                                                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--decode-attention-backend`           | `None`            | `ascend`                                                                                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--sampling-backend`                   | `None`            | `pytorch`,<br/>`ascend`                                                                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--grammar-backend`                    | `None`            | `xgrammar`                                                                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--mm-attention-backend`               | `None`            | `ascend_attn`                                                                                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--nsa-prefill-backend`                | `flashmla_sparse` | `flashmla_sparse`,<br/> `flashmla_decode`,<br/> `fa3`,<br/> `tilelang`,<br/> `aiter`            |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--nsa-decode-backend`                 | `fa3`             | `flashmla_prefill`,<br/> `flashmla_kv`,<br/> `fa3`,<br/> `tilelang`,<br/> `aiter`               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--fp8-gemm-backend`                   | `auto`            | `auto`,<br/> `deep_gemm`,<br/> `flashinfer_trtllm`,<br/> `cutlass`,<br/> `triton`,<br/> `aiter` |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-flashinfer-`<br/>`autotune` | `False`           | bool flag<br/> (set to enable)                                                                  |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Speculative decoding

| Argument                                                         | Defaults  | Options                  |                    A2                    |                    A3                    |
|------------------------------------------------------------------|-----------|--------------------------|:----------------------------------------:|:----------------------------------------:|
| `--speculative-algorithm`                                        | `None`    | `EAGLE3`,<br/> `NEXTN`   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-draft-model-path`<br/>`--speculative-draft-model` | `None`    | Type: str                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-draft-model-`<br/>`revision`                      | `None`    | Type: str                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-draft-load-format`                                | `None`    | `auto`                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-num-steps`                                        | `None`    | Type: int                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-eagle-topk`                                       | `None`    | Type: int                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-num-draft-tokens`                                 | `None`    | Type: int                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-accept-`<br/>`threshold-single`                   | `1.0`     | Type: float              |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--speculative-accept-`<br/>`threshold-acc`                      | `1.0`     | Type: float              |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--speculative-token-map`                                        | `None`    | Type: str                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-attention-`<br/>`mode`                            | `prefill` | `prefill`,<br/> `decode` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-moe-runner-`<br/>`backend`                        | `None`    | `auto`                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-moe-a2a-`<br/>`backend`                           | `None`    | `ascend_fuseep`          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-draft-attention-backend`                          | `None`    | `ascend`                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--speculative-draft-model-quantization`                         | `None`    | `unquant`                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Ngram speculative decoding

| Argument                                           | Defaults   | Options            |                   A2                   |                   A3                   |
|----------------------------------------------------|------------|--------------------|:--------------------------------------:|:--------------------------------------:|
| `--speculative-ngram-`<br/>`min-match-window-size` | `1`        | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`max-match-window-size` | `12`       | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`min-bfs-breadth`       | `1`        | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`max-bfs-breadth`       | `10`       | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`match-type`            | `BFS`      | `BFS`,<br/> `PROB` | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`branch-length`         | `18`       | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--speculative-ngram-`<br/>`capacity`              | `10000000` | Type: int          | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Expert parallelism

| Argument                                              | Defaults  | Options                                     |                    A2                     |                    A3                    |
|-------------------------------------------------------|-----------|---------------------------------------------|:-----------------------------------------:|:----------------------------------------:|
| `--expert-parallel-size`<br/>`--ep-size`<br/>`--ep`   | `1`       | Type: int                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--moe-a2a-backend`                                   | `none`    | `none`,<br/> `deepep`,<br/> `ascend_fuseep` | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--moe-runner-backend`                                | `auto`    | `auto`, `triton`                            | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--flashinfer-mxfp4-`<br/>`moe-precision`             | `default` | `default`,<br/> `bf16`                      |  **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |
| `--enable-flashinfer-`<br/>`allreduce-fusion`         | `False`   | bool flag<br/> (set to enable)              |  **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |
| `--deepep-mode`                                       | `auto`    | `normal`, <br/>`low_latency`,<br/> `auto`   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--deepep-config`                                     | `None`    | Type: str                                   |  **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |
| `--ep-num-redundant-experts`                          | `0`       | Type: int                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--ep-dispatch-algorithm`                             | `None`    | Type: str                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--init-expert-location`                              | `trivial` | Type: str                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--enable-eplb`                                       | `False`   | bool flag<br/> (set to enable)              | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--eplb-algorithm`                                    | `auto`    | Type: str                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--eplb-rebalance-layers-`<br/>`per-chunk`            | `None`    | Type: int                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--eplb-min-rebalancing-`<br/>`utilization-threshold` | `1.0`     | Type: float                                 | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--expert-distribution-`<br/>`recorder-mode`          | `None`    | Type: str                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--expert-distribution-`<br/>`recorder-buffer-size`   | `None`    | Type: int                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--enable-expert-distribution-`<br/>`metrics`         | `False`   | bool flag <br/>(set to enable)              | ***<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--moe-dense-tp-size`                                 | `None`    | Type: int                                   | **<span style="color: green;">√</span>**  | **<span style="color: green;">√</span>** |
| `--elastic-ep-backend`                                | `None`    | `none`, `mooncake`                          |  **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |
| `--mooncake-ib-device`                                | `None`    | Type: str                                   |  **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |

## Mamba Cache

| Argument                     | Defaults  | Options                                       |                   A2                   |                   A3                   |
|------------------------------|-----------|-----------------------------------------------|:--------------------------------------:|:--------------------------------------:|
| `--max-mamba-cache-size`     | `None`    | Type: int                                     | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--mamba-ssm-dtype`          | `float32` | `float32`,<br/> `bfloat16`                    | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--mamba-full-memory-ratio`  | `0.2`     | Type: float                                   | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--mamba-scheduler-strategy` | `auto`    | `auto`, <br/>`no_buffer`,<br/> `extra_buffer` | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--mamba-track-interval`     | `256`     | Type: int                                     | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Hierarchical cache

| Argument                                        | Defaults        | Options                                                            |                    A2                    |                    A3                    |
|-------------------------------------------------|-----------------|--------------------------------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--enable-hierarchical-`<br/>`cache`            | `False`         | bool flag<br/> (set to enable)                                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-ratio`                               | `2.0`           | Type: float                                                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-size`                                | `0`             | Type: int                                                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-write-policy`                        | `write_through` | `write_back`,<br/> `write_through`,<br/> `write_through_selective` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--radix-eviction-policy`                       | `lru`           | `lru`, `lfu`                                                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-io-backend`                          | `kernel`        | `kernel_ascend`,<br/>`direct`                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-mem-layout`                          | `layer_first`   | `page_first_direct`,<br/> `page_first_kv_split`                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--hicache-storage-`<br/>`backend`              | `None`          | `file`                                                             |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--hicache-storage-`<br/>`prefetch-policy`      | `best_effort`   | `best_effort`,<br/> `wait_complete`,<br/> `timeout`                |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--hicache-storage-`<br/>`backend-extra-config` | `None`          | Type: str                                                          |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## LMCache

| Argument           | Defaults | Options                        |                   A2                   |                   A3                   |
|--------------------|----------|--------------------------------|:--------------------------------------:|:--------------------------------------:|
| `--enable-lmcache` | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Ktransformer server args

| Argument                                     | Defaults  | Options   |                   A2                   |                   A3                   |
|----------------------------------------------|-----------|-----------|:--------------------------------------:|:--------------------------------------:|
| `--kt-weight-path`                           | `None`    | Type: str | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--kt-method`                                | `AMXINT4` | Type: str | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--kt-cpuinfer`                              | `None`    | Type: int | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--kt-threadpool-count`                      | 2         | Type: int | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--kt-num-gpu-experts`                       | `None`    | Type: int | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--kt-max-deferred-`<br/>`experts-per-token` | `None`    | Type: int | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Double Sparsity

| Argument                              | Defaults | Options                        |                   A2                   |                   A3                   |
|---------------------------------------|----------|--------------------------------|:--------------------------------------:|:--------------------------------------:|
| `--enable-double-sparsity`            | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--ds-channel-config-path`            | `None`   | Type: str                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--ds-heavy-channel-num`              | `32`     | Type: int                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--ds-heavy-token-num`                | `256`    | Type: int                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--ds-heavy-channel-type`             | `qk`     | Type: str                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--ds-sparse-decode-`<br/>`threshold` | `4096`   | Type: int                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Offloading

| Argument                  | Defaults | Options   |                    A2                    |                    A3                    |
|---------------------------|----------|-----------|:----------------------------------------:|:----------------------------------------:|
| `--cpu-offload-gb`        | `0`      | Type: int | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--offload-group-size`    | `-1`     | Type: int |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--offload-num-in-group`  | `1`      | Type: int |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--offload-prefetch-step` | `1`      | Type: int |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--offload-mode`          | `cpu`    | Type: str |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Args for multi-item scoring

| Argument                         | Defaults | Options   |                   A2                   |                   A3                   |
|----------------------------------|----------|-----------|:--------------------------------------:|:--------------------------------------:|
| `--multi-item-scoring-delimiter` | `None`   | Type: int | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## Optimization/debug options

| Argument                                                | Defaults | Options                        |                    A2                    |                    A3                    |
|---------------------------------------------------------|----------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--disable-radix-cache`                                 | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--cuda-graph-max-bs`                                   | `None`   | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--cuda-graph-bs`                                       | `None`   | List[int]                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disable-cuda-graph`                                  | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disable-cuda-graph-`<br/>`padding`                   | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-profile-`<br/>`cuda-graph`                    | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-cudagraph-gc`                                 | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-nccl-nvls`                                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-symm-mem`                                     | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-flashinfer-`<br/>`cutlass-moe-fp4-allgather` | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-tokenizer-`<br/>`batch-encode`                | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disable-tokenizer-`<br/>`batch-encode`               | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disable-outlines-`<br/>`disk-cache`                  | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disable-custom-`<br/>`all-reduce`                    | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-mscclpp`                                      | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-torch-`<br/>`symm-mem`                        | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-overlap`<br/>`-schedule`                     | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-mixed-`<br/>`chunk`                           | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-dp-attention`                                 | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-dp-lm-head`                                   | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-two-`<br/>`batch-overlap`                     | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-single-`<br/>`batch-overlap`                  | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--tbo-token-`<br/>`distribution-threshold`             | `0.48`   | Type: float                    |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-torch-`<br/>`compile`                         | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-torch-`<br/>`compile-debug-mode`              | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-piecewise-`<br/>`cuda-graph`                  | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--piecewise-cuda-`<br/>`graph-tokens`                  | `None`   | Type: JSON<br/> list           |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--piecewise-cuda-`<br/>`graph-compiler`                | `eager`  | ["eager", "inductor"]          |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--torch-compile-max-bs`                                | `32`     | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--piecewise-cuda-`<br/>`graph-max-tokens`              | `4096`   | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--torchao-config`                                      | ``       | Type: str                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-nan-detection`                                | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-p2p-check`                                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--triton-attention-`<br/>`reduce-in-fp32`              | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--triton-attention-`<br/>`num-kv-splits`               | `8`      | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--triton-attention-`<br/>`split-tile-size`             | `None`   | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--num-continuous-`<br/>`decode-steps`                  | `1`      | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--delete-ckpt-`<br/>`after-loading`                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-memory-saver`                                 | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-weights-`<br/>`cpu-backup`                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-draft-weights-`<br/>`cpu-backup`              | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--allow-auto-truncate`                                 | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-custom-`<br/>`logit-processor`                | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--flashinfer-mla-`<br/>`disable-ragged`                | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-shared-`<br/>`experts-fusion`                | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-chunked-`<br/>`prefix-cache`                 | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disable-fast-`<br/>`image-processor`                 | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--keep-mm-feature-`<br/>`on-device`                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-return-`<br/>`hidden-states`                  | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-return-`<br/>`routed-experts`                 | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--scheduler-recv-`<br/>`interval`                      | `1`      | Type: int                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--numa-node`                                           | `None`   | List[int]                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--rl-on-policy-target`                                 | `None`   | `fsdp`                         |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-layerwise-`<br/>`nvtx-marker`                 | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-attn-tp-`<br/>`input-scattered`               | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--enable-nsa-prefill-`<br/>`context-parallel`          | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--enable-fused-qk-`<br/>`norm-rope`                    | `False`  | bool flag<br/> (set to enable) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Dynamic batch tokenizer

| Argument                                         | Defaults | Options                        |                    A2                    |                    A3                    |
|--------------------------------------------------|----------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--enable-dynamic-`<br/>`batch-tokenizer`        | `False`  | bool flag<br/> (set to enable) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--dynamic-batch-`<br/>`tokenizer-batch-size`    | `32`     | Type: int                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--dynamic-batch-`<br/>`tokenizer-batch-timeout` | `0.002`  | Type: float                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Debug tensor dumps

| Argument                                   | Defaults | Options   |                    A2                    |                    A3                    |
|--------------------------------------------|----------|-----------|:----------------------------------------:|:----------------------------------------:|
| `--debug-tensor-dump-`<br/>`output-folder` | `None`   | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--debug-tensor-dump-`<br/>`layers`        | `None`   | List[int] | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--debug-tensor-dump-`<br/>`input-file`    | `None`   | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## PD disaggregation

| Argument                                                | Defaults   | Options                               |                    A2                    |                    A3                    |
|---------------------------------------------------------|------------|---------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--disaggregation-mode`                                 | `null`     | `null`,<br/> `prefill`,<br/> `decode` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-transfer-backend`                     | `mooncake` | `ascend`                              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-bootstrap-port`                       | `8998`     | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-decode-tp`                            | `None`     | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-decode-dp`                            | `None`     | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-ib-device`                            | `None`     | Type: str                             |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| `--disaggregation-decode-`<br/>`enable-offload-kvcache` | `False`    | bool flag<br/> (set to enable)        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-decode-`<br/>`enable-fake-auto`       | `False`    | bool flag<br/> (set to enable)        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--num-reserved-decode-tokens`                          | `512`      | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--disaggregation-decode-`<br/>`polling-interval`       | `1`        | Type: int                             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Encode prefill disaggregation

| Argument                     | Defaults           | Options                                                        |                    A2                    |                    A3                    |
|------------------------------|--------------------|----------------------------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| `--encoder-only`             | `False`            | bool flag<br/> (set to enable)                                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--language-only`            | `False`            | bool flag<br/> (set to enable)                                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--encoder-transfer-backend` | `zmq_to_scheduler` | `zmq_to_scheduler`, <br/> `zmq_to_tokenizer`,<br/>  `mooncake` | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--encoder-urls`             | `[]`               | List[str]                                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Custom weight loader

| Argument                                                                | Defaults | Options                         |                    A2                    | A3                                       |
|-------------------------------------------------------------------------|----------|---------------------------------|:----------------------------------------:|------------------------------------------|
| `--custom-weight-loader`                                                | `None`   | List[str]                       |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |
| `--weight-loader-disable-`<br/>`mmap`                                   | `False`  | bool flag<br/> (set to enable)  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| `--remote-instance-weight-`<br/>`loader-seed-instance-ip`               | `None`   | Type: str                       |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |
| `--remote-instance-weight-`<br/>`loader-seed-instance-service-port`     | `None`   | Type: int                       |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |
| `--remote-instance-weight-`<br/>`loader-send-weights-group-ports`       | `None`   | Type: JSON<br/> list            |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |
| `--remote-instance-weight-`<br/>`loader-backend`                        | `nccl`   | `transfer_engine`, <br/> `nccl` |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |
| `--remote-instance-weight-`<br/>`loader-start-seed-via-transfer-engine` | `False`  | bool flag<br/> (set to enable)  |  **<span style="color: red;">×</span>**  | **<span style="color: red;">×</span>**   |

## For PD-Multiplexing

| Argument              | Defaults | Options                        |                   A2                   |                   A3                   |
|-----------------------|----------|--------------------------------|:--------------------------------------:|:--------------------------------------:|
| `--enable-pdmux`      | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--pdmux-config-path` | `None`   | Type: str                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--sm-group-num`      | `8`      | Type: int                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## For Multi-Modal

| Argument                                      | Defaults | Options                        | A2                                       | A3                                       |
|-----------------------------------------------|----------|--------------------------------|------------------------------------------|------------------------------------------|
| `--mm-max-concurrent-calls`                   | 32       | Type: int                      | **<span style="color: red;">×</span>**   | **<span style="color: red;">×</span>**   |
| `--mm-per-request-timeout`                    | 10.0     | Type: float                    | **<span style="color: red;">×</span>**   | **<span style="color: red;">×</span>**   |
| `--enable-broadcast-mm-`<br/>`inputs-process` | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>**   | **<span style="color: red;">×</span>**   |
| `--mm-process-config`                         | `None`   | Type: JSON / Dict              | **<span style="color: red;">×</span>**   | **<span style="color: red;">×</span>**   |
| `--mm-enable-dp-encoder`                      | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>**   | **<span style="color: red;">×</span>**   |
| `--limit-mm-data-per-request`                 | `None`   | Type: JSON / Dict              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## For checkpoint decryption

| Argument                        | Defaults | Options                        |                   A2                   |                   A3                   |
|---------------------------------|----------|--------------------------------|:--------------------------------------:|:--------------------------------------:|
| `--decrypted-config-file`       | `None`   | Type: str                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--decrypted-draft-config-file` | `None`   | Type: str                      | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
| `--enable-prefix-mm-cache`      | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## For deterministic inference

| Argument                                  | Defaults | Options                        | A2                                     | A3                                     |
|-------------------------------------------|----------|--------------------------------|----------------------------------------|----------------------------------------|
| `--enable-deterministic-`<br/>`inference` | `False`  | bool flag<br/> (set to enable) | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |

## For registering hooks

| Argument          | Defaults | Options         | A2                                       | A3                                       |
|-------------------|----------|-----------------|------------------------------------------|------------------------------------------|
| `--forward-hooks` | `None`   | Type: JSON list | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Configuration file support

| Argument   | Defaults | Options   | A2                                       | A3                                       |
|------------|----------|-----------|------------------------------------------|------------------------------------------|
| `--config` | `None`   | Type: str | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
