# Support Features on Ascend NPU

This section describes the basic functions and features supported by the Ascend NPU.If you encounter issues or have any
questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

If you want to know the meaning and usage of each parameter,
click [Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html).

## Model and tokenizer

| Argument                               | Defaults | Options                               | Server supported |
|----------------------------------------|----------|---------------------------------------|:----------------:|
| `--model-path`<br/>`--model`           | `None`   | Type: str                             |      A2, A3      |
| `--tokenizer-path`                     | `None`   | Type: str                             |      A2, A3      |
| `--tokenizer-mode`                     | `auto`   | `auto`, `slow`                        |      A2, A3      |
| `--tokenizer-worker-num`               | `1`      | Type: int                             |      A2, A3      |
| `--skip-tokenizer-init`                | `False`  | bool flag (set to enable)             |      A2, A3      |
| `--load-format`                        | `auto`   | `auto`, `safetensors`                 |      A2, A3      |
| `--model-loader-` <br/> `extra-config` | {}       | Type: str                             |      A2, A3      |
| `--trust-remote-code`                  | `False`  | bool flag (set to enable)             |      A2, A3      |
| `--context-length`                     | `None`   | Type: int                             |      A2, A3      |
| `--is-embedding`                       | `False`  | bool flag (set to enable)             |      A2, A3      |
| `--enable-multimodal`                  | `None`   | bool flag (set to enable)             |      A2, A3      |
| `--revision`                           | `None`   | Type: str                             |      A2, A3      |
| `--model-impl`                         | `auto`   | `auto`, `sglang`,<br/> `transformers` |      A2, A3      |

## HTTP server

| Argument               | Defaults    | Options                   | Server supported |
|------------------------|-------------|---------------------------|:----------------:|
| `--host`               | `127.0.0.1` | Type: str                 |      A2, A3      |
| `--port`               | `30000`     | Type: int                 |      A2, A3      |
| `--skip-server-warmup` | `False`     | bool flag (set to enable) |      A2, A3      |
| `--warmups`            | `None`      | Type: str                 |      A2, A3      |
| `--nccl-port`          | `None`      | Type: int                 |      A2, A3      |
| `--fastapi-root-path`  | `None`      | Type: str                 |      A2, A3      |
| `--grpc-mode`          | `False`     | bool flag (set to enable) |      A2, A3      |

## Quantization and data type

| Argument                                    | Defaults | Options                                 | Server supported |
|---------------------------------------------|----------|-----------------------------------------|:----------------:|
| `--dtype`                                   | `auto`   | `auto`,<br/> `float16`,<br/> `bfloat16` |      A2, A3      |
| `--quantization`                            | `None`   | `modelslim`                             |      A2, A3      |
| `--quantization-param-path`                 | `None`   | Type: str                               | Special For GPU  |
| `--kv-cache-dtype`                          | `auto`   | `auto`                                  |      A2, A3      |
| `--enable-fp32-lm-head`                     | `False`  | bool flag <br/> (set to enable)         |      A2, A3      |
| `--modelopt-quant`                          | `None`   | Type: str                               | Special For GPU  |
| `--modelopt-checkpoint-`<br/>`restore-path` | `None`   | Type: str                               | Special For GPU  |
| `--modelopt-checkpoint-`<br/>`save-path`    | `None`   | Type: str                               | Special For GPU  |
| `--modelopt-export-path`                    | `None`   | Type: str                               | Special For GPU  |
| `--quantize-and-serve`                      | `False`  | bool flag <br/> (set to enable)         | Special For GPU  |
| `--rl-quant-profile`                        | `None`   | Type: str                               | Special For GPU  |

## Memory and scheduling

| Argument                                            | Defaults | Options                        | Server supported |
|-----------------------------------------------------|----------|--------------------------------|:----------------:|
| `--mem-fraction-static`                             | `None`   | Type: float                    |      A2, A3      |
| `--max-running-requests`                            | `None`   | Type: int                      |      A2, A3      |
| `--prefill-max-requests`                            | `None`   | Type: int                      |      A2, A3      |
| `--max-queued-requests`                             | `None`   | Type: int                      |      A2, A3      |
| `--max-total-tokens`                                | `None`   | Type: int                      |      A2, A3      |
| `--chunked-prefill-size`                            | `None`   | Type: int                      |      A2, A3      |
| `--max-prefill-tokens`                              | `16384`  | Type: int                      |      A2, A3      |
| `--schedule-policy`                                 | `fcfs`   | `lpm`, `fcfs`                  |      A2, A3      |
| `--enable-priority-`<br/>`scheduling`               | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--schedule-low-priority-`<br/>`values-first`       | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--priority-scheduling-`<br/>`preemption-threshold` | `10`     | Type: int                      |      A2, A3      |
| `--schedule-conservativeness`                       | `1.0`    | Type: float                    |      A2, A3      |
| `--page-size`                                       | `128`    | Type: int                      |      A2, A3      |
| `--swa-full-tokens-ratio`                           | `0.8`    | Type: float                    |      A2, A3      |
| `--disable-hybrid-swa-memory`                       | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--abort-on-priority-`<br/>`when-disabled`          | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-dynamic-chunking`                         | `False`  | bool flag<br/> (set to enable) |      A2, A3      |

## Runtime options

| Argument                                           | Defaults | Options                   | Server supported |
|----------------------------------------------------|----------|---------------------------|:----------------:|
| `--device`                                         | `None`   | Type: str                 |      A2, A3      |
| `--tensor-parallel-size`<br/>`--tp-size`           | `1`      | Type: int                 |      A2, A3      |
| `--pipeline-parallel-size`<br/>`--pp-size`         | `1`      | Type: int                 |      A2, A3      |
| `--pp-max-micro-batch-size`                        | `None`   | Type: int                 |      A2, A3      |
| `--pp-async-batch-depth`                           | `None`   | Type: int                 |      A2, A3      |
| `--stream-interval`                                | `1`      | Type: int                 |      A2, A3      |
| `--stream-output`                                  | `False`  | bool flag (set to enable) |      A2, A3      |
| `--random-seed`                                    | `None`   | Type: int                 |      A2, A3      |
| `--constrained-json-`<br/>`whitespace-pattern`     | `None`   | Type: str                 |      A2, A3      |
| `--constrained-json-`<br/>`disable-any-whitespace` | `False`  | bool flag (set to enable) |      A2, A3      |
| `--watchdog-timeout`                               | `300`    | Type: float               |      A2, A3      |
| `--soft-watchdog-timeout`                          | `300`    | Type: float               |      A2, A3      |
| `--dist-timeout`                                   | `None`   | Type: int                 |      A2, A3      |
| `--base-gpu-id`                                    | `0`      | Type: int                 |      A2, A3      |
| `--gpu-id-step`                                    | `1`      | Type: int                 |      A2, A3      |
| `--sleep-on-idle`                                  | `False`  | bool flag (set to enable) |      A2, A3      |
| `--custom-sigquit-handler`                         | `None`   | Optional[Callable]        |      A2, A3      |

## Logging

| Argument                                           | Defaults          | Options                        | Server supported |
|----------------------------------------------------|-------------------|--------------------------------|:----------------:|
| `--log-level`                                      | `info`            | Type: str                      |      A2, A3      |
| `--log-level-http`                                 | `None`            | Type: str                      |      A2, A3      |
| `--log-requests`                                   | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--log-requests-level`                             | `2`               | `0`, `1`, `2`, `3`             |      A2, A3      |
| `--log-requests-format`                            | text              | text, json                     |      A2, A3      |
| `--crash-dump-folder`                              | `None`            | Type: str                      |      A2, A3      |
| `--enable-metrics`                                 | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-metrics-for-`<br/>`all-schedulers`       | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--tokenizer-metrics-`<br/>`custom-labels-header`  | `x-custom-labels` | Type: str                      |      A2, A3      |
| `--tokenizer-metrics-`<br/>`allowed-custom-labels` | `None`            | List[str]                      |      A2, A3      |
| `--bucket-time-to-`<br/>`first-token`              | `None`            | List[float]                    |      A2, A3      |
| `--bucket-inter-token-`<br/>`latency`              | `None`            | List[float]                    |      A2, A3      |
| `--bucket-e2e-request-`<br/>`latency`              | `None`            | List[float]                    |      A2, A3      |
| `--collect-tokens-`<br/>`histogram`                | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--prompt-tokens-buckets`                          | `None`            | List[str]                      |      A2, A3      |
| `--generation-tokens-buckets`                      | `None`            | List[str]                      |      A2, A3      |
| `--gc-warning-threshold-secs`                      | `0.0`             | Type: float                    |      A2, A3      |
| `--decode-log-interval`                            | `40`              | Type: int                      |      A2, A3      |
| `--enable-request-time-`<br/>`stats-logging`       | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--kv-events-config`                               | `None`            | Type: str                      | Special for GPU  |
| `--enable-trace`                                   | `False`           | bool flag<br/> (set to enable) |      A2, A3      |
| `--oltp-traces-endpoint`                           | `localhost:4317`  | Type: str                      |      A2, A3      |

## RequestMetricsExporter configuration

| Argument                              | Defaults | Options                        | Server supported |
|---------------------------------------|----------|--------------------------------|:----------------:|
| `--export-metrics-to-`<br/>`file`     | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--export-metrics-to-`<br/>`file-dir` | `None`   | Type: str                      |      A2, A3      |

## API related

| Argument                | Defaults  | Options                        | Server supported |
|-------------------------|-----------|--------------------------------|:----------------:|
| `--api-key`             | `None`    | Type: str                      |      A2, A3      |
| `--admin-api-key`       | `None`    | Type: str                      |      A2, A3      |
| `--served-model-name`   | `None`    | Type: str                      |      A2, A3      |
| `--weight-version`      | `default` | Type: str                      |      A2, A3      |
| `--chat-template`       | `None`    | Type: str                      |      A2, A3      |
| `--completion-template` | `None`    | Type: str                      |      A2, A3      |
| `--enable-cache-report` | `False`   | bool flag<br/> (set to enable) |      A2, A3      |
| `--reasoning-parser`    | `None`    | `deepseek-r1`                  |      A2, A3      |
| `--tool-call-parser`    | `None`    | `llama`,`pythonic`             |      A2, A3      |
| `--sampling-defaults`   | `model`   | `openai`, `model`              |      A2, A3      |

## Data parallelism

| Argument                               | Defaults      | Options                                                   | Server supported |
|----------------------------------------|---------------|-----------------------------------------------------------|:----------------:|
| `--data-parallel-size`<br/>`--dp-size` | `1`           | Type: int                                                 |      A2, A3      |
| `--load-balance-method`                | `round_robin` | `round_robin`,<br/> `total_requests`,<br/> `total_tokens` |      A2, A3      |
| `--prefill-round-robin-balance`        | `False`       | bool flag<br/> (set to enable)                            |      A2, A3      |

## Multi-node distributed serving

| Argument                                  | Defaults | Options   | Server supported |
|-------------------------------------------|----------|-----------|:----------------:|
| `--dist-init-addr`<br/>`--nccl-init-addr` | `None`   | Type: str |      A2, A3      |
| `--nnodes`                                | `1`      | Type: int |      A2, A3      |
| `--node-rank`                             | `0`      | Type: int |      A2, A3      |

## Model override args

| Argument                             | Defaults | Options   | Server supported |
|--------------------------------------|----------|-----------|:----------------:|
| `--json-model-override-`<br/>`args`  | `{}`     | Type: str |      A2, A3      |
| `--preferred-sampling-`<br/>`params` | `None`   | Type: str |      A2, A3      |

## LoRA

| Argument                 | Defaults | Options                             | Server supported |
|--------------------------|----------|-------------------------------------|:----------------:|
| `--enable-lora`          | `False`  | Bool flag <br/>(set to enable)      |      A2, A3      |
| `--max-lora-rank`        | `None`   | Type: int                           |      A2, A3      |
| `--lora-target-modules`  | `None`   | `all`                               |      A2, A3      |
| `--lora-paths`           | `None`   | Type: List[str] /<br/> JSON objects |      A2, A3      |
| `--max-loras-per-batch`  | `8`      | Type: int                           |      A2, A3      |
| `--max-loaded-loras`     | `None`   | Type: int                           |      A2, A3      |
| `--lora-eviction-policy` | `lru`    | `lru`,<br/> `fifo`                  |      A2, A3      |
| `--lora-backend`         | `triton` | `triton`                            |      A2, A3      |
| `--max-lora-chunk-size`  | `16`     | `16`, `32`,<br/> `64`, `128`        | Special for GPU  |

## Kernel Backends (Attention, Sampling, Grammar, GEMM)

| Argument                               | Defaults          | Options                                                                                        | Server supported |
|----------------------------------------|-------------------|------------------------------------------------------------------------------------------------|:----------------:|
| `--attention-backend`                  | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--prefill-attention-backend`          | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--decode-attention-backend`           | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--sampling-backend`                   | `None`            | `pytorch`,<br/>`ascend`                                                                        |      A2, A3      |
| `--grammar-backend`                    | `None`            | `xgrammar`                                                                                     |      A2, A3      |
| `--mm-attention-backend`               | `None`            | `ascend_attn`                                                                                  |      A2, A3      |
| `--nsa-prefill-backend`                | `flashmla_sparse` | `flashmla_sparse`,<br/> `flashmla_decode`,<br/>`fa3`,<br/> `tilelang`,<br/> `aiter`            | Special for GPU  |
| `--nsa-decode-backend`                 | `fa3`             | `flashmla_prefill`,<br/> `flashmla_kv`,<br/> `fa3`,<br/>`tilelang`,<br/> `aiter`               | Special for GPU  |
| `--fp8-gemm-backend`                   | `auto`            | `auto`,<br/> `deep_gemm`,<br/> `flashinfer_trtllm`,<br/>`cutlass`,<br/> `triton`,<br/> `aiter` | Special for GPU  |
| `--disable-flashinfer-`<br/>`autotune` | `False`           | bool flag<br/> (set to enable)                                                                 | Special for GPU  |

## Speculative decoding

| Argument                                                         | Defaults  | Options                  | Server supported |
|------------------------------------------------------------------|-----------|--------------------------|:----------------:|
| `--speculative-algorithm`                                        | `None`    | `EAGLE3`,<br/> `NEXTN`   |      A2, A3      |
| `--speculative-draft-model-path`<br/>`--speculative-draft-model` | `None`    | Type: str                |      A2, A3      |
| `--speculative-draft-model-`<br/>`revision`                      | `None`    | Type: str                |      A2, A3      |
| `--speculative-draft-load-format`                                | `None`    | `auto`                   |      A2, A3      |
| `--speculative-num-steps`                                        | `None`    | Type: int                |      A2, A3      |
| `--speculative-eagle-topk`                                       | `None`    | Type: int                |      A2, A3      |
| `--speculative-num-draft-tokens`                                 | `None`    | Type: int                |      A2, A3      |
| `--speculative-accept-`<br/>`threshold-single`                   | `1.0`     | Type: float              | Special for GPU  |
| `--speculative-accept-`<br/>`threshold-acc`                      | `1.0`     | Type: float              | Special for GPU  |
| `--speculative-token-map`                                        | `None`    | Type: str                |      A2, A3      |
| `--speculative-attention-`<br/>`mode`                            | `prefill` | `prefill`,<br/> `decode` |      A2, A3      |
| `--speculative-moe-runner-`<br/>`backend`                        | `None`    | `auto`                   |      A2, A3      |
| `--speculative-moe-a2a-`<br/>`backend`                           | `None`    | `ascend_fuseep`          |      A2, A3      |
| `--speculative-draft-attention-backend`                          | `None`    | `ascend`                 |      A2, A3      |
| `--speculative-draft-model-quantization`                         | `None`    | `unquant`                |      A2, A3      |

## Ngram speculative decoding

| Argument                                           | Defaults   | Options            | Server supported |
|----------------------------------------------------|------------|--------------------|:----------------:|
| `--speculative-ngram-`<br/>`min-match-window-size` | `1`        | Type: int          |   Experimental   |
| `--speculative-ngram-`<br/>`max-match-window-size` | `12`       | Type: int          |   Experimental   |
| `--speculative-ngram-`<br/>`min-bfs-breadth`       | `1`        | Type: int          |   Experimental   |
| `--speculative-ngram-`<br/>`max-bfs-breadth`       | `10`       | Type: int          |   Experimental   |
| `--speculative-ngram-`<br/>`match-type`            | `BFS`      | `BFS`,<br/> `PROB` |   Experimental   |
| `--speculative-ngram-`<br/>`branch-length`         | `18`       | Type: int          |   Experimental   |
| `--speculative-ngram-`<br/>`capacity`              | `10000000` | Type: int          |   Experimental   |

## Expert parallelism

| Argument                                              | Defaults  | Options                                     | Server supported |
|-------------------------------------------------------|-----------|---------------------------------------------|:----------------:|
| `--expert-parallel-size`<br/>`--ep-size`<br/>`--ep`   | `1`       | Type: int                                   |      A2, A3      |
| `--moe-a2a-backend`                                   | `none`    | `none`,<br/> `deepep`,<br/> `ascend_fuseep` |      A2, A3      |
| `--moe-runner-backend`                                | `auto`    | `auto`, `triton`                            |      A2, A3      |
| `--flashinfer-mxfp4-`<br/>`moe-precision`             | `default` | `default`,<br/> `bf16`                      | Special for GPU  |
| `--enable-flashinfer-`<br/>`allreduce-fusion`         | `False`   | bool flag<br/> (set to enable)              | Special for GPU  |
| `--deepep-mode`                                       | `auto`    | `normal`, <br/>`low_latency`,<br/> `auto`   |      A2, A3      |
| `--deepep-config`                                     | `None`    | Type: str                                   | Special for GPU  |
| `--ep-num-redundant-experts`                          | `0`       | Type: int                                   |      A2, A3      |
| `--ep-dispatch-algorithm`                             | `None`    | Type: str                                   |      A2, A3      |
| `--init-expert-location`                              | `trivial` | Type: str                                   |      A2, A3      |
| `--enable-eplb`                                       | `False`   | bool flag<br/> (set to enable)              |      A2, A3      |
| `--eplb-algorithm`                                    | `auto`    | Type: str                                   |      A2, A3      |
| `--eplb-rebalance-layers-`<br/>`per-chunk`            | `None`    | Type: int                                   |      A2, A3      |
| `--eplb-min-rebalancing-`<br/>`utilization-threshold` | `1.0`     | Type: float                                 |      A2, A3      |
| `--expert-distribution-`<br/>`recorder-mode`          | `None`    | Type: str                                   |      A2, A3      |
| `--expert-distribution-`<br/>`recorder-buffer-size`   | `None`    | Type: int                                   |      A2, A3      |
| `--enable-expert-distribution-`<br/>`metrics`         | `False`   | bool flag (set to enable)                   |      A2, A3      |
| `--moe-dense-tp-size`                                 | `None`    | Type: int                                   |      A2, A3      |
| `--elastic-ep-backend`                                | `None`    | `none`, `mooncake`                          | Special for GPU  |
| `--mooncake-ib-device`                                | `None`    | Type: str                                   | Special for GPU  |

## Mamba Cache

| Argument                     | Defaults  | Options                                       | Server supported |
|------------------------------|-----------|-----------------------------------------------|:----------------:|
| `--max-mamba-cache-size`     | `None`    | Type: int                                     |      A2, A3      |
| `--mamba-ssm-dtype`          | `float32` | `float32`,<br/> `bfloat16`                    |      A2, A3      |
| `--mamba-full-memory-ratio`  | `0.2`     | Type: float                                   |      A2, A3      |
| `--mamba-scheduler-strategy` | `auto`    | `auto`, <br/>`no_buffer`,<br/> `extra_buffer` |      A2, A3      |
| `--mamba-track-interval`     | `256`     | Type: int                                     |      A2, A3      |

## Hierarchical cache

| Argument                                        | Defaults        | Options                                                             | Server supported |
|-------------------------------------------------|-----------------|---------------------------------------------------------------------|:----------------:|
| `--enable-hierarchical-`<br/>`cache`            | `False`         | bool flag<br/> (set to enable)                                      |      A2, A3      |
| `--hicache-ratio`                               | `2.0`           | Type: float                                                         |      A2, A3      |
| `--hicache-size`                                | `0`             | Type: int                                                           |      A2, A3      |
| `--hicache-write-policy`                        | `write_through` | `write_back`,<br/> `write_through`,<br/>  `write_through_selective` |      A2, A3      |
| `--radix-eviction-policy`                       | `lru`           | `lru`,                                     `lfu`                    |      A2, A3      |
| `--hicache-io-backend`                          | `kernel`        | `kernel_ascend`,<br/>                     `direct`                  |      A2, A3      |
| `--hicache-mem-layout`                          | `layer_first`   | `page_first_direct`,<br/>                  `page_first_kv_split`    |      A2, A3      |
| `--hicache-storage-`<br/>`backend`              | `None`          | `file`                                                              |      A2, A3      |
| `--hicache-storage-`<br/>`prefetch-policy`      | `best_effort`   | `best_effort`,<br/> `wait_complete`,<br/>  `timeout`                | Special for GPU  |
| `--hicache-storage-`<br/>`backend-extra-config` | `None`          | Type: str                                                           | Special for GPU  |

## LMCache

| Argument           | Defaults | Options                        | Server supported |
|--------------------|----------|--------------------------------|:----------------:|
| `--enable-lmcache` | `False`  | bool flag<br/> (set to enable) | Special for GPU  |

## Offloading

| Argument                  | Defaults | Options   | Server supported |
|---------------------------|----------|-----------|:----------------:|
| `--cpu-offload-gb`        | `0`      | Type: int |      A2, A3      |
| `--offload-group-size`    | `-1`     | Type: int |      A2, A3      |
| `--offload-num-in-group`  | `1`      | Type: int |      A2, A3      |
| `--offload-prefetch-step` | `1`      | Type: int |      A2, A3      |
| `--offload-mode`          | `cpu`    | Type: str |      A2, A3      |

## Args for multi-item scoring

| Argument                         | Defaults | Options   | Server supported |
|----------------------------------|----------|-----------|:----------------:|
| `--multi-item-scoring-delimiter` | `None`   | Type: int |      A2, A3      |

## Optimization/debug options

| Argument                                                | Defaults | Options                        | Server supported |
|---------------------------------------------------------|----------|--------------------------------|:----------------:|
| `--disable-radix-cache`                                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--cuda-graph-max-bs`                                   | `None`   | Type: int                      |      A2, A3      |
| `--cuda-graph-bs`                                       | `None`   | List[int]                      |      A2, A3      |
| `--disable-cuda-graph`                                  | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-cuda-graph-`<br/>`padding`                   | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-profile-`<br/>`cuda-graph`                    | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-cudagraph-gc`                                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-nccl-nvls`                                    | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--enable-symm-mem`                                     | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--disable-flashinfer-`<br/>`cutlass-moe-fp4-allgather` | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--enable-tokenizer-`<br/>`batch-encode`                | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-tokenizer-`<br/>`batch-encode`               | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-outlines-`<br/>`disk-cache`                  | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-custom-`<br/>`all-reduce`                    | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-mscclpp`                                      | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--enable-torch-`<br/>`symm-mem`                        | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--disable-overlap`<br/>`-schedule`                     | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-mixed-`<br/>`chunk`                           | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-dp-attention`                                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-dp-lm-head`                                   | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-two-`<br/>`batch-overlap`                     | `False`  | bool flag<br/> (set to enable) |     Planned      |
| `--enable-single-`<br/>`batch-overlap`                  | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--tbo-token-`<br/>`distribution-threshold`             | `0.48`   | Type: float                    |     Planned      |
| `--enable-torch-`<br/>`compile`                         | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-torch-`<br/>`compile-debug-mode`              | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-piecewise-`<br/>`cuda-graph`                  | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-tokens`                  | `None`   | Type: JSON<br/> list           |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-compiler`                | `eager`  | ["eager", "inductor"]          |      A2, A3      |
| `--torch-compile-max-bs`                                | `32`     | Type: int                      |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-max-tokens`              | `4096`   | Type: int                      |      A2, A3      |
| `--torchao-config`                                      | ``       | Type: str                      | Special for GPU  |
| `--enable-nan-detection`                                | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-p2p-check`                                    | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--triton-attention-`<br/>`reduce-in-fp32`              | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--triton-attention-`<br/>`num-kv-splits`               | `8`      | Type: int                      | Special for GPU  |
| `--triton-attention-`<br/>`split-tile-size`             | `None`   | Type: int                      | Special for GPU  |
| `--delete-ckpt-`<br/>`after-loading`                    | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-memory-saver`                                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-weights-`<br/>`cpu-backup`                    | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-draft-weights-`<br/>`cpu-backup`              | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--allow-auto-truncate`                                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-custom-`<br/>`logit-processor`                | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--flashinfer-mla-`<br/>`disable-ragged`                | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--disable-shared-`<br/>`experts-fusion`                | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-chunked-`<br/>`prefix-cache`                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--disable-fast-`<br/>`image-processor`                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--keep-mm-feature-`<br/>`on-device`                    | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-return-`<br/>`hidden-states`                  | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-return-`<br/>`routed-experts`                 | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--scheduler-recv-`<br/>`interval`                      | `1`      | Type: int                      |      A2, A3      |
| `--numa-node`                                           | `None`   | List[int]                      |      A2, A3      |
| `--rl-on-policy-target`                                 | `None`   | `fsdp`                         |     Planned      |
| `--enable-layerwise-`<br/>`nvtx-marker`                 | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--enable-attn-tp-`<br/>`input-scattered`               | `False`  | bool flag<br/> (set to enable) |   Experimental   |
| `--enable-nsa-prefill-`<br/>`context-parallel`          | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--enable-fused-qk-`<br/>`norm-rope`                    | `False`  | bool flag<br/> (set to enable) | Special for GPU  |

## Dynamic batch tokenizer

| Argument                                         | Defaults | Options                        | Server supported |
|--------------------------------------------------|----------|--------------------------------|:----------------:|
| `--enable-dynamic-`<br/>`batch-tokenizer`        | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--dynamic-batch-`<br/>`tokenizer-batch-size`    | `32`     | Type: int                      |      A2, A3      |
| `--dynamic-batch-`<br/>`tokenizer-batch-timeout` | `0.002`  | Type: float                    |      A2, A3      |

## Debug tensor dumps

| Argument                                   | Defaults | Options   | Server supported |
|--------------------------------------------|----------|-----------|:----------------:|
| `--debug-tensor-dump-`<br/>`output-folder` | `None`   | Type: str |      A2, A3      |
| `--debug-tensor-dump-`<br/>`layers`        | `None`   | List[int] |      A2, A3      |
| `--debug-tensor-dump-`<br/>`input-file`    | `None`   | Type: str |      A2, A3      |

## PD disaggregation

| Argument                                                | Defaults   | Options                               | Server supported |
|---------------------------------------------------------|------------|---------------------------------------|:----------------:|
| `--disaggregation-mode`                                 | `null`     | `null`,<br/> `prefill`,<br/> `decode` |      A2, A3      |
| `--disaggregation-transfer-backend`                     | `mooncake` | `ascend`                              |      A2, A3      |
| `--disaggregation-bootstrap-port`                       | `8998`     | Type: int                             |      A2, A3      |
| `--disaggregation-decode-tp`                            | `None`     | Type: int                             |      A2, A3      |
| `--disaggregation-decode-dp`                            | `None`     | Type: int                             |      A2, A3      |
| `--disaggregation-ib-device`                            | `None`     | Type: str                             | Special for GPU  |
| `--disaggregation-decode-`<br/>`enable-offload-kvcache` | `False`    | bool flag<br/> (set to enable)        |      A2, A3      |
| `--disaggregation-decode-`<br/>`enable-fake-auto`       | `False`    | bool flag<br/> (set to enable)        |      A2, A3      |
| `--num-reserved-decode-tokens`                          | `512`      | Type: int                             |      A2, A3      |
| `--disaggregation-decode-`<br/>`polling-interval`       | `1`        | Type: int                             |      A2, A3      |

## Encode prefill disaggregation

| Argument                     | Defaults           | Options                                                        | Server supported |
|------------------------------|--------------------|----------------------------------------------------------------|:----------------:|
| `--encoder-only`             | `False`            | bool flag<br/> (set to enable)                                 |      A2, A3      |
| `--language-only`            | `False`            | bool flag<br/> (set to enable)                                 |      A2, A3      |
| `--encoder-transfer-backend` | `zmq_to_scheduler` | `zmq_to_scheduler`, <br/> `zmq_to_tokenizer`,<br/>  `mooncake` |      A2, A3      |
| `--encoder-urls`             | `[]`               | List[str]                                                      |      A2, A3      |

## Custom weight loader

| Argument                                                                | Defaults | Options                         | Server supported |
|-------------------------------------------------------------------------|----------|---------------------------------|:----------------:|
| `--custom-weight-loader`                                                | `None`   | List[str]                       |      A2, A3      |
| `--weight-loader-disable-`<br/>`mmap`                                   | `False`  | bool flag<br/> (set to enable)  |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-seed-instance-ip`               | `None`   | Type: str                       |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-seed-instance-service-port`     | `None`   | Type: int                       |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-send-weights-group-ports`       | `None`   | Type: JSON<br/> list            |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-backend`                        | `nccl`   | `transfer_engine`, <br/> `nccl` |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-start-seed-via-transfer-engine` | `False`  | bool flag<br/> (set to enable)  | Special for GPU  |

## For PD-Multiplexing

| Argument              | Defaults | Options                        | Server supported |
|-----------------------|----------|--------------------------------|:----------------:|
| `--enable-pdmux`      | `False`  | bool flag<br/> (set to enable) | Special for GPU  |
| `--pdmux-config-path` | `None`   | Type: str                      | Special for GPU  |
| `--sm-group-num`      | `8`      | Type: int                      | Special for GPU  |

## For Multi-Modal

| Argument                                      | Defaults | Options                        | Server supported |
|-----------------------------------------------|----------|--------------------------------|:----------------:|
| `--mm-max-concurrent-calls`                   | 32       | Type: int                      |      A2, A3      |
| `--mm-per-request-timeout`                    | 10.0     | Type: float                    |      A2, A3      |
| `--enable-broadcast-mm-`<br/>`inputs-process` | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--mm-process-config`                         | `None`   | Type: JSON / Dict              |      A2, A3      |
| `--mm-enable-dp-encoder`                      | `False`  | bool flag<br/> (set to enable) |      A2, A3      |
| `--limit-mm-data-per-request`                 | `None`   | Type: JSON / Dict              |      A2, A3      |

## For checkpoint decryption

| Argument                        | Defaults | Options                        | Server supported |
|---------------------------------|----------|--------------------------------|:----------------:|
| `--decrypted-config-file`       | `None`   | Type: str                      |      A2, A3      |
| `--decrypted-draft-config-file` | `None`   | Type: str                      |      A2, A3      |
| `--enable-prefix-mm-cache`      | `False`  | bool flag<br/> (set to enable) |      A2, A3      |

## For deterministic inference

| Argument                                  | Defaults | Options                        | Server supported |
|-------------------------------------------|----------|--------------------------------|:----------------:|
| `--enable-deterministic-`<br/>`inference` | `False`  | bool flag<br/> (set to enable) |     Planned      |

## For registering hooks

| Argument          | Defaults | Options         | Server supported |
|-------------------|----------|-----------------|:----------------:|
| `--forward-hooks` | `None`   | Type: JSON list |      A2, A3      |

## Configuration file support

| Argument   | Defaults | Options   | Server supported |
|------------|----------|-----------|:----------------:|
| `--config` | `None`   | Type: str |      A2, A3      |

## Other Params

The following parameters are not supported because the third-party components that depend on are not compatible with the
NPU, like Ktransformer, checkpoint-engine etc.

| Argument                                                          | Defaults  | Options                   |
|-------------------------------------------------------------------|-----------|---------------------------|
| `--checkpoint-engine-` <br/> `wait-weights-` <br/> `before-ready` | `False`   | bool flag (set to enable) |
| `--kt-weight-path`                                                | `None`    | Type: str                 |
| `--kt-method`                                                     | `AMXINT4` | Type: str                 |
| `--kt-cpuinfer`                                                   | `None`    | Type: int                 |
| `--kt-threadpool-count`                                           | 2         | Type: int                 |
| `--kt-num-gpu-experts`                                            | `None`    | Type: int                 |
| `--kt-max-deferred-`<br/>`experts-per-token`                      | `None`    | Type: int                 |

The following parameters have some functional deficiencies on community

| Argument                              | Defaults | Options                        |
|---------------------------------------|----------|--------------------------------|
| `--enable-double-sparsity`            | `False`  | bool flag<br/> (set to enable) |
| `--ds-channel-config-path`            | `None`   | Type: str                      |
| `--ds-heavy-channel-num`              | `32`     | Type: int                      |
| `--ds-heavy-token-num`                | `256`    | Type: int                      |
| `--ds-heavy-channel-type`             | `qk`     | Type: str                      |
| `--ds-sparse-decode-`<br/>`threshold` | `4096`   | Type: int                      |
| `--tool-server`                       | `None`   | Type: str                      |
