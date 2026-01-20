# Server Arguments

This page provides a list of server arguments used in the command line to configure the behavior
and performance of the language model server during deployment. These arguments enable users to
customize key aspects of the server, including model selection, parallelism policies,
memory management, and optimization techniques.
You can find all arguments by `python3 -m sglang.launch_server --help`

## Common launch commands

- To use a configuration file, create a YAML file with your server arguments and specify it with `--config`. CLI arguments will override config file values.

  ```bash
  # Create config.yaml
  cat > config.yaml << EOF
  model-path: meta-llama/Meta-Llama-3-8B-Instruct
  host: 0.0.0.0
  port: 30000
  tensor-parallel-size: 2
  enable-metrics: true
  log-requests: true
  EOF

  # Launch server with config file
  python -m sglang.launch_server --config config.yaml
  ```

- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total. We recommend [SGLang Router](../advanced_features/router.md) for data parallelism.

  ```bash
  python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
  ```

- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```

- See [hyperparameter tuning](hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- For docker and Kubernetes runs, you need to set up shared memory which is used for communication between processes. See `--shm-size` for docker and `/dev/shm` size update for Kubernetes manifests.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
  ```

- To enable `torch.compile` acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. By default, the cache path is located at `/tmp/torchinductor_root`, you can customize it using environment variable `TORCHINDUCTOR_CACHE_DIR`. For more details, please refer to [PyTorch official documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) and [Enabling cache for torch.compile](https://docs.sglang.io/references/torch_compile_cache.html).
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other [quantization strategies (INT8/FP8)](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L671) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- To enable deterministic inference and batch invariant operations, add `--enable-deterministic-inference`. More details can be found in [deterministic inference document](../advanced_features/deterministic_inference.md).
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](../references/custom_chat_template.md). If the tokenizer has multiple named templates (e.g., 'default', 'tool_use'), you can select one using `--hf-chat-template-name tool_use`.
- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`

  ```bash
  # Node 0
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 0

  # Node 1
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 1
  ```

Please consult the documentation below and [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) to learn more about the arguments you may provide when launching a server.

## Model and tokenizer
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--model-path`<br>`--model` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. | `None` | Type: str |
| `--tokenizer-path` | The path of the tokenizer. | `None` | Type: str |
| `--tokenizer-mode` | Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer. | `auto` | `auto`, `slow` |
| `--tokenizer-worker-num` | The worker num of the tokenizer manager. | `1` | Type: int |
| `--skip-tokenizer-init` | If set, skip init tokenizer and pass input_ids in generate request. | `False` | bool flag (set to enable) |
| `--load-format` | The format of the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly for profiling."gguf" will load the weights in the gguf format. "bitsandbytes" will load the weights using bitsandbytes quantization."layered" loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller. "flash_rl" will load the weights in flash_rl format. "fastsafetensors" and "private" are also supported. | `auto` | `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `sharded_state`, `gguf`, `bitsandbytes`, `layered`, `flash_rl`, `remote`, `remote_instance`, `fastsafetensors`, `private` |
| `--model-loader-extra-config` | Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. | `{}` | Type: str |
| `--trust-remote-code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. | `False` | bool flag (set to enable) |
| `--context-length` | The model's maximum context length. Defaults to None (will use the value from the model's config.json instead). | `None` | Type: int |
| `--is-embedding` | Whether to use a CausalLM as an embedding model. | `False` | bool flag (set to enable) |
| `--enable-multimodal` | Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen | `None` | bool flag (set to enable) |
| `--revision` | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | `None` | Type: str |
| `--model-impl` | Which implementation of the model to use. * "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. * "sglang" will use the SGLang model implementation. * "transformers" will use the Transformers model implementation. | `auto` | Type: str |

## HTTP server
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--host` | The host of the HTTP server. | `127.0.0.1` | Type: str |
| `--port` | The port of the HTTP server. | `30000` | Type: int |
| `--fastapi-root-path` | App is behind a path based routing proxy. | `""` | Type: str |
| `--grpc-mode` | If set, use gRPC server instead of HTTP server. | `False` | bool flag (set to enable) |
| `--skip-server-warmup` | If set, skip warmup. | `False` | bool flag (set to enable) |
| `--warmups` | Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests | `None` | Type: str |
| `--nccl-port` | The port for NCCL distributed environment setup. Defaults to a random port. | `None` | Type: int |
| `--checkpoint-engine-wait-weights-before-ready` | If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests. | `False` | bool flag (set to enable) |

## Quantization and data type
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dtype` | Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision. | `auto` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32` |
| `--quantization` | The quantization method. | `None` | `awq`, `fp8`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `petit_nvfp4`, `w8a8_int8`, `w8a8_fp8`, `moe_wna16`, `qoq`, `w4afp8`, `mxfp4`, `auto-round`, `compressed-tensors`, `modelslim`, `quark_int4fp8_moe` |
| `--quantization-param-path` | Path to the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues. | `None` | Type: Optional[str] |
| `--kv-cache-dtype` | Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+ | `auto` | `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `fp4_e2m1` |
| `--enable-fp32-lm-head` | If set, the LM head outputs (logits) are in FP32. | `False` | bool flag (set to enable) |
| `--modelopt-quant` | The ModelOpt quantization configuration. Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt | `None` | Type: str |
| `--modelopt-checkpoint-restore-path` | Path to restore a previously saved ModelOpt quantized checkpoint. If provided, the quantization process will be skipped and the model will be loaded from this checkpoint. | `None` | Type: str |
| `--modelopt-checkpoint-save-path` | Path to save the ModelOpt quantized checkpoint after quantization. This allows reusing the quantized model in future runs. | `None` | Type: str |
| `--modelopt-export-path` | Path to export the quantized model in HuggingFace format after ModelOpt quantization. The exported model can then be used directly with SGLang for inference. If not provided, the model will not be exported. | `None` | Type: str |
| `--quantize-and-serve` | Quantize the model with ModelOpt and immediately serve it without exporting. This is useful for development and prototyping. For production, it's recommended to use separate quantization and deployment steps. | `False` | bool flag (set to enable) |
| `--rl-quant-profile` | Path to the FlashRL quantization profile. Required when using --load-format flash_rl. | `None` | Type: str |

## Memory and scheduling
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--mem-fraction-static` | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. | `None` | Type: float |
| `--max-running-requests` | The maximum number of running requests. | `None` | Type: int |
| `--max-queued-requests` | The maximum number of queued requests. This option is ignored when using disaggregation-mode. | `None` | Type: int |
| `--max-total-tokens` | The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes. | `None` | Type: int |
| `--chunked-prefill-size` | The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill. | `None` | Type: int |
| `--prefill-max-requests` | The maximum number of requests in a prefill batch. If not specified, there is no limit. | `None` | Type: int |
| `--enable-dynamic-chunking` | Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks. | `False` | bool flag (set to enable) |
| `--max-prefill-tokens` | The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. | `16384` | Type: int |
| `--schedule-policy` | The scheduling policy of the requests. | `fcfs` | `lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority`, `routing-key` |
| `--enable-priority-scheduling` | Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default. | `False` | bool flag (set to enable) |
| `--abort-on-priority-when-disabled` | If set, abort requests that specify a priority when priority scheduling is disabled. | `False` | bool flag (set to enable) |
| `--schedule-low-priority-values-first` | If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first. | `False` | bool flag (set to enable) |
| `--priority-scheduling-preemption-threshold` | Minimum difference in priorities for an incoming request to have to preempt running request(s). | `10` | Type: int |
| `--schedule-conservativeness` | How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. | `1.0` | Type: float |
| `--page-size` | The number of tokens in a page. | `1` | Type: int |
| `--swa-full-tokens-ratio` | The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens. | `0.8` | Type: float |
| `--disable-hybrid-swa-memory` | Disable the hybrid SWA memory. | `False` | bool flag (set to enable) |
| `--radix-eviction-policy` | The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used. | `lru` | `lru`, `lfu` |
| `--enable-prefill-delayer` | Enable prefill delayer for DP attention to reduce idle time. | `False` | bool flag (set to enable) |
| `--prefill-delayer-max-delay-passes` | Maximum forward passes to delay prefill. | `30` | Type: int |
| `--prefill-delayer-token-usage-low-watermark` | Token usage low watermark for prefill delayer. | `None` | Type: float |
| `--prefill-delayer-forward-passes-buckets` | Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added. | `None` | List[float] |
| `--prefill-delayer-wait-seconds-buckets` | Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added. | `None` | List[float] |

## Runtime options
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--device` | The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified. | `None` | Type: str |
| `--tensor-parallel-size`<br>`--tp-size` | The tensor parallelism size. | `1` | Type: int |
| `--pipeline-parallel-size`<br>`--pp-size` | The pipeline parallelism size. | `1` | Type: int |
| `--pp-max-micro-batch-size` | The maximum micro batch size in pipeline parallelism. | `None` | Type: int |
| `--pp-async-batch-depth` | The async batch depth of pipeline parallelism. | `0` | Type: int |
| `--stream-interval` | The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher | `1` | Type: int |
| `--stream-output` | Whether to output as a sequence of disjoint segments. | `False` | bool flag (set to enable) |
| `--random-seed` | The random seed. | `None` | Type: int |
| `--constrained-json-whitespace-pattern` | (outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model to generate consecutive whitespaces, set the pattern to [\n\t ]* | `None` | Type: str |
| `--constrained-json-disable-any-whitespace` | (xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output. | `False` | bool flag (set to enable) |
| `--watchdog-timeout` | Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging. | `300` | Type: float |
| `--soft-watchdog-timeout` | Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging. | `None` | Type: float |
| `--dist-timeout` | Set timeout for torch.distributed initialization. | `None` | Type: int |
| `--download-dir` | Model download directory for huggingface. | `None` | Type: str |
| `--model-checksum` | Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID. | `None` | Type: str |
| `--base-gpu-id` | The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine. | `0` | Type: int |
| `--gpu-id-step` | The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,... | `1` | Type: int |
| `--sleep-on-idle` | Reduce CPU usage when sglang is idle. | `False` | bool flag (set to enable) |
| `--custom-sigquit-handler` | Register a custom sigquit handler so you can do additional cleanup after the server is shutdown. This is only available for Engine, not for CLI. | `None` | Type: str |

## Logging
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--log-level` | The logging level of all loggers. | `info` | Type: str |
| `--log-level-http` | The logging level of HTTP server. If not set, reuse --log-level by default. | `None` | Type: str |
| `--log-requests` | Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level | `False` | bool flag (set to enable) |
| `--log-requests-level` | 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output. | `2` | `0`, `1`, `2`, `3` |
| `--log-requests-format` | Format for request logging: 'text' (human-readable) or 'json' (structured) | `text` | `text`, `json` |
| `--log-requests-target` | Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. | `None` | List[str] |
| `--uvicorn-access-log-exclude-prefixes` | Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). | `[]` | List[str] |
| `--crash-dump-folder` | Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled. | `None` | Type: str |
| `--show-time-cost` | Show time cost of custom marks. | `False` | bool flag (set to enable) |
| `--enable-metrics` | Enable log prometheus metrics. | `False` | bool flag (set to enable) |
| `--enable-metrics-for-all-schedulers` | Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0. | `False` | bool flag (set to enable) |
| `--tokenizer-metrics-custom-labels-header` | Specify the HTTP header for passing custom labels for tokenizer metrics. | `x-custom-labels` | Type: str |
| `--tokenizer-metrics-allowed-custom-labels` | The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set. | `None` | List[str] |
| `--bucket-time-to-first-token` | The buckets of time to first token, specified as a list of floats. | `None` | List[float] |
| `--bucket-inter-token-latency` | The buckets of inter-token latency, specified as a list of floats. | `None` | List[float] |
| `--bucket-e2e-request-latency` | The buckets of end-to-end request latency, specified as a list of floats. | `None` | List[float] |
| `--collect-tokens-histogram` | Collect prompt/generation tokens histogram. | `False` | bool flag (set to enable) |
| `--prompt-tokens-buckets` | The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). | `None` | List[str] |
| `--generation-tokens-buckets` | The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). | `None` | List[str] |
| `--gc-warning-threshold-secs` | The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable. | `0.0` | Type: float |
| `--decode-log-interval` | The log interval of decode batch. | `40` | Type: int |
| `--enable-request-time-stats-logging` | Enable per request time stats logging | `False` | bool flag (set to enable) |
| `--kv-events-config` | Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. | `None` | Type: str |
| `--enable-trace` | Enable opentelemetry trace | `False` | bool flag (set to enable) |
| `--otlp-traces-endpoint` | Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port> | `localhost:4317` | Type: str |

## RequestMetricsExporter configuration
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--export-metrics-to-file` | Export performance metrics for each request to local file (e.g. for forwarding to external systems). | `False` | bool flag (set to enable) |
| `--export-metrics-to-file-dir` | Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled). | `None` | Type: str |

## API related
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--api-key` | Set API key of the server. It is also used in the OpenAI API compatible server. | `None` | Type: str |
| `--admin-api-key` | Set **admin API key** for administrative/control endpoints (e.g., weights update, cache flush, `/get_server_info`). Endpoints marked as admin-only require `Authorization: Bearer <admin_api_key>` when this is set. | `None` | Type: str |
| `--served-model-name` | Override the model name returned by the v1/models endpoint in OpenAI API server. | `None` | Type: str |
| `--weight-version` | Version identifier for the model weights. Defaults to 'default' if not specified. | `default` | Type: str |
| `--chat-template` | The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. | `None` | Type: str |
| `--hf-chat-template-name` | When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used. | `None` | Type: str |
| `--completion-template` | The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently. | `None` | Type: str |
| `--file-storage-path` | The path of the file storage in backend. | `sglang_storage` | Type: str |
| `--enable-cache-report` | Return number of cached tokens in usage.prompt_tokens_details for each openai request. | `False` | bool flag (set to enable) |
| `--reasoning-parser` | Specify the parser for reasoning models. Supported parsers: [deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3]. | `None` | `deepseek-r1`, `deepseek-v3`, `glm45`, `gpt-oss`, `kimi`, `qwen3`, `qwen3-thinking`, `step3` |
| `--tool-call-parser` | Specify the parser for handling tool-call interactions. Supported parsers: [deepseekv3, deepseekv31, glm, glm45, glm47, gpt-oss, kimi_k2, llama3, mistral, pythonic, qwen, qwen25, qwen3_coder, step3]. | `None` | `deepseekv3`, `deepseekv31`, `glm`, `glm45`, `glm47`, `gpt-oss`, `kimi_k2`, `llama3`, `mistral`, `pythonic`, `qwen`, `qwen25`, `qwen3_coder`, `step3` |
| `--tool-server` | Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used. | `None` | Type: str |
| `--sampling-defaults` | Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'. | `model` | `openai`, `model` |

## Data parallelism
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--data-parallel-size`<br>`--dp-size` | The data parallelism size. | `1` | Type: int |
| `--load-balance-method` | The load balancing strategy for data parallelism. The `total_tokens` algorithm can only be used when DP attention is applied. This algorithm performs load balancing based on the real-time token load of the DP workers. | `auto` | `auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens` |

## Multi-node distributed serving
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dist-init-addr`<br>`--nccl-init-addr` | The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). | `None` | Type: str |
| `--nnodes` | The number of nodes. | `1` | Type: int |
| `--node-rank` | The node rank. | `0` | Type: int |

## Model override args
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--json-model-override-args` | A dictionary in JSON string format used to override default model configurations. | `{}` | Type: str |
| `--preferred-sampling-params` | json-formatted sampling settings that will be returned in /get_model_info | `None` | Type: str |

## LoRA
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-lora` | Enable LoRA support for the model. This argument is automatically set to `True` if `--lora-paths` is provided for backward compatibility. | `False` | Bool flag (set to enable) |
| `--enable-lora-overlap-loading` | Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters. | `False` | Bool flag (set to enable)
| `--max-lora-rank` | The maximum LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup. | `None` | Type: int |
| `--lora-target-modules` | The union set of all target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. You can also set it to `all` to enable LoRA for all supported modules; note this may introduce minor performance overhead. | `None` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qkv_proj`, `gate_up_proj`, `all` |
| `--lora-paths` | The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: `<PATH>` \| `<NAME>=<PATH>` \| JSON with schema `{"lora_name": str, "lora_path": str, "pinned": bool}`. | `None` | Type: List[str] / JSON objects |
| `--max-loras-per-batch` | Maximum number of adapters for a running batch, including base-only requests. | `8` | Type: int |
| `--max-loaded-loras` | If specified, limits the maximum number of LoRA adapters loaded in CPU memory at a time. Must be ≥ `--max-loras-per-batch`. | `None` | Type: int |
| `--lora-eviction-policy` | LoRA adapter eviction policy when the GPU memory pool is full. | `lru` | `lru`, `fifo` |
| `--lora-backend` | Choose the kernel backend for multi-LoRA serving. | `csgmv` | `triton`, `csgmv`, `ascend`, `torch_native` |
| `--max-lora-chunk-size` | Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when `--lora-backend` is `csgmv`. Larger values may improve performance. | `16` | `16`, `32`, `64`, `128` |

## Kernel Backends (Attention, Sampling, Grammar, GEMM)
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--attention-backend` | Choose the kernels for attention layers. | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--prefill-attention-backend` | Choose the kernels for prefill attention layers (have priority over --attention-backend). | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--decode-attention-backend` | Choose the kernels for decode attention layers (have priority over --attention-backend). | `None` | `triton`, `torch_native`, `flex_attention`, `nsa`, `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--sampling-backend` | Choose the kernels for sampling layers. | `None` | `flashinfer`, `pytorch`, `ascend` |
| `--grammar-backend` | Choose the backend for grammar-guided decoding. | `None` | `xgrammar`, `outlines`, `llguidance`, `none` |
| `--mm-attention-backend` | Set multimodal attention backend. | `None` | `sdpa`, `fa3`, `triton_attn`, `ascend_attn`, `aiter_attn` |
| `--nsa-prefill-backend` | Choose the NSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek NSA-style attention). | `flashmla_sparse` | `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter` |
| `--nsa-decode-backend` | Choose the NSA backend for the decode stage when running DeepSeek NSA-style attention. Overrides `--attention-backend` for decoding. | `fa3` | `flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`, `aiter` |
| `--fp8-gemm-backend` | Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). **NOTE**: This replaces the deprecated environment variables SGLANG_ENABLE_FLASHINFER_FP8_GEMM and SGLANG_SUPPORT_CUTLASS_BLOCK_FP8. | `auto` | `auto`, `deep_gemm`, `flashinfer_trtllm`, `cutlass`, `triton`, `aiter` |
| `--fp4-gemm-backend` | Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default, auto-selects between flashinfer_cudnn/flashinfer_cutlass based on CUDA/cuDNN version), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutlass' (FlashInfer CUTLASS backend, optimal on CUDA 12), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling). All backends are from FlashInfer; when FlashInfer is unavailable, sgl-kernel CUTLASS is used as an automatic fallback. **NOTE**: This replaces the deprecated environment variable SGLANG_FLASHINFER_FP4_GEMM_BACKEND. | `auto` | `auto`, `flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm` |
| `--disable-flashinfer-autotune` | Flashinfer autotune is enabled by default. Set this flag to disable the autotune. | `False` | bool flag (set to enable) |

## Speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--speculative-algorithm` | Speculative algorithm. | `None` | `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM` |
| `--speculative-draft-model-path`<br>`--speculative-draft-model` | The path of the draft model weights. This can be a local folder or a Hugging Face repo ID. | `None` | Type: str |
| `--speculative-draft-model-revision` | The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | `None` | Type: str |
| `--speculative-draft-load-format` | The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling. | `None` | Same as --load-format options |
| `--speculative-num-steps` | The number of steps sampled from draft model in Speculative Decoding. | `None` | Type: int |
| `--speculative-eagle-topk` | The number of tokens sampled from the draft model in eagle2 each step. | `None` | Type: int |
| `--speculative-num-draft-tokens` | The number of tokens sampled from the draft model in Speculative Decoding. | `None` | Type: int |
| `--speculative-accept-threshold-single` | Accept a draft token if its probability in the target model is greater than this threshold. | `1.0` | Type: float |
| `--speculative-accept-threshold-acc` | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). | `1.0` | Type: float |
| `--speculative-token-map` | The path of the draft model's small vocab table. | `None` | Type: str |
| `--speculative-attention-mode` | Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'. | `prefill` | `prefill`, `decode` |
| `--speculative-draft-attention-backend` | Attention backend for speculative decoding drafting. | `None` | Same as attention backend options |
| `--speculative-moe-runner-backend` | MOE backend for EAGLE speculative decoding, see --moe-runner-backend for options. Same as moe runner backend if unset. | `None` | Same as --moe-runner-backend options |
| `--speculative-moe-a2a-backend` | MOE A2A backend for EAGLE speculative decoding, see --moe-a2a-backend for options. Same as moe a2a backend if unset. | `None` | Same as --moe-a2a-backend options |
| `--speculative-draft-model-quantization` | The quantization method for speculative model. | `None` | Same as --quantization options |

## Ngram speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--speculative-ngram-min-match-window-size` | The minimum window size for pattern matching in ngram speculative decoding. | `1` | Type: int |
| `--speculative-ngram-max-match-window-size` | The maximum window size for pattern matching in ngram speculative decoding. | `12` | Type: int |
| `--speculative-ngram-min-bfs-breadth` | The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding. | `1` | Type: int |
| `--speculative-ngram-max-bfs-breadth` | The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding. | `10` | Type: int |
| `--speculative-ngram-match-type` | The match type for cache tree. | `BFS` | `BFS`, `PROB` |
| `--speculative-ngram-branch-length` | The branch length for ngram speculative decoding. | `18` | Type: int |
| `--speculative-ngram-capacity` | The cache capacity for ngram speculative decoding. | `10000000` | Type: int |

## Multi-layer Eagle speculative decoding
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-multi-layer-eagle` | Enable multi-layer Eagle speculative decoding. | `False` | bool flag (set to enable) |

## MoE
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--expert-parallel-size`<br>`--ep-size`<br>`--ep` | The expert parallelism size. | `1` | Type: int |
| `--moe-a2a-backend` | Select the backend for all-to-all communication for expert parallelism. | `none` | `none`, `deepep`, `mooncake`, `ascend_fuseep`|
| `--moe-runner-backend` | Choose the runner backend for MoE. | `auto` | `auto`, `deep_gemm`, `triton`, `triton_kernel`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_mxfp4`, `flashinfer_cutedsl`, `cutlass` |
| `--flashinfer-mxfp4-moe-precision` | Choose the computation precision of flashinfer mxfp4 moe | `default` | `default`, `bf16` |
| `--enable-flashinfer-allreduce-fusion` | Enable FlashInfer allreduce fusion with Residual RMSNorm. | `False` | bool flag (set to enable) |
| `--deepep-mode` | Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch. | `auto` | `normal`, `low_latency`, `auto` |
| `--ep-num-redundant-experts` | Allocate this number of redundant experts in expert parallel. | `0` | Type: int |
| `--ep-dispatch-algorithm` | The algorithm to choose ranks for redundant experts in expert parallel. | `None` | Type: str |
| `--init-expert-location` | Initial location of EP experts. | `trivial` | Type: str |
| `--enable-eplb` | Enable EPLB algorithm | `False` | bool flag (set to enable) |
| `--eplb-algorithm` | Chosen EPLB algorithm | `auto` | Type: str |
| `--eplb-rebalance-num-iterations` | Number of iterations to automatically trigger a EPLB re-balance. | `1000` | Type: int |
| `--eplb-rebalance-layers-per-chunk` | Number of layers to rebalance per forward pass. | `None` | Type: int |
| `--eplb-min-rebalancing-utilization-threshold` | Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0]. | `1.0` | Type: float |
| `--expert-distribution-recorder-mode` | Mode of expert distribution recorder. | `None` | Type: str |
| `--expert-distribution-recorder-buffer-size` | Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer. | `None` | Type: int |
| `--enable-expert-distribution-metrics` | Enable logging metrics for expert balancedness | `False` | bool flag (set to enable) |
| `--deepep-config` | Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path. | `None` | Type: str |
| `--moe-dense-tp-size` | TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports. | `None` | Type: int |
| `--elastic-ep-backend` | Specify the collective communication backend for elastic EP. Currently supports 'mooncake'. | `none` | `none`, `mooncake` |
| `--mooncake-ib-device` | The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled. | `None` | Type: str |

## Mamba Cache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--max-mamba-cache-size` | The maximum size of the mamba cache. | `None` | Type: int |
| `--mamba-ssm-dtype` | The data type of the SSM states in mamba cache. | `float32` | `float32`, `bfloat16` |
| `--mamba-full-memory-ratio` | The ratio of mamba state memory to full kv cache memory. | `0.9` | Type: float |
| `--mamba-scheduler-strategy` | The strategy to use for mamba scheduler. `auto` currently defaults to `no_buffer`. 1. `no_buffer` does not support overlap scheduler due to not allocating extra mamba state buffers. Branching point caching support is feasible but not implemented. 2. `extra_buffer` supports overlap schedule by allocating extra mamba state buffers to track mamba state for caching (mamba state usage per running req becomes `2x` for non-spec; `1+(1/(2+speculative_num_draft_tokens))x` for spec dec (e.g. 1.16x if speculative_num_draft_tokens==4)). 2a. `extra_buffer` is strictly better for non-KV-cache-bound cases; for KV-cache-bound cases, the tradeoff depends on whether enabling overlap outweighs reduced max running requests. 2b. mamba caching at radix cache branching point is strictly better than non-branch but requires kernel support (currently only FLA backend), currently only extra_buffer supports branching. | `auto` | `auto`, `no_buffer`, `extra_buffer` |
| `--mamba-track-interval` | The interval (in tokens) to track the mamba state during decode. Only used when `--mamba-scheduler-strategy` is `extra_buffer`. Must be divisible by page_size if set, and must be >= speculative_num_draft_tokens when using speculative decoding. | `256` | Type: int |

## Hierarchical cache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-hierarchical-cache` | Enable hierarchical cache | `False` | bool flag (set to enable) |
| `--hicache-ratio` | The ratio of the size of host KV cache memory pool to the size of device pool. | `2.0` | Type: float |
| `--hicache-size` | The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set. | `0` | Type: int |
| `--hicache-write-policy` | The write policy of hierarchical cache. | `write_through` | `write_back`, `write_through`, `write_through_selective` |
| `--hicache-io-backend` | The IO backend for KV cache transfer between CPU and GPU | `kernel` | `direct`, `kernel`, `kernel_ascend` |
| `--hicache-mem-layout` | The layout of host memory pool for hierarchical cache. | `layer_first` | `layer_first`, `page_first`, `page_first_direct`, `page_first_kv_split`, `page_head` |
| `--hicache-storage-backend` | The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name). | `None` | `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic` |
| `--hicache-storage-prefetch-policy` | Control when prefetching from the storage backend should stop. | `best_effort` | `best_effort`, `wait_complete`, `timeout` |
| `--hicache-storage-backend-extra-config` | A dictionary in JSON string format containing extra configuration for the storage backend. | `None` | Type: str |

## Hierarchical sparse attention
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--hierarchical-sparse-attention-extra-config` | A dictionary in JSON string format for hierarchical sparse attention configuration. Required fields: `algorithm` (str), `backend` (str). All other fields are algorithm-specific and passed to the algorithm constructor. | `None` | Type: str |

## LMCache
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-lmcache` | Using LMCache as an alternative hierarchical cache solution | `False` | bool flag (set to enable) |

## Ktransformers
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--kt-weight-path` | [ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder. | `None` | Type: str |
| `--kt-method` | [ktransformers parameter] Quantization formats for CPU execution. | `AMXINT4` | Type: str |
| `--kt-cpuinfer` | [ktransformers parameter] The number of CPUInfer threads. | `None` | Type: int |
| `--kt-threadpool-count` | [ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA). | `2` | Type: int |
| `--kt-num-gpu-experts` | [ktransformers parameter] The number of GPU experts. | `None` | Type: int |
| `--kt-max-deferred-experts-per-token` | [ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0. | `None` | Type: int |

## Diffusion LLM
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--dllm-algorithm` | The diffusion LLM algorithm, such as LowConfidence. | `None` | Type: str |
| `--dllm-algorithm-config` | The diffusion LLM algorithm configurations. Must be a YAML file. | `None` | Type: str |

## Double Sparsity
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-double-sparsity` | Enable double sparsity attention | `False` | bool flag (set to enable) |
| `--ds-channel-config-path` | The path of the double sparsity channel config | `None` | Type: str |
| `--ds-heavy-channel-num` | The number of heavy channels in double sparsity attention | `32` | Type: int |
| `--ds-heavy-token-num` | The number of heavy tokens in double sparsity attention | `256` | Type: int |
| `--ds-heavy-channel-type` | The type of heavy channels in double sparsity attention | `qk` | Type: str |
| `--ds-sparse-decode-threshold` | The minimum decode sequence length required before the double-sparsity backend switches from the dense fallback to the sparse decode kernel. | `4096` | Type: int |

## Offloading
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--cpu-offload-gb` | How many GBs of RAM to reserve for CPU offloading. | `0` | Type: int |
| `--offload-group-size` | Number of layers per group in offloading. | `-1` | Type: int |
| `--offload-num-in-group` | Number of layers to be offloaded within a group. | `1` | Type: int |
| `--offload-prefetch-step` | Steps to prefetch in offloading. | `1` | Type: int |
| `--offload-mode` | Mode of offloading. | `cpu` | Type: str |

## Args for multi-item scoring
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--multi-item-scoring-delimiter` | Delimiter token ID for multi-item scoring. Used to combine Query and Items into a single sequence: Query<delimiter>Item1<delimiter>Item2<delimiter>... This enables efficient batch processing of multiple items against a single query. | `None` | Type: int |

## Optimization/debug options
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--disable-radix-cache` | Disable RadixAttention for prefix caching. | `False` | bool flag (set to enable) |
| `--cuda-graph-max-bs` | Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value. | `None` | Type: int |
| `--cuda-graph-bs` | Set the list of batch sizes for cuda graph. | `None` | List[int] |
| `--disable-cuda-graph` | Disable cuda graph. | `False` | bool flag (set to enable) |
| `--disable-cuda-graph-padding` | Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed. | `False` | bool flag (set to enable) |
| `--enable-profile-cuda-graph` | Enable profiling of cuda graph capture. | `False` | bool flag (set to enable) |
| `--enable-cudagraph-gc` | Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process. | `False` | bool flag (set to enable) |
| `--enable-layerwise-nvtx-marker` | Enable layerwise NVTX profiling annotations for the model. This adds NVTX markers to every layer for detailed per-layer performance analysis with Nsight Systems. | `False` | bool flag (set to enable) |
| `--enable-nccl-nvls` | Enable NCCL NVLS for prefill heavy requests when available. | `False` | bool flag (set to enable) |
| `--enable-symm-mem` | Enable NCCL symmetric memory for fast collectives. | `False` | bool flag (set to enable) |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | Disables quantize before all-gather for flashinfer cutlass moe. | `False` | bool flag (set to enable) |
| `--enable-tokenizer-batch-encode` | Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds. | `False` | bool flag (set to enable) |
| `--disable-tokenizer-batch-decode` | Disable batch decoding when decoding multiple completions. | `False` | bool flag (set to enable) |
| `--disable-outlines-disk-cache` | Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency. | `False` | bool flag (set to enable) |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel and fall back to NCCL. | `False` | bool flag (set to enable) |
| `--enable-mscclpp` | Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL. | `False` | bool flag (set to enable) |
| `--enable-torch-symm-mem` | Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM10 supports world size 6, 8. | `False` | bool flag (set to enable) |
| `--disable-overlap-schedule` | Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. | `False` | bool flag (set to enable) |
| `--enable-mixed-chunk` | Enabling mixing prefill and decode in a batch when using chunked prefill. | `False` | bool flag (set to enable) |
| `--enable-dp-attention` | Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported. | `False` | bool flag (set to enable) |
| `--enable-dp-lm-head` | Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. | `False` | bool flag (set to enable) |
| `--enable-two-batch-overlap` | Enabling two micro batches to overlap. | `False` | bool flag (set to enable) |
| `--enable-single-batch-overlap` | Let computation and communication overlap within one micro batch. | `False` | bool flag (set to enable) |
| `--tbo-token-distribution-threshold` | The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap. | `0.48` | Type: float |
| `--enable-torch-compile` | Optimize the model with torch.compile. Experimental feature. | `False` | bool flag (set to enable) |
| `--enable-torch-compile-debug-mode` | Enable debug mode for torch compile. | `False` | bool flag (set to enable) |
| `--enable-piecewise-cuda-graph` | Optimize the model with piecewise cuda graph for extend/prefill only. Experimental feature. | `False` | bool flag (set to enable) |
| `--piecewise-cuda-graph-tokens` | Set the list of tokens when using piecewise cuda graph. | `None` | Type: JSON list |
| `--piecewise-cuda-graph-compiler` | Set the compiler for piecewise cuda graph. Choices are: eager, inductor. | `eager` | `eager`, `inductor` |
| `--torch-compile-max-bs` | Set the maximum batch size when using torch compile. | `32` | Type: int |
| `--piecewise-cuda-graph-max-tokens` | Set the maximum tokens when using piecewise cuda graph. | `4096` | Type: int |
| `--torchao-config` | Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row | `` | Type: str |
| `--enable-nan-detection` | Enable the NaN detection for debugging purposes. | `False` | bool flag (set to enable) |
| `--enable-p2p-check` | Enable P2P check for GPU access, otherwise the p2p access is allowed by default. | `False` | bool flag (set to enable) |
| `--triton-attention-reduce-in-fp32` | Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16. This only affects Triton attention kernels. | `False` | bool flag (set to enable) |
| `--triton-attention-num-kv-splits` | The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8. | `8` | Type: int |
| `--triton-attention-split-tile-size` | The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference. | `None` | Type: int |
| `--num-continuous-decode-steps` | Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time. | `1` | Type: int |
| `--delete-ckpt-after-loading` | Delete the model checkpoint after loading the model. | `False` | bool flag (set to enable) |
| `--enable-memory-saver` | Allow saving memory using release_memory_occupation and resume_memory_occupation | `False` | bool flag (set to enable) |
| `--enable-weights-cpu-backup` | Save model weights to CPU memory during release_weights_occupation and resume_weights_occupation | `False` | bool flag (set to enable) |
| `--enable-draft-weights-cpu-backup` | Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation | `False` | bool flag (set to enable) |
| `--allow-auto-truncate` | Allow automatically truncating requests that exceed the maximum input length instead of returning an error. | `False` | bool flag (set to enable) |
| `--enable-custom-logit-processor` | Enable users to pass custom logit processors to the server (disabled by default for security) | `False` | bool flag (set to enable) |
| `--flashinfer-mla-disable-ragged` | Not using ragged prefill wrapper when running flashinfer mla | `False` | bool flag (set to enable) |
| `--disable-shared-experts-fusion` | Disable shared experts fusion optimization for deepseek v3/r1. | `False` | bool flag (set to enable) |
| `--disable-chunked-prefix-cache` | Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences. | `False` | bool flag (set to enable) |
| `--disable-fast-image-processor` | Adopt base image processor instead of fast image processor. | `False` | bool flag (set to enable) |
| `--keep-mm-feature-on-device` | Keep multimodal feature tensors on device after processing to save D2H copy. | `False` | bool flag (set to enable) |
| `--enable-return-hidden-states` | Enable returning hidden states with responses. | `False` | bool flag (set to enable) |
| `--enable-return-routed-experts` | Enable returning routed experts of each layer with responses. | `False` | bool flag (set to enable) |
| `--scheduler-recv-interval` | The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this. | `1` | Type: int |
| `--numa-node` | Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. | `None` | List[int] |
| `--enable-deterministic-inference` | Enable deterministic inference mode with batch invariant ops. | `False` | bool flag (set to enable) |
| `--rl-on-policy-target` | The training system that SGLang needs to match for true on-policy. | `None` | `fsdp` |
| `--enable-attn-tp-input-scattered` | Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent. | `False` | bool flag (set to enable) |
| `--enable-nsa-prefill-context-parallel` | Enable context parallelism used in the long sequence prefill phase of DeepSeek v3.2. | `False` | bool flag (set to enable) |
| `--nsa-prefill-cp-mode` | Token splitting mode for the prefill phase of DeepSeek v3.2 under context parallelism. Optional values: `in-seq-split` (default), `round-robin-split`. `round-robin-split` distributes tokens across ranks based on `token_idx % cp_size`. It supports multi-batch prefill, fused MoE, and FP8 KV cache. | `in-seq-split` | `in-seq-split`, `round-robin-split` |
| `--enable-fused-qk-norm-rope` | Enable fused qk normalization and rope rotary embedding. | `False` | bool flag (set to enable) |
| `--enable-precise-embedding-interpolation` | Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values. | `False` | bool flag (set to enable) |

## Dynamic batch tokenizer
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-dynamic-batch-tokenizer` | Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently. | `False` | bool flag (set to enable) |
| `--dynamic-batch-tokenizer-batch-size` | [Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer. | `32` | Type: int |
| `--dynamic-batch-tokenizer-batch-timeout` | [Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests. | `0.002` | Type: float |

## Debug tensor dumps
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--debug-tensor-dump-output-folder` | The output folder for dumping tensors. | `None` | Type: str |
| `--debug-tensor-dump-layers` | The layer ids to dump. Dump all layers if not specified. | `None` | Type: JSON list |
| `--debug-tensor-dump-input-file` | The input filename for dumping tensors | `None` | Type: str |
| `--debug-tensor-dump-inject` | Inject the outputs from jax as the input of every layer. | `False` | Type: str |

## PD disaggregation
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--disaggregation-mode` | Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated | `null` | `null`, `prefill`, `decode` |
| `--disaggregation-transfer-backend` | The backend for disaggregation transfer. Default is mooncake. | `mooncake` | `mooncake`, `nixl`, `ascend`, `fake` |
| `--disaggregation-bootstrap-port` | Bootstrap server port on the prefill server. Default is 8998. | `8998` | Type: int |
| `--disaggregation-decode-tp` | Decode tp size. If not set, it matches the tp size of the current engine. This is only set on the prefill server. | `None` | Type: int |
| `--disaggregation-decode-dp` | Decode dp size. If not set, it matches the dp size of the current engine. This is only set on the prefill server. | `None` | Type: int |
| `--disaggregation-prefill-pp` | Prefill pp size. If not set, it is default to 1. This is only set on the decode server. | `1` | Type: int |
| `--disaggregation-ib-device` | The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when mooncake backend is enabled. | `None` | Type: str |
| `--disaggregation-decode-enable-offload-kvcache` | Enable async KV cache offloading on decode server (PD mode). | `False` | bool flag (set to enable) |
| `--disaggregation-decode-enable-fake-auto` | Auto enable FAKE mode for decode node testing, no need to pass bootstrap_host and bootstrap_room in request. | `False` | bool flag (set to enable) |
| `--num-reserved-decode-tokens` | Number of decode tokens that will have memory reserved when adding new request to the running batch. | `512` | Type: int |
| `--disaggregation-decode-polling-interval` | The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this. | `1` | Type: int |

## Encode prefill disaggregation
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--encoder-only` | For MLLM with an encoder, launch an encoder-only server | `False` | bool flag (set to enable) |
| `--language-only` | For VLM, load weights for the language model only. | `False` | bool flag (set to enable) |
| `--encoder-transfer-backend` | The backend for encoder disaggregation transfer. Default is zmq_to_scheduler. | `zmq_to_scheduler` | `zmq_to_scheduler`, `zmq_to_tokenizer`, `mooncake` |
| `--encoder-urls` | List of encoder server urls. | `[]` | Type: JSON list |

## Custom weight loader
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--custom-weight-loader` | The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func | `None` | List[str] |
| `--weight-loader-disable-mmap` | Disable mmap while loading weight using safetensors. | `False` | bool flag (set to enable) |
| `--remote-instance-weight-loader-seed-instance-ip` | The ip of the seed instance for loading weights from remote instance. | `None` | Type: str |
| `--remote-instance-weight-loader-seed-instance-service-port` | The service port of the seed instance for loading weights from remote instance. | `None` | Type: int |
| `--remote-instance-weight-loader-send-weights-group-ports` | The communication group ports for loading weights from remote instance. | `None` | Type: JSON list |
| `--remote-instance-weight-loader-backend` | The backend for loading weights from remote instance. Can be 'transfer_engine' or 'nccl'. Default is 'nccl'. | `nccl` | `transfer_engine`, `nccl` |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | Start seed server via transfer engine backend for remote instance weight loader. | `False` | bool flag (set to enable) |

## For PD-Multiplexing
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-pdmux` | Enable PD-Multiplexing, PD running on greenctx stream. | `False` | bool flag (set to enable) |
| `--pdmux-config-path` | The path of the PD-Multiplexing config file. | `None` | Type: str |
| `--sm-group-num` | Number of sm partition groups. | `8` | Type: int |

## Configuration file support
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--config` | Read CLI options from a config file. Must be a YAML file with configuration options. | `None` | Type: str |

## For Multi-Modal
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--mm-max-concurrent-calls` | The max concurrent calls for async mm data processing. | `32` | Type: int |
| `--mm-per-request-timeout` | The timeout for each multi-modal request in seconds. | `10.0` | Type: int |
| `--enable-broadcast-mm-inputs-process` | Enable broadcast mm-inputs process in scheduler. | `False` | bool flag (set to enable) |
| `--mm-process-config` | Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`. | `{}` | Type: JSON / Dict |
| `--mm-enable-dp-encoder` | Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically. | `False` | bool flag (set to enable) |
| `--limit-mm-data-per-request` | Limit the number of multimodal inputs per request. e.g. '{"image": 1, "video": 1, "audio": 1}' | `None` | Type: JSON / Dict |

## For checkpoint decryption
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--decrypted-config-file` | The path of the decrypted config file. | `None` | Type: str |
| `--decrypted-draft-config-file` | The path of the decrypted draft config file. | `None` | Type: str |
| `--enable-prefix-mm-cache` | Enable prefix multimodal cache. Currently only supports mm-only. | `False` | bool flag (set to enable) |

## Forward hooks
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--forward-hooks` | JSON-formatted list of forward hook specifications. Each element must include `target_modules` (list of glob patterns matched against `model.named_modules()` names) and `hook_factory` (Python import path to a factory, e.g. `my_package.hooks:make_hook`). An optional `name` field is used for logging, and an optional `config` object is passed as a `dict` to the factory. | `None` | Type: JSON list |

## Deprecated arguments
| Argument | Description | Defaults | Options |
| --- | --- | --- | --- |
| `--enable-ep-moe` | NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead. | `None` | N/A |
| `--enable-deepep-moe` | NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead. | `None` | N/A |
| `--prefill-round-robin-balance` | Note: Note: --prefill-round-robin-balance is deprecated now. | `None` | N/A |
| `--enable-flashinfer-cutlass-moe` | NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead. | `None` | N/A |
| `--enable-flashinfer-cutedsl-moe` | NOTE: --enable-flashinfer-cutedsl-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutedsl' instead. | `None` | N/A |
| `--enable-flashinfer-trtllm-moe` | NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead. | `None` | N/A |
| `--enable-triton-kernel-moe` | NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead. | `None` | N/A |
| `--enable-flashinfer-mxfp4-moe` | NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead. | `None` | N/A |
| `--crash-on-nan` | Crash the server on nan logprobs. | `False` | Type: str |
| `--hybrid-kvcache-ratio` | Mix ratio in [0,1] between uniform and hybrid kv buffers (0.0 = pure uniform: swa_size / full_size = 1)(1.0 = pure hybrid: swa_size / full_size = local_attention_size / context_length) | `None` | Optional[float] |
| `--load-watch-interval` | The interval of load watching in seconds. | `0.1` | Type: float |
| `--nsa-prefill` | Choose the NSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek NSA-style attention). | `flashmla_sparse` | `flashmla_sparse`, `flashmla_decode`, `fa3`, `tilelang`, `aiter` |
| `--nsa-decode` | Choose the NSA backend for the decode stage when running DeepSeek NSA-style attention. Overrides `--attention-backend` for decoding. | `flashmla_kv` | `flashmla_prefill`, `flashmla_kv`, `fa3`, `tilelang`, `aiter` |
