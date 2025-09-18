# Server Arguments

This page provides a list of server arguments used in the command line to configure the behavior
and performance of the language model server during deployment. These arguments enable users to
customize key aspects of the server, including model selection, parallelism policies,
memory management, and optimization techniques.
You can find all arguments by `python3 -m sglang.launch_server --help`

## Common launch commands

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

- To enable `torch.compile` acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. By default, the cache path is located at `/tmp/torchinductor_root`, you can customize it using environment variable `TORCHINDUCTOR_CACHE_DIR`. For more details, please refer to [PyTorch official documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) and [Enabling cache for torch.compile](https://docs.sglang.ai/backend/hyperparameter_tuning.html#enabling-cache-for-torch-compile).
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other quantization strategies (INT8/FP8) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](../references/custom_chat_template.md).
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

Please consult the documentation below and [`server_args.py`](python/sglang/srt/server_args.py:1) to learn more about the arguments you may provide when launching a server.

## Model and tokenizer

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--model-path` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. | None |
| `--tokenizer-path` | The path of the tokenizer. | None |
| `--tokenizer-mode` | Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer. | auto |
| `--tokenizer-worker-num` | The worker count of the tokenizer manager (for parallel tokenization). | 1 |
| `--skip-tokenizer-init` | If set, skip init tokenizer and pass input_ids in generate request. | False |
| `--load-format` | The format of the model weights to load. 'auto' will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. 'pt' will load the weights in the pytorch bin format. 'safetensors' will load the weights in the safetensors format. 'npcache' will load the weights in pytorch format and store a numpy cache to speed up the loading. 'dummy' will initialize the weights with random values, which is mainly for profiling. 'gguf' will load the weights in the gguf format. 'bitsandbytes' will load the weights using bitsandbytes quantization. 'layered' loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller. | auto |
| `--model-loader-extra-config` | Extra JSON string passed to the model loader corresponding to the chosen load_format. | {} |
| `--trust-remote-code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. | False |
| `--context-length` | The model's maximum context length. Defaults to None (will use the value from the model's config.json instead). | None |
| `--is-embedding` | Whether to use a CausalLM as an embedding model. | False |
| `--enable-multimodal` | Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen. | None |
| `--revision` | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | None |
| `--model-impl` | Which implementation of the model to use. 'auto' will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. 'sglang' will use the SGLang model implementation. 'transformers' will use the Transformers model implementation. | auto |

## Remote instance weight loading

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--remote-instance-weight-loader-seed-instance-ip` | The IP of the seed instance for loading weights from a remote instance. | None |
| `--remote-instance-weight-loader-seed-instance-service-port` | The service port of the seed instance for loading weights from a remote instance. | None |
| `--remote-instance-weight-loader-send-weights-group-ports` | A JSON list of ports used to form communication groups when loading weights from a remote instance. | None |

## HTTP server

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--host` | The host address for the server. | 127.0.0.1 |
| `--port` | The port number for the server. | 30000 |
| `--skip-server-warmup` | If set, skip the server warmup process. | False |
| `--warmups` | Comma-separated warmup function names to run before the server starts (e.g., warmup_name1,warmup_name2). | None |
| `--nccl-port` | The port for NCCL initialization. | None |

## Quantization and data type

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--dtype` | Data type for model weights and activations. 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. 'half' for FP16. Recommended for AWQ quantization. 'float16' is the same as 'half'. 'bfloat16' for a balance between precision and range. 'float' is shorthand for FP32 precision. 'float32' for FP32 precision. | auto |
| `--quantization` | The quantization method. | None |
| `--quantization-param-path` | Path to the JSON file containing the KV cache scaling factors. This should generally be supplied when KV cache dtype is FP8. Otherwise, scaling factors default to 1.0, which may cause accuracy issues. | None |
| `--kv-cache-dtype` | Data type for kv cache storage. 'auto' will use model data type. 'fp8_e5m2' and 'fp8_e4m3' is supported for CUDA 11.8+. | auto |

## Memory and scheduling

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--mem-fraction-static` | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. | None |
| `--max-running-requests` | The maximum number of running requests. | None |
| `--max-queued-requests` | The maximum number of queued requests. Ignored in disaggregation mode. | None |
| `--max-total-tokens` | The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes. | None |
| `--chunked-prefill-size` | The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill. | None |
| `--max-prefill-tokens` | The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. | 16384 |
| `--schedule-policy` | The scheduling policy of the requests. | fcfs |
| `--enable-priority-scheduling` | Enable priority scheduling. Requests with higher priority integer values are scheduled first by default. | False |
| `--schedule-low-priority-values-first` | With priority scheduling enabled, schedule lower integer values first. | False |
| `--priority-scheduling-preemption-threshold` | Minimum difference in priorities for an incoming request to preempt running request(s). | 10 |
| `--schedule-conservativeness` | How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. | 1.0 |
| `--page-size` | The number of tokens in a page. | 1 |
| `--hybrid-kvcache-ratio` | Mix ratio in [0,1] between uniform and hybrid kv buffers (0.0 = pure uniform; 1.0 = pure hybrid). If provided without value, 0.5 is used. | None |
| `--swa-full-tokens-ratio` | Ratio of SWA layer KV tokens to full layer KV tokens. Should be between 0 and 1. | 0.8 |
| `--disable-hybrid-swa-memory` | Disable the hybrid SWA memory. | False |

## Runtime options

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--device` | The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified. | None |
| `--tp-size` | The tensor parallelism size. | 1 |
| `--pp-size` | The pipeline parallelism size. | 1 |
| `--max-micro-batch-size` | The maximum micro batch size in pipeline parallelism. | None |
| `--stream-interval` | The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher. | 1 |
| `--stream-output` | Whether to output as a sequence of disjoint segments. | False |
| `--random-seed` | The random seed. | None |
| `--constrained-json-whitespace-pattern` | Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*. | None |
| `--watchdog-timeout` | Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging. | 300 |
| `--dist-timeout` | Set timeout for torch.distributed initialization. | None |
| `--download-dir` | Model download directory for huggingface. | None |
| `--base-gpu-id` | The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine. | 0 |
| `--gpu-id-step` | The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,.... | 1 |
| `--sleep-on-idle` | Reduce CPU usage when sglang is idle. | False |

## Logging

| Arguments                             | Description                                                                                                                                                                                                                                                                                                                                                                                  | Defaults |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--log-level`                         | The logging level of all loggers.                                                                                                                                                                                                                                                                                                                                                            | info     |
| `--log-level-http`                    | The logging level of HTTP server. If not set, reuse --log-level by default.                                                                                                                                                                                                                                                                                                                  | None     |
| `--log-requests`                      | Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level.                                                                                                                                                                                                                                                                                             | False    |
| `--log-requests-level`                | 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.                                                                                                                                                                                                                 | 2        |
| `--crash-dump-folder`                 | Folder path to dump requests from the last 5 minutes before a crash (if any). If not specified, crash dumping is disabled.                                                                                                                                                                                                                                                                   | None     |
| `--show-time-cost`                    | Show time cost of custom marks.                                                                                                                                                                                                                                                                                                                                                              | False    |
| `--enable-metrics`                    | Enable log prometheus metrics.                                                                                                                                                                                                                                                                                                                                                               | False    |
| `--enable-metrics-for-all-schedulers` | When enabled, schedulers on all TP ranks (not just TP 0) record request metrics separately. Especially useful when dp_attention is enabled.                                                                                                                                                                                                                                                  | False    |
| `--tokenizer-metrics-custom-labels-header` | HTTP header name for passing customer labels for tokenizer metrics.                                                                                                                                                                                                                                                                                                                          | x-customer-labels |
| `--tokenizer-metrics-allowed-customer-labels` | The list of allowed customer label keys for tokenizer metrics.                                                                                                                                                                                                                                                                                                                                | None     |
| `--bucket-time-to-first-token`        | The buckets of time to first token, specified as a list of floats.                                                                                                                                                                                                                                                                                                                           | None     |
| `--bucket-inter-token-latency`        | The buckets of inter-token latency, specified as a list of floats.                                                                                                                                                                                                                                                                                                                           | None     |
| `--bucket-e2e-request-latency`        | The buckets of end-to-end request latency, specified as a list of floats.                                                                                                                                                                                                                                                                                                                    | None     |
| `--collect-tokens-histogram`          | Collect prompt/generation tokens histogram.                                                                                                                                                                                                                                                                                                                                                  | False    |
| `--prompt-tokens-buckets`             | The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two-sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]); 'customer <value1> <value2> ...' uses custom bucket values (e.g., 'customer 10 50 100 500'). | None     |
| `--generation-tokens-buckets`         | The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two-sides exponential distributed buckets; 'customer <value1> <value2> ...' uses custom bucket values.                                                                                                                                   | None     |
| `--gc-warning-threshold-secs`         | Threshold (seconds) for logging a warning on long garbage collection pauses. Set to 0 to disable.                                                                                                                                                                                                                                                                                            | 0.0      |
| `--decode-log-interval`               | The log interval of decode batch.                                                                                                                                                                                                                                                                                                                                                            | 40       |
| `--enable-request-time-stats-logging` | Enable per request time stats logging.                                                                                                                                                                                                                                                                                                                                                       | False    |
| `--kv-events-config`                  | Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.                                                                                                                                                                                                                                                                                | None     |
| `--enable-trace`                      | Enable OpenTelemetry tracing.                                                                                                                                                                                                                                                                                                                                                                | False    |
| `--oltp-traces-endpoint`              | OpenTelemetry collector endpoint if --enable-trace is set. Format: <ip>:<port>.                                                                                                                                                                                                                                                                                                              | localhost:4317 |

## API related

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--api-key` | Set API key of the server. It is also used in the OpenAI API compatible server. | None |
| `--served-model-name` | Override the model name returned by the v1/models endpoint in OpenAI API server. | None |
| `--weight-version` | Version identifier for the model weights. | default |
| `--chat-template` | The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. | None |
| `--completion-template` | The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently. | None |
| `--file-storage-path` | The path of the file storage in backend. | sglang_storage |
| `--enable-cache-report` | Return number of cached tokens in usage.prompt_tokens_details for each openai request. | False |
| `--reasoning-parser` | Specify the parser for reasoning models. | None |
| `--tool-call-parser` | Specify the parser for handling tool-call interactions. | None |
| `--tool-server` | 'demo' or a comma-separated list of tool server URLs. If not specified, no tool server will be used. | None |

## Data parallelism

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--dp-size` | The data parallelism size. | 1 |
| `--load-balance-method` | The load balancing strategy for data parallelism. Options include: 'round_robin', 'shortest_queue', 'minimum_tokens'. The Minimum Token algorithm can only be used when DP attention is applied. This algorithm performs load balancing based on the real-time token load of the DP workers. | round_robin |
| `--load-watch-interval` | The interval (seconds) to poll DP worker loads for load balancing. | 0.1 |
| `--prefill-round-robin-balance` | Balance prefill requests via round-robin to ensure decode servers get the correct DP rank in PD disaggregation setups. | False |

## Multi-node distributed serving

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--dist-init-addr` | The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). | None |
| `--nnodes` | The number of nodes. | 1 |
| `--node-rank` | The node rank. | 0 |

## Model override args in JSON

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--json-model-override-args` | A dictionary in JSON string format used to override default model configurations. | {} |
| `--preferred-sampling-params` | json-formatted sampling settings that will be returned in /get_model_info. | None |

## LoRA

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-lora` | Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility. | False |
| `--max-lora-rank` | The maximum LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup. | None |
| `--lora-target-modules` | The union set of all target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of different target modules after server startup. You can also set it to `all` to enable LoRA for all supported modules. However, enabling LoRA on additional modules introduces a minor performance overhead. If your application is performance-sensitive, we recommend only specifying the modules for which you plan to load adapters. | None |
| `--lora-paths` | The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> \| <NAME>=<PATH> \| JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool} | None |
| `--max-loras-per-batch` | Maximum number of adapters for a running batch, include base-only request. | 8 |
| `--max-loaded-loras` | If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`. | None |
| `--lora-backend` | Choose the kernel backend for multi-LoRA serving. | triton |

## Kernel backend

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--attention-backend` | Choose the kernels for attention layers. | None |
| `--prefill-attention-backend` | (Experimental) This argument specifies the backend for prefill attention computation. Note that this argument has priority over `attention_backend`. | None |
| `--decode-attention-backend` | (Experimental) This argument specifies the backend for decode attention computation. Note that this argument has priority over `attention_backend`. | None |
| `--sampling-backend` | Choose the kernels for sampling layers. | None |
| `--grammar-backend` | Choose the backend for grammar-guided decoding. | None |
| `--mm-attention-backend` | Set multimodal attention backend. | None |

## Speculative decoding

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--speculative-algorithm` | Speculative algorithm. | None |
| `--speculative-draft-model-path` | The path of the draft model weights. This can be a local folder or a Hugging Face repo ID. | None |
| `--speculative-draft-model-revision` | The specific draft model version to use (branch, tag, or commit id). | None |
| `--speculative-num-steps` | The number of steps sampled from draft model in Speculative Decoding. | None |
| `--speculative-eagle-topk` | The number of tokens sampled from the draft model in eagle2 each step. | None |
| `--speculative-num-draft-tokens` | The number of tokens sampled from the draft model in Speculative Decoding. | None |
| `--speculative-accept-threshold-single` | Accept a draft token if its probability in the target model is greater than this threshold. | 1.0 |
| `--speculative-accept-threshold-acc` | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). | 1.0 |
| `--speculative-token-map` | The path of the draft model's small vocab table. | None |
| `--speculative-attention-mode` | Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'. | prefill |

## Expert parallelism

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--ep-size` | The expert parallelism size. | 1 |
| `--moe-a2a-backend` | Select the backend for all-to-all communication for expert parallelism. | none |
| `--moe-runner-backend` | Select the runner backend for MoE. | auto |
| `--flashinfer-mxfp4-moe-precision` | Computation precision used by FlashInfer MXFP4 MoE. | default |
| `--enable-flashinfer-allreduce-fusion` | Enable FlashInfer AllReduce fusion with Residual RMSNorm. | False |
| `--deepep-mode` | Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch. | auto |
| `--ep-num-redundant-experts` | Allocate this number of redundant experts in expert parallel. | 0 |
| `--ep-dispatch-algorithm` | The algorithm to choose ranks for redundant experts in EPLB. | None |
| `--init-expert-location` | Initial location of EP experts. | trivial |
| `--enable-eplb` | Enable EPLB algorithm. | False |
| `--eplb-algorithm` | Chosen EPLB algorithm. | auto |
| `--eplb-rebalance-num-iterations` | Number of iterations to automatically trigger a EPLB re-balance. | 1000 |
| `--eplb-rebalance-layers-per-chunk` | Number of layers to rebalance per forward pass. | None |
| `--expert-distribution-recorder-mode` | Mode of expert distribution recorder. | None |
| `--expert-distribution-recorder-buffer-size` | Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer. | None |
| `--enable-expert-distribution-metrics` | Enable logging metrics for expert balancedness. | False |
| `--deepep-config` | Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path. | None |
| `--moe-dense-tp-size` | TP size for MoE dense MLP layers. Useful when large TP size causes GEMM min-dimension errors. | None |

## Mamba cache

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--max-mamba-cache-size` | The maximum size of the Mamba cache. | None |
| `--mamba-ssm-dtype` | Data type of the SSM states in Mamba cache. | float32 |

## Hierarchical cache

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-hierarchical-cache` | Enable hierarchical cache. | False |
| `--hicache-ratio` | The ratio of the size of host KV cache memory pool to the size of device pool. | 2.0 |
| `--hicache-size` | The size of host KV cache memory pool in GB (overrides ratio if set). | 0 |
| `--hicache-write-policy` | The write policy for hierarchical cache. One of 'write_back', 'write_through', 'write_through_selective'. | write_through |
| `--hicache-io-backend` | The IO backend for KV cache transfer between CPU and GPU. One of 'direct', 'kernel'. | kernel |
| `--hicache-mem-layout` | The layout of host memory pool for hierarchical cache. One of 'layer_first', 'page_first', 'page_first_direct'. | layer_first |
| `--hicache-storage-backend` | The storage backend for hierarchical KV cache. One of 'file', 'mooncake', 'hf3fs', 'nixl'. | None |
| `--hicache-storage-prefetch-policy` | Control when prefetching from the storage backend should stop. One of 'best_effort', 'wait_complete', 'timeout'. | best_effort |
| `--hicache-storage-backend-extra-config` | JSON string with extra configuration for the storage backend. | None |
| `--enable-lmcache` | Use LMCache as an alternative hierarchical cache solution. | False |

## Double sparsity

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-double-sparsity` | Enable double sparsity attention. | False |
| `--ds-channel-config-path` | Path to the double sparsity channel config. | None |
| `--ds-heavy-channel-num` | Number of heavy channels. | 32 |
| `--ds-heavy-token-num` | Number of heavy tokens. | 256 |
| `--ds-heavy-channel-type` | Type of heavy channels (e.g., 'qk'). | qk |
| `--ds-sparse-decode-threshold` | Threshold of prompt length to enable sparse decode. | 4096 |

## Offloading

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--cpu-offload-gb` | How many GBs of RAM to reserve for CPU offloading. | 0 |
| `--offload-group-size` | Number of layers per group in offloading. | -1 |
| `--offload-num-in-group` | Number of layers to be offloaded within a group. | 1 |
| `--offload-prefetch-step` | Steps to prefetch in offloading. | 1 |
| `--offload-mode` | Mode of offloading. | cpu |

## Optimization/debug options

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--disable-radix-cache` | Disable RadixAttention for prefix caching. | False |
| `--cuda-graph-max-bs` | Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value. | None |
| `--cuda-graph-bs` | Set the list of batch sizes for cuda graph. | None |
| `--disable-cuda-graph` | Disable cuda graph. | False |
| `--disable-cuda-graph-padding` | Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed. | False |
| `--enable-profile-cuda-graph` | Enable profiling of cuda graph capture. | False |
| `--enable-cudagraph-gc` | Enable garbage collection during CUDA graph capture (default is disabled which freezes GC for faster capture). | False |
| `--enable-nccl-nvls` | Enable NCCL NVLS for prefill heavy requests when available. | False |
| `--enable-symm-mem` | Enable NCCL symmetric memory for fast collectives. | False |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | Disable "quantize before all-gather" for FlashInfer CUTLASS MoE (modelopt_fp4). | False |
| `--enable-tokenizer-batch-encode` | Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds. | False |
| `--disable-outlines-disk-cache` | Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency. | False |
| `--disable-custom-all-reduce` | Disable the custom all-reduce kernel and fall back to NCCL. | False |
| `--enable-mscclpp` | Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL. | False |
| `--disable-overlap-schedule` | Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. | False |
| `--enable-mixed-chunk` | Enable mixing prefill and decode in a batch when using chunked prefill. | False |
| `--enable-dp-attention` | Enable data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported. | False |
| `--enable-dp-lm-head` | Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. | False |
| `--enable-two-batch-overlap` | Enable two micro batches to overlap. | False |
| `--tbo-token-distribution-threshold` | Threshold of token distribution between two batches in micro-batch-overlap; switches between two-batch-overlap and two-chunk-overlap. Set 0 to disable two-chunk-overlap. | 0.48 |
| `--enable-torch-compile` | Optimize the model with torch.compile. Experimental feature. | False |
| `--torch-compile-max-bs` | Set the maximum batch size when using torch compile. | 32 |
| `--torchao-config` | Optimize the model with torchao. Experimental feature. Current choices include: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row. |  |
| `--enable-nan-detection` | Enable the NaN detection for debugging purposes. | False |
| `--enable-p2p-check` | Enable P2P check for GPU access, otherwise the p2p access is allowed by default. | False |
| `--triton-attention-reduce-in-fp32` | Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16 (affects Triton attention kernels only). | False |
| `--triton-attention-num-kv-splits` | The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. | 8 |
| `--triton-attention-split-tile-size` | The size of split KV tile in flash decoding Triton kernel (used for deterministic inference). | None |
| `--num-continuous-decode-steps` | Run multiple continuous decoding steps to reduce scheduling overhead. This can increase throughput but may increase time-to-first-token latency. The default value is 1, meaning only 1 decoding step happens at a time. | 1 |
| `--delete-ckpt-after-loading` | Delete the model checkpoint after loading the model. | False |
| `--enable-memory-saver` | Allow saving memory using release_memory_occupation and resume_memory_occupation. | False |
| `--allow-auto-truncate` | Allow automatically truncating requests that exceed the maximum input length instead of returning an error. | False |
| `--enable-custom-logit-processor` | Enable users to pass custom logit processors to the server (disabled by default for security). | False |
| `--flashinfer-mla-disable-ragged` | Disable ragged processing in FlashInfer MLA. | False |
| `--disable-shared-experts-fusion` | Disable shared experts fusion. | False |
| `--disable-chunked-prefix-cache` | Disable chunked prefix cache. | False |
| `--disable-fast-image-processor` | Disable fast image processor. | False |
| `--enable-return-hidden-states` | Enable returning hidden states. | False |
| `--scheduler-recv-interval` | Interval to poll requests in the scheduler. Can be set >1 to reduce polling overhead. | 1 |
| `--numa-node` | NUMA node affinity for subprocesses. i-th element corresponds to i-th subprocess. |  |

## Dynamic batch tokenizer

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-dynamic-batch-tokenizer` | Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently. | False |
| `--dynamic-batch-tokenizer-batch-size` | Maximum batch size for dynamic batch tokenizer (effective only if enabled). | 32 |
| `--dynamic-batch-tokenizer-batch-timeout` | Timeout in seconds for batching tokenization requests (effective only if enabled). | 0.002 |

## Debug tensor dumps

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--debug-tensor-dump-output-folder` | The output folder for debug tensor dumps. | None |
| `--debug-tensor-dump-input-file` | The input file for debug tensor dumps. | None |
| `--debug-tensor-dump-inject` | Enable injection of debug tensor dumps. | False |
| `--debug-tensor-dump-prefill-only` | Enable prefill-only mode for debug tensor dumps. | False |

## PD disaggregation

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--disaggregation-mode` | PD disaggregation mode: "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only). | null |
| `--disaggregation-transfer-backend` | The transfer backend for PD disaggregation. | mooncake |
| `--disaggregation-bootstrap-port` | The bootstrap port for PD disaggregation. | 8998 |
| `--disaggregation-decode-tp` | The decode TP for PD disaggregation (set on prefill server). | None |
| `--disaggregation-decode-dp` | The decode DP for PD disaggregation (set on prefill server). | None |
| `--disaggregation-prefill-pp` | The prefill PP for PD disaggregation (set on decode server). | 1 |
| `--disaggregation-ib-device` | InfiniBand device(s) to use for disaggregation transfer (e.g., mlx5_0 or 'mlx5_0,mlx5_1'). | None |
| `--num-reserved-decode-tokens` | Number of decode tokens to reserve memory for when adding a new request to a running batch. | 512 |
| `--disaggregation-decode-polling-interval` | Interval (in steps) to poll requests on the decode server. | 1 |

## Model weight update

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--custom-weight-loader` | Custom weight loader import paths. | None |
| `--weight-loader-disable-mmap` | Disable mmap for weight loader. | False |

## PD-Multiplexing

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-pdmux` | Enable PD-Multiplexing. | False |
| `--sm-group-num` | Number of SM groups for PD-Multiplexing. | 3 |

## Deprecated arguments

These flags are kept for backward compatibility but are deprecated in favor of newer arguments:

| Arguments | Replacement |
|-----------|-------------|
| `--enable-ep-moe` | Use `--ep-size` with the same value as `--tp-size`. |
| `--enable-deepep-moe` | Use `--moe-a2a-backend deepep`. |
| `--enable-flashinfer-cutlass-moe` | Use `--moe-runner-backend flashinfer_cutlass`. |
| `--enable-flashinfer-cutedsl-moe` | Use `--moe-runner-backend flashinfer_cutedsl`. |
| `--enable-flashinfer-trtllm-moe` | Use `--moe-runner-backend flashinfer_trtllm`. |
| `--enable-triton-kernel-moe` | Use `--moe-runner-backend triton_kernel`. |
| `--enable-flashinfer-mxfp4-moe` | Use `--moe-runner-backend flashinfer_mxfp4`. |
