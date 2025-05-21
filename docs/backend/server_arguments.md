# Server Arguments

This page provides a list of server arguments used in the command line to configure the behavior
and performance of the language model server during deployment. These arguments enable users to
customize key aspects of the server, including model selection, parallelism policies,
memory management, and optimization techniques.

## Common launch commands

- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total. We recommend [SGLang Router](../router/router.md) for data parallelism.

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
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other [quantization strategies (INT8/FP8)](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L671) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](custom_chat_template.md).
- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`

  ```bash
  # Node 0
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

  # Node 1
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1
  ```

Please consult the documentation below and [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) to learn more about the arguments you may provide when launching a server.

## Model, processor and tokenizer

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `model_path` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. | None |
| `tokenizer_path` | The path of the tokenizer. Defaults to the `model_path`. | None |
| `tokenizer_mode` | See [different mode](https://huggingface.co/docs/transformers/en/main_classes/tokenizer). | `auto` |
| `load_format` | The format of the model weights to load.  | `auto` |
| `trust_remote_code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. | `False` |
| `dtype` | Dtype used for the model. | `auto` |
| `kv_cache_dtype` | Dtype of the kv cache. | `auto` |
| `context_length` | The model's maximum context length. Defaults to None (will use the value from the model's config.json instead). Note that extending the default might lead to strange behavior. | None |
| `device` | The device we put the model. | None |
| `served_model_name` | Override the model name returned by the v1/models endpoint in OpenAI API server.| None |
| `is_embedding` | Set to `true` to perform [embedding](./openai_api_embeddings.ipynb) / [encode](https://docs.sglang.ai/backend/native_api#Encode-(embedding-model)) and [reward](https://docs.sglang.ai/backend/native_api#Classify-(reward-model)) tasks. | `False` |
| `revision` | Adjust if a specific version of the model should be used. | None |
| `skip_tokenizer_init` | Set to `true` to provide the tokens to the engine and get the output tokens directly, typically used in RLHF. See [example](https://github.com/sgl-project/sglang/blob/main/examples/runtime/token_in_token_out/). | `False` |
| `json_model_override_args` | A dictionary in JSON string format used to override default model configurations. | `"{}"` |
| `disable_fast_image_processor` | Adopt base image processor instead of fast image processor (which is by default). See [details](https://huggingface.co/docs/transformers/main/en/main_classes/image_processor#image-processor). | `False` |

## Serving: HTTP & API

### HTTP Server configuration

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `host` | Host for the HTTP server. | `"127.0.0.1"` |
| `port` | Port for the HTTP server. | `30000` |

### API configuration

| Arguments | Description | Defaults |
|-----------|-------------|---------|
| `api_key` | Sets an API key for the server and the OpenAI-compatible API. | None |
| `file_storage_path` | Directory for storing uploaded or generated files from API calls. | `"sglang_storage"` |
| `enable_cache_report` | If set, includes detailed usage of cached tokens in the response usage. | `False` |

## Parallelism

### Tensor parallelism

| Argument | Description | Default |
|----------|-------------|---------|
| `tp_size` | The number of GPUs the model weights get sharded over. Mainly for saving memory rather than for high throughput, see [this tutorial: How Tensor Parallel works?](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works). | `1` |

### Data parallelism

| Arguments | Description | Defaults |
|-----------|-------------|---------|
| `dp_size` | For non-DeepSeek models, this is the the number of data-parallel copies of the model. For DeepSeek models, this is the group size of [data parallel attention](https://docs.sglang.ai/references/deepseek.html#data-parallelism-attention) on DeepSeek models. | `1` |
| `load_balance_method` | Will be deprecated. Load balancing strategy for data parallel requests. | `"round_robin"` |

### Expert parallelism

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `enable_ep_moe` | Enables expert parallelism that distributes the experts onto multiple GPUs for MoE models. | `False` |
| `ep_size` | The size of EP. Please shard the model weights with `tp_size=ep_size`. For benchmarking, refer to [this PR](https://github.com/sgl-project/sglang/pull/2203). | `1` |
| `enable_deepep_moe` | Enables expert parallelism that distributes the experts onto multiple GPUs for DeepSeek-V3 model based on `deepseek-ai/DeepEP`. | `False` |
| `deepep_mode` | Select the mode when using DeepEP MoE: can be `normal`, `low_latency`, or `auto`. `auto` means `low_latency` for decode batch and `normal` for prefill batch. | `auto` |

## Memory and scheduling

| Arguments | Description | Defaults |
|----------|-------------|----------|
| `mem_fraction_static` | Fraction of the free GPU memory used for static memory like model weights and KV cache. Increase it if KV cache building fails. Decrease it if CUDA runs out of memory. | None |
| `max_running_requests` | The maximum number of requests to run concurrently. | None |
| `max_total_tokens` | The maximum number of tokens that can be stored in the KV cache. Mainly used for debugging. | None |
| `chunked_prefill_size` | Perform prefill in chunks of this size. Larger sizes speed up prefill but increase VRAM usage. Decrease if CUDA runs out of memory. | None |
| `max_prefill_tokens` | Token budget for how many tokens can be accepted in one prefill batch. The actual limit is the max of this value and `context_length`. | `16384` |
| `schedule_policy` | The scheduling policy to control how waiting prefill requests are processed by a single engine. | `"fcfs"` |
| `schedule_conservativeness` | Controls how conservative the server is when accepting new prefill requests. High conservativeness may cause starvation; low conservativeness may slow down decode. | `1.0` |
| `cpu_offload_gb` | Amount of RAM (in GB) to reserve for offloading model parameters to the CPU. | `0` |

## Other runtime options

| Arguments | Description | Defaults |
|-----------|-------------|---------|
| `stream_interval` | Interval (in tokens) for streaming responses. Smaller values lead to smoother streaming; larger values improve throughput. | `1` |
| `random_seed` | Can be used to enforce more deterministic behavior. | None |
| `watchdog_timeout` | Timeout setting for the watchdog thread before it kills the server if batch generation takes too long. | `300` |
| `download_dir` | Overrides the default Hugging Face cache directory for model weights. | None |
| `base_gpu_id` | Sets the first GPU to use when distributing the model across multiple GPUs. | `0` |
| `allow_auto_truncate`| Automatically truncate requests that exceed the maximum input length. | `False` |

## Logging

| Arguments | Description | Defaults |
|-----------|-------------|---------|
| `log_level` | Global log verbosity. | `"info"` |
| `log_level_http` | Separate verbosity level for the HTTP server logs. | None |
| `log_requests` | Logs the inputs and outputs of all requests for debugging. | `False` |
| `log_requests_level` | Ranges from 0 to 2: level 0 only shows some basic metadata in requests, level 1 and 2 show request details (e.g., text, images), and level 1 limits output to 2048 characters. | `0` |
| `show_time_cost` | Prints or logs detailed timing info for internal operations (helpful for performance tuning). | `False` |
| `enable_metrics` | Exports Prometheus-like metrics for request usage and performance. | `False` |
| `decode_log_interval` | How often (in tokens) to log decode progress. | `40` |

## Multi-node distributed serving

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `dist_init_addr` | The TCP address used for initializing PyTorch's distributed backend (e.g. `192.168.0.2:25000`). | None |
| `nnodes` | Total number of nodes in the cluster. See [Llama 405B guide](https://docs.sglang.ai/references/multi_node.html#llama-3-1-405b). | `1` |
| `node_rank` | Rank (ID) of this node among the `nnodes` in the distributed setup. | `0` |

## LoRA

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `lora_paths` | List of adapters to apply to your model. Each batch element uses the proper LoRA adapter. `radix_attention` is not supported with this, so it must be disabled manually. See related [issues](https://github.com/sgl-project/sglang/issues/2929). | None |
| `max_loras_per_batch` | Maximum number of LoRAs allowed in a running batch, including the base model. | `8` |
| `lora_backend` | Backend used to run GEMM kernels for LoRA modules. Can be `triton` or `flashinfer`. | `triton` |

## Kernel backend

| Arguments              | Description | Defaults |
|------------------------|-------------|---------|
| `attention_backend`    | This argument specifies the backend for attention computation and KV cache management, which can be `fa3`, `flashinfer`, `triton`, `cutlass_mla`, or `torch_native`. When deploying DeepSeek models, use this argument to specify the MLA backend. | None |
| `sampling_backend`     | Specifies the backend used for sampling. | None |
| `mm_attention_backend` | Set multimodal attention backend.

## Constrained Decoding

| Arguments | Description | Defaults |
|----------|-------------| ----------|
| `grammar_backend` | The grammar backend for constraint decoding. See [detailed usage](https://docs.sglang.ai/backend/structured_outputs.html). | None |
| `constrained_json_whitespace_pattern` | Use with `Outlines` grammar backend to allow JSON with syntactic newlines, tabs, or multiple spaces. See [details](https://dottxt-ai.github.io/outlines/latest/reference/generation/json/#using-pydantic). |

## Speculative decoding

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `speculative_draft_model_path` | The draft model path for speculative decoding. | None |
| `speculative_algorithm` | The algorithm for speculative decoding. Currently [EAGLE](https://arxiv.org/html/2406.16858v1) and [EAGLE3](https://arxiv.org/pdf/2503.01840) are supported. Note that the radix cache, chunked prefill, and overlap scheduler are disabled when using eagle speculative decoding. | None |
| `speculative_num_steps` | How many draft passes we run before verifying. | None |
| `speculative_num_draft_tokens` | The number of tokens proposed in a draft. | None |
| `speculative_eagle_topk` | The number of top candidates we keep for verification at each step for [Eagle](https://arxiv.org/html/2406.16858v1). | None |
| `speculative_token_map` | Optional, the path to the high frequency token list of [FR-Spec](https://arxiv.org/html/2502.14856v1), used for accelerating [Eagle](https://arxiv.org/html/2406.16858v1). | None |

## Debug options

*Note: We recommend to stay with the defaults and only use these options for debugging for best possible performance.*

| Arguments | Description | Defaults |
|----------|-------------|---------|
| `disable_radix_cache` | Disable [Radix](https://lmsys.org/blog/2024-01-17-sglang/) backend for prefix caching. | `False` |
| `disable_cuda_graph` | Disable [CUDA Graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) for model forward. Use if encountering uncorrectable CUDA ECC errors. | `False` |
| `disable_cuda_graph_padding` | Disable CUDA Graph when padding is needed; otherwise, still use CUDA Graph. | `False` |
| `disable_outlines_disk_cache` | Disable disk cache for outlines grammar backend. | `False` |
| `disable_custom_all_reduce` | Disable usage of custom all-reduce kernel. | `False` |
| `disable_overlap_schedule` | Disable the [Overhead-Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler). | `False` |
| `enable_nan_detection` | Enable warning if the logits contain `NaN`. | `False` |
| `enable_p2p_check` | Turns off the default of always allowing P2P checks when accessing GPU. | `False` |
| `triton_attention_reduce_in_fp32` | In Triton kernels, cast the intermediate attention result to `float32`. | `False` |

## Optimization

*Note: Some of these options are still in experimental stage.*

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `enable_mixed_chunk` | Enables mixing prefill and decode, see [this discussion](https://github.com/sgl-project/sglang/discussions/1163). | `False` |
| `enable_dp_attention` | Enable [Data Parallelism Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models) for Deepseek models. | `False` |
| `enable_torch_compile` | Torch compile the model. Note that compiling a model takes a long time but has a great performance boost. The compiled model can also be [cached for future use](https://docs.sglang.ai/backend/hyperparameter_tuning.html#enabling-cache-for-torch-compile). | `False` |
| `torch_compile_max_bs` | The maximum batch size when using `torch_compile`. | `32` |
| `cuda_graph_max_bs` | Adjust the maximum batchsize when using CUDA graph. By default this is chosen for you based on GPU specifics. | None |
| `cuda_graph_bs` | The batch sizes to capture by `CudaGraphRunner`. By default this is done for you. | None |
| `torchao_config` | Experimental feature that optimizes the model with [torchao](https://github.com/pytorch/ao). Possible choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row. | `int8dq` |
| `triton_attention_num_kv_splits` | Use to adjust the number of KV splits in triton kernels. | `8` |
| `flashinfer_mla_disable_ragged` | Disable the use of the [ragged prefill](https://github.com/flashinfer-ai/flashinfer/blob/5751fc68f109877f6e0fc54f674cdcdef361af56/docs/tutorials/kv_layout.rst#L26) wrapper for the FlashInfer MLA attention backend. Ragged prefill increases throughput by computing MHA instead of paged MLA when there is no prefix match. Only use it when FlashInfer is being used as the MLA backend. | `False` |
| `disable_chunked_prefix_cache` | Disable the use of chunked prefix cache for DeepSeek models. Only use it when FA3 is attention backend. | `False` |
| `enable_dp_lm_head` | Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. | `False` |
