# Server Arguments

## Common launch commands

- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
```
- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total. We recommend [SGLang Router](https://docs.sglang.ai/router/router.html) for data parallelism.
```
python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
```

- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
```
- See [hyperparameter tuning](../references/hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
```
- To enable torch.compile acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. This does not work for FP8 currently.
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other [quantization strategies (INT8/FP8)](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L671) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](../references/custom_chat_template.md).

- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1
```

Please consult the documentation below to learn more about the parameters you may provide when launching a server.


## Model and tokenizer

* `model_path`: Path to the model that will be served.
* `tokenizer_path`: Defaults to the `model_path`.
* `tokenizer_mode`: By default `auto`, see [here](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) for different mode.
* `load_format`: The format the weights are loaded in. Defaults to `*.safetensors`/`*.bin`.
* `trust_remote_code`:  If `True`, will use locally cached config files, otherwise use remote configs in HuggingFace.
* `dtype`: Dtype used for the model, defaults to `bfloat16`.
* `kv_cache_dtype`: Dtype of the kv cache, defaults to the `dtype`.
* `context_length`: The number of tokens our model can process *including the input*. Note that extending the default might lead to strange behavior.
* `device`: The device we put the model, defaults to `cuda`.
* `chat_template`: The chat template to use. Deviating from the default might lead to unexpected responses. For multi-modal chat templates, refer to [here](https://docs.sglang.ai/backend/openai_api_vision.html#Chat-Template).
* `is_embedding`: Set to true to perform [embedding](https://docs.sglang.ai/backend/openai_api_embeddings.html) / [encode](https://docs.sglang.ai/backend/native_api.html#Encode-(embedding-model)) and [reward](https://docs.sglang.ai/backend/native_api.html#Classify-(reward-model)) tasks.
* `revision`: Adjust if a specific version of the model should be used.
* `skip_tokenizer_init`: Set to true to provide the tokens to the engine and get the output tokens directly, typically used in RLHF.
* `json_model_override_args`: Override model config with the provided JSON.
* `delete_ckpt_after_loading`: Delete the model checkpoint after loading the model.

> [!IMPORTANT]
> **Make sure the correct `chat_template` is passed, or performance degradation may occur.**

## Serving: HTTP & API

### HTTP Server configuration

* `port` and `host`: Setup the host for HTTP server. By default `host: str = "127.0.0.1"` and `port: int = 30000`

### API configuration

* `api_key`: Sets an API key for the server and the OpenAI-compatible API.
* `file_storage_pth`: Directory for storing uploaded or generated files from API calls.
* `enable_cache_report`: If set, includes detailed usage of cached tokens in the response usage.

## Parallelism

### Tensor parallelism

* `tp_size`: The number of GPUs the model weights get sharded over. Mainly for saving memory rather than for high throughput, see [this blogpost](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works).

### Data parallelism

* `dp_size`: Will be deprecated. The number of data-parallel copies of the model. [SGLang router](https://docs.sglang.ai/router/router.html) is recommended instead of the current naive data parallel.
* `load_balance_method`: Will be deprecated. Load balancing strategy for data parallel requests.

### Expert parallelism
* `enable_ep_moe`: Enables expert parallelism that distributes the experts onto multiple GPUs for MoE models.
* `ep_size`: The size of EP. Please shard the model weights with `tp_size=ep_size`, for detailed benchmarking refer to [this PR](https://github.com/sgl-project/sglang/pull/2203). If not set, `ep_size` will be automatically set to `tp_size`.

## Memory and scheduling

* `mem_fraction_static`: Fraction of the free GPU memory used for static memory like model weights and KV cache. If building KV cache fails, it should be increased. If CUDA runs out of memory, it should be decreased.
* `max_running_requests`: The maximum number of requests to run concurrently.
* `max_total_tokens`: The maximum number of tokens that can be stored into the KV cache. Use mainly for debugging.
* `chunked_prefill_size`: Perform the prefill in chunks of these size. Larger chunk size speeds up the prefill phase but increases the VRAM consumption. If CUDA runs out of memory, it should be decreased.
* `max_prefill_tokens`: Token budget of how many tokens to accept in one prefill batch. The actual number is the max of this parameter and the `context_length`.
* `schedule_policy`: The scheduling policy to control the processing order of waiting prefill requests in a single engine.
* `schedule_conservativeness`: Can be used to decrease/increase the conservativeness of the server when taking new requests. Highly conservative behavior leads to starvation, but low conservativeness leads to slowed-down performance.
* `cpu_offload_gb`: Reserve this amount of RAM in GB for offloading of model parameters to the CPU.
* `prefill_only_one_req`: When this flag is turned on, the engine prefills only one request at a time.

## Other runtime options

* `stream_interval`: Interval (in tokens) for streaming responses. Smaller values lead to smoother streaming, and larger values lead to better throughput.
* `random_seed`: Can be used to enforce more deterministic behavior.
* `watchdog_timeout`: Adjusts the watchdog thread’s timeout before killing the server if batch generation takes too long.
* `download_dir`: Use to override the default Hugging Face cache directory for model weights.
* `base_gpu_id`: Use to adjust first GPU used to distribute the model across available GPUs.
* `allow_auto_truncate`: Automatically truncate requests that exceed the maximum input length.

## Logging

* `log_level`: Global log verbosity.
* `log_level_http`: Separate verbosity level for the HTTP server logs (if unset, defaults to `log_level`).
* `log_requests`: Logs the inputs and outputs of all requests for debugging.
* `show_time_cost`: Prints or logs detailed timing info for internal operations (helpful for performance tuning).
* `enable_metrics`: Exports Prometheus-like metrics for request usage and performance.
* `decode_log_interval`: How often (in tokens) to log decode progress.

## Multi-node distributed serving

* `dist_init_addr`: The TCP address used for initializing PyTorch’s distributed backend (e.g. `192.168.0.2:25000`).
* `nnodes`: Total number of nodes in the cluster. Refer to how to run the [Llama 405B model](https://docs.sglang.ai/references/llama_405B.html#run-405b-fp16-on-two-nodes).
* `node_rank`: Rank (ID) of this node among the `nnodes` in the distributed setup.


## LoRA

* `lora_paths`: You may provide a list of adapters to your model as a list. Each batch element will get model response with the corresponding lora adapter applied. Currently `cuda_graph` and `radix_attention` are not supportet with this option so you need to disable them manually. We are still working on through these [issues](https://github.com/sgl-project/sglang/issues/2929).
* `max_loras_per_batch`: Maximum number of LoRAs in a running batch including base model.
* `lora_backend`: The backend of running GEMM kernels for Lora modules, can be one of `triton` or `flashinfer`. Defaults to be `triton`.

## Kernel backend

* `attention_backend`: The backend for attention computation and KV cache management.
* `sampling_backend`: The backend for sampling.

## Constrained Decoding

* `grammar_backend`: The grammar backend for constraint decoding. Detailed usage can be found in this [document](https://docs.sglang.ai/backend/structured_outputs.html).
* `constrained_json_whitespace_pattern`: Use with `Outlines` grammar backend to allow JSON with syntatic newlines, tabs or multiple spaces. Details can be found [here](https://dottxt-ai.github.io/outlines/latest/reference/generation/json/#using-pydantic).

## Speculative decoding

* `speculative_draft_model_path`: The draft model path for speculative decoding.
* `speculative_algorithm`: The algorithm for speculative decoding. Currently only [Eagle](https://arxiv.org/html/2406.16858v1) is supported. Note that the radix cache, chunked prefill, and overlap scheduler are disabled when using eagle speculative decoding.
* `speculative_num_steps`: How many draft passes we run before verifying.
* `speculative_num_draft_tokens`: The number of tokens proposed in a draft.
* `speculative_eagle_topk`: The number of top candidates we keep for verification at each step for [Eagle](https://arxiv.org/html/2406.16858v1).


## Double Sparsity

* `enable_double_sparsity`: Enables [double sparsity](https://arxiv.org/html/2408.07092v2) which increases throughput.
* `ds_channel_config_path`: The double sparsity config. For a guide on how to generate the config for your model see [this repo](https://github.com/andy-yang-1/DoubleSparse/tree/main/config).
* `ds_heavy_channel_num`: Number of channel indices to keep for each layer.
* `ds_heavy_token_num`: Number of tokens used for attention during decode. Skip sparse decoding if `min_seq_len` in batch < this number.
* `ds_heavy_channel_type`: The type of heavy channels. Either `q`, `k` or `qk`.
* `ds_sparse_decode_threshold`: Don't apply sparse decoding if `max_seq_len` in batch < this threshold.

## Debug options

*Note: We recommend to stay with the defaults and only use these options for debugging for best possible performance.*

* `disable_radix_cache`: Disable [Radix](https://lmsys.org/blog/2024-01-17-sglang/) backend for prefix caching.
* `disable_jump_forward`: Disable [jump-forward](https://lmsys.org/blog/2024-02-05-compressed-fsm/#our-method-jump-forward-decoding-with-a-compressed-finite-state-machine) for outlines grammar backend.
* `disable_cuda_graph`: Disable [cuda graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) for model forward. Use if encountering uncorrectable CUDA ECC errors.
* `disable_cuda_graph_padding`: Disable cuda graph when padding is needed. In other case still use cuda graph.
* `disable_outlines_disk_cache`: Disable disk cache for outlines grammar backend.
* `disable_custom_all_reduce`: Disable usage of custom all reduce kernel.
* `disable_mla`: Disable [Multi-Head Latent Attention](https://arxiv.org/html/2405.04434v5) for Deepseek model.
* `disable_overlap_schedule`: Disable the [Overhead-Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler).
* `enable_nan_detection`: Turning this on makes the sampler print a warning if the logits contain `NaN`.
* `enable_p2p_check`: Turns off the default of allowing always p2p check when accessing GPU.
* `triton_attention_reduce_in_fp32`: In triton kernels this will cast the intermediate attention result to `float32`.

## Optimization

*Note: Some of these options are still in experimental stage.*

* `enable_mixed_chunk`: Enables mixing prefill and decode, see [this discussion](https://github.com/sgl-project/sglang/discussions/1163).
* `enable_dp_attention`: Enable [Data Parallelism Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models) for Deepseek models. Note that you need to choose `dp_size = tp_size` for this.
* `enable_torch_compile`: Torch compile the model. This is an experimental feature.
* `torch_compile_max_bs`: The maximum batch size when using `torch_compile`.
* `cuda_graph_max_bs`: Adjust the maximum batchsize when using cuda graph. By default this is chosen for you based on GPU specifics.
* `cuda_graph_bs`: The batch sizes to capture by `CudaGraphRunner`. By default this is done for you.
* `torchao_config`: Experimental feature that optimizes the model with [torchao](https://github.com/pytorch/ao). Possible choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row.
* `triton_attention_num_kv_splits`: Use to adjust the number of KV splits in triton kernels. Default is 8.
