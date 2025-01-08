# Server Args

## Model and tokenizer

* `model_path`: Path to the model we want to serve.
* `tokenizer_path`: Defaults to the `model_path`.
* `tokenizer_mode`: By default `auto`, see [here](https://huggingface.co/docs/transformers/en/main_classes/tokenizer)
* `load_format`: The format the weights are loaded in. Defaults to `*.safetensors`/`*.bin`.
* `trust_remote_code`:  If `True`, will use local cached config files, other wise use remote configs in HuggingFace..
* `dtype`: Dtype we use for the model, defaults to `bfloat16`. 
* `kv_cache_dtype`: Dtype of the kv cache, defaults to the `dtype`.
* `context_length`: The number of tokens our model can process *including the input*. Not that extending the default might lead to strange behavior.
* `device`: The device we put the model, defaults to `cuda`.
* `chat_template`: The chat template we use. Not that deviating from the default might lead to unexpected responses. 
* `is_embedding`: Set to true if we want to perform embedding task.
* `revision`: Adjust if a specific version of the model should be used.
* `skip_tokenizer_init`: Set to true if you want to provide the tokens to the engine and get the output tokens directly. 


## Port for the HTTP server

* Use `port` and `host` to setup the host for your HTTP server. By default `host: str = "127.0.0.1"` and `port: int = 30000`

## Memory and scheduling

* `mem_fraction_static`: Fraction of the free GPU memory used for static memory like model weights and KV cache. Decrease this if you meet OOM errors.
* `max_running_requests`: The maximum number of requests to run concurrently.
* `max_total_tokens`: The maximum number of tokens that can be stored into the KV cache. Use mainly for debugging.
* `chunked_prefill_size`: Perform the prefill in chunks of these size. Larger chunk size speeds up the prefill phase but increases the VRAM consumption.
* `max_prefill_tokens`: Token budget of how many tokens to accept in one prefill batch is the max of this parameter and the `context_length`.
* `schedule_policy`: The scheduling policy to control the processing order of waiting prefill requests.
* `schedule_conservativeness`: TODO
* `cpu_offload_gb`: TODO
* `prefill_only_one_req`: When this flag is turned on we prefill only one request at a time.

## Other runtime options

* `tp_size`: The number of GPUs the model weights get sharded over. Mainly for memory efficency rather than for high throughput, see [this blogpost](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works).

* `stream_interval`: TODO

* `random_seed`: Can be used to enforce more deterministic behavior. 

* `watchdog_timeout`: Adjusts the watchdog thread’s timeout before killing the server if batch generation takes too long.

* `download_dir`: Use to override the default Hugging Face cache directory for model weights.

* `base_gpu_id`: Use to adjust first GPU used to distribute the model across available GPUs.


## Logging

TODO

## API related

TODO

## Data parallelism 

* `dp_size`: The number of data-parallel copies of the model. For maximum throughput, maximize `dp_size` and only split weights via tensor parallelism as needed for memory. Ensure `dp_size * tp_size = N` where `N` is the total number of GPUs.

* `load_balance_method`: TODO

## Expert parallelism

* `ep_size`: For MoE models we can distribute the experts onto this number of GPUs. Remember to shard the rest of the model weights with `tp_size=ep_size`, for detailed benchmarking [see the PR that implemented this technique](https://github.com/sgl-project/sglang/pull/2203).

## Multi-node distributed serving
TODO

## Model override args in JSON

* `json_model_override_args`: If you want to override the model config provide the adjustment in this argument as a JSON file.

## LoRA

* `lora_paths`: You may provide a list of adapters to your model as a list. Each batch element will get model response with the corresponding lora adapter applied. Currently `cuda_graph` and `radix_attention` are not supportet with this option so you need to disable them manually.
* `max_loras_per_batch`: Maximum number of LoRAs in a running batch including base model.

## Kernel backend

* `attention_backend`: If you want to change the attention backend for some reason you may adjust the default `flashinfer` backend to `triton` or `torch_native` backend. 
* `sampling_backend`: If you want to change the sampling backend for some reason you may adjust the default `flashinfer` backend to `torch_native` backend.
* `grammar_backend`: You may want to change the default `outlines` grammar backend to `xgrammar` backend for [10 x speedup](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#fast-structured-outputs-with-xgrammar) in case you want to perform constrained decoding.

## Speculative decoding
TODO

## Double Sparsity
TODO

## Optimization/debug options
TODO