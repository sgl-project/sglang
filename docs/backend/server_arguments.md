## Model and tokenizer

* `model_path`: The model we want to serve. We can get the path from the corresponding Huggingface Repo.
* `tokenizer_path`: The path to the tokenizer. If not provided this will agree with the `model_path` by default.
* `tokenizer_mode`: By default `auto`. If set to `slow` this will disable the fast version of Huggingface tokenizer, see [here](https://huggingface.co/docs/transformers/en/main_classes/tokenizer)
* `load_format`: The format the weights are loaded in. Defaults to `*.safetensors`/`*.bin` format. See [python/sglang/srt/model_loader/loader.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py)
* `trust_remote_code`: Needed for Config from HF models.
* `dtype`: The dtype we use our model in. Note that compute capability must not be below `sm80` to use `bfloat16`. 
* `kv_cache_dtype`: Dtype of the kv cache. By default is set to the dtype of the model, see [python/sglang/srt/model_executor/model_runner.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py)
* `quantization`: For a list of supported quantizations see: [python/sglang/srt/configs/model_config.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py). Defaults to `None`.
* `context_length`: The number of tokens our model can process *including the input*. By default this is derived from the HF model. Be aware that inappropriate setting of this (i.e. exceeding the default context length) might lead to strange behavior.
* `device`: The device we put the model on. Defaults to `cuda`.
* `served_model_name`: We might serve the same model multiple times. This parameter let's us distinguish between them..
* `chat_template`: The chat template we use. See [python/sglang/lang/chat_template.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py). Be aware that the wrong chat template might lead to unexpeted behavior. By default this is chosen for us.
* `is_embedding`: Set to true if we want to perform embedding extraction.
* `revision`: If we want to choose a specific revision of our model.
* `skip_tokenizer_init`: Set to true if you want to provide the tokenized text instead of text (See [test/srt/test_skip_tokenizer_init.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_skip_tokenizer_init.py) for usage). 
* `return_token_ids`: Set to true if we don't want to decode the model output.

## Port for the HTTP server

* Use `port` and `host` to setup the host for your HTTP server. By default `host: str = "127.0.0.1"` and `port: int = 30000`

## Memory and scheduling

* `mem_fraction_static`: Fraction of the GPU used for static memory like model weights and KV cache.
* `max_running_requests`: The maximum number of requests to run concurrently.
* `max_total_tokens`: Global capacity of tokens that can be stored in the KV cache.
* `chunked_prefill_size`: Perform the prefill in chunks of these size. Larger chunk size speeds up the prefill phase but increases the time taken to complete decoding of other ongoing requests.
* `max_prefill_tokens`: The maximum number of tokens we can prefill.
* `schedule_policy`: The policy which controls in which order to process the waiting prefill requests.. See [python/sglang/srt/managers/schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
* `schedule_conservativeness`: If we decrease this parameter from 1 towards 0 we will make the server less conservative about taking new requests. Similar increasing to a value above one will make the server more conservative. A lower value indicates we suspect `max_total_tokens` is set to a value that is too large. See [python/sglang/srt/managers/scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
* `cpu_offload_gb`: TODO
* `prefill_only_one_req`: When this flag is turned on we prefill only one request at a time

## Other runtime options

* `tp_size`: This parameter is important if we have multiple GPUs and our model doesn't fit on a single GPU. *Tensor parallelism* means we distribute our model weights over multiple GPUs. Note that his technique is mainly aimed at *memory efficency* and not at a *higher throughput* as there is inter GPU communication needed to obtain the final output of each layer. For better understanding of the concept you may look for example [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works).

* `stream_interval`: If we stream the output to the user this parameter determines at which interval we perform streaming. The interval length is measured in tokens.

* `random_seed`: Can be used to enforce deterministic behavior. 

* `constrained_json_whitespace_pattern`: When using `Outlines` grammar backend we can use this to allow JSON with syntatic newlines, tabs or multiple spaces.

* `watchdog_timeout`: With this flag we can adjust the timeout the watchdog thread in the Scheduler uses to kill of the server if a batch generation takes too much time.

* `download_dir`: By default the model weights get loaded into the huggingface cache directory. This parameter can be used to adjust this behavior.

* `base_gpu_id`: This parameter is used to initialize from which GPU we start to distribute our model onto the available GPUs.

## Logging

TODO

## API related

TODO

## Data parallelism 

* `dp_size`: In the case of data parallelism we distribute our weights onto multiple GPUs and divide the batch on multiple GPUs. Note that this can also be combined with tensor parallelism. For example if we had 4 GPUS and our model doesn't fit onto a single GPUs but on two we might choose `tp_size=2` and `dp_size=2`. This would than mean that we have 2 full copies of the model, each sharded onto two GPUs. We can than feed half of the batch two the first copy on the model and the other half to the second copy. If memory allows you should prefer data parallism to tensor parallelism as it doesn't require the overhead of GPU inter communication. Keep in mind that if `N` is the number of GPUs in order to leverage full compute we must choose `dp_size * tp_size = N`.

* `load_balance_method`: TODO

## Expert parallelism

* `ep_size`: This can be used for M(ixture)O(f)E(xperts) Models like `neuralmagic/DeepSeek-Coder-V2-Instruct-FP8`. With this flag each expert layer is distributed according to this flag. The flag should match `tp_size`. For example we have a model with 4 experts and 4 GPUs than `tp_size=4` and `ep_size=4` will result in the usual sharding for all but the expert layers. The expert layers get than sharded such that each GPU processes one expert. A detailed performance analysis was performed [in the PR that implemented this technique](https://github.com/sgl-project/sglang/pull/2203).