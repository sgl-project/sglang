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
* `schedule_conservativeness`: If we decrease this parameter from 1 towards 0 we will make the server less conservative about taking new requests. Similar increasing to a value above one will make the server less conservative. A lower value indicates we suspect `max_total_tokens` is set to a value that is too large. See [python/sglang/srt/managers/scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
* `cpu_offload_gb`
* `prefill_only_one_req`: When this flag is turned on we prefill only one request at a time