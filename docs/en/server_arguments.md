# Server Arguments

All the arguments can be found in the [`server_args.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py).

## Model and Tokenizer Configuration

### `model_path`: str
- **Description**: The path to the model weights. It's required.
- **Usage**: Can be a local folder or a Hugging Face repo ID.

### `tokenizer_path`: Optional[str]
- **Description**: The path to the tokenizer.
- **Note**: It defaults to the same value as `model_path` if not provided.

### `tokenizer_mode`: str
- **Description**: Specifies [the tokenizer mode](https://huggingface.co/learn/nlp-course/chapter6/3).
- **Default**: "auto"
- **Options**: 
  - "auto": Uses the fast tokenizer if available
  - "slow": Always use the slow tokenizer

### `skip_tokenizer_init`: bool
- **Description**: If set, skips tokenizer initialization.
- **Default**: False
- **Usage**: When True, `input_ids` should be passed directly when sending requests. Otherwise, pass in the full prompt directly.

### `load_format`: str

- **Description**: The format of the model weights to load.
- **Default**: "auto"
- **Options**:
  - "auto": Tries safetensors, falls back to pytorch bin
  - "pt": PyTorch bin format
  - "safetensors": SafeTensors format
  - "npcache": PyTorch format with numpy cache for faster loading
  - "dummy": Initializes weights with random values (for profiling)

### `dtype`: str
- **Description**: Data type for model weights and activations.
- **Default**: "auto"
- **Options**:
  - "auto": FP16 for FP32/FP16 models, BF16 for BF16 models
  - "half"/"float16": FP16 precision (recommended for AWQ quantization)
  - "bfloat16": Balanced precision and range
  - "float"/"float32": FP32 precision
- **Note**: [TODO]

### `kv_cache_dtype`: str

- **Description**: Data type for KV cache storage.
- **Default**: "auto"
- **Options**:
  - "auto": Uses model data type
  - "fp8_e5m2": Supported for CUDA 11.8+

### `trust_remote_code`: bool

- **Description**: Allows custom tokenizer defined in our own path. In the original HuggingFace implementation, if you clone a model to your local device, take [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) as an example, it usually contains the [modeling file](https://huggingface.co/openbmb/MiniCPM3-4B/blob/main/modeling_minicpm.py) and [tokenization file](https://huggingface.co/openbmb/MiniCPM3-4B/blob/main/tokenization_minicpm.py). If you want to adjust the local model's behavior. You should change these files and set  `trust_remote_code` to True. The "remote" here means that, for the HuggingFace hub, your local files are the "remote". However, in SGLang, we already pre-defined the modeling files in the [models' file](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models), so if you want to adjust local model behavior, you should change the model class in the [models' file](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). In other words, for SGLang the `trust_remote_code=True` is useful only when you change the tokenization in your local path.
- **Default**: True
- **Note**: Set to False for "Alibaba-NLP/gte-Qwen2-1.5B-instruct" model due to tokenizer issues.

### `context_length`: Optional[int]
- **Description**: The model's maximum context length.
- **Default**: None (uses the value from the model's `config.json`)
- **Note**: If you pass in an extremely long prompt to the engine, the prompt won't be truncated, instead, this request will fail and return an error code of 400, indicating that the context is out of length. Also, do not set the `context_length` larger than the default configuration, since if you do not do the [ROPE extension](https://blog.eleuther.ai/yarn/), the model will respond with meaningless contents after the input prompt passes the length of the default configuration.

### `quantization`: Optional[str]
- **Description**: The quantization method to use.
- **Default**: None
- **Options**: "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "squeezellm", "bitsandbytes"

### `served_model_name`: Optional[str]
- **Description**: Overrides the model name returned by the v1/models endpoint in OpenAI API server.
- **Default**: None (uses `model_path` if not specified)
- **Note**: For the `model` arguments while sending requests to the server, if you do not set `served_model_name` when you launch the server, it should be the same as the `model_path`. Otherwise, it's the `served_model_name`.

### `chat_template`: Optional[str]
- **Description**: The built-in chat template name or path to a chat template file.
- **Default**: None
- **Usage**: Used only for OpenAI-compatible API servers.
- **Note**: Follow [this document](https://sglang.readthedocs.io/en/latest/custom_chat_template.html) to set your Customized Chat Template.

### `is_embedding`: bool

- **Description**: Whether to use the model as an embedding model.
- **Default**: False
- **Note**: For the "Alibaba-NLP/gte-Qwen2-1.5B-instruct" model, it can be used both for casual completions and to generate embeddings. So setting this parameter is required when you want to use your model as an embedding model.

## Port Configuration

### `host`: str
- **Description**: The host address on which the server will run.
- **Default**: "127.0.0.1"
- **Usage**: Specifies the IP address or hostname for the server to bind to.
- **Note**: Using host `0.0.0.0` to enable external access, i.e., allowing other clients from the same subnet to send requests to your server’s IP address. Otherwise, the client and server should be on the same device and send requests only to the local host / 127.0.0.1.

### `port`: int

- **Description**: The primary port number on which the server will listen.
- **Default**: 30000
- **Usage**: Defines the main port for incoming connections to the server.

### `additional_ports`: Optional[Union[List[int], int]]

- **Description**: Additional port(s) that can be used by the server.
- **Default**: None
- **Usage**: Can be either a single integer or a list of integers representing extra ports.
- **Note**: If a single integer is provided, it will be converted to a list containing that integer. If None is provided, it will be treated as an empty list.