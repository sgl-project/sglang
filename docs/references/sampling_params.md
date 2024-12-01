# Sampling Parameters in SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.
It is the low-level endpoint of the runtime.
If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API
](https://github.com/sgl-project/sglang?tab=readme-ov-file#openai-compatible-api).

The `/generate` endpoint accepts the following arguments in the JSON format.

```python
@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
```

The `sampling_params` follows this format

```python
# The maximum number of output tokens
max_new_tokens: int = 128,
# Stop when hitting any of the strings in this list.
stop: Optional[Union[str, List[str]]] = None,
# Stop when hitting any of the token_ids in this list. Could be useful when mixed with
# `min_new_tokens`.
stop_token_ids: Optional[List[int]] = [],
# Sampling temperature
temperature: float = 1.0,
# Top-p sampling
top_p: float = 1.0,
# Top-k sampling
top_k: int = -1,
# Min-p sampling
min_p: float = 0.0,
# Whether to ignore EOS token.
ignore_eos: bool = False,
# Whether to skip the special tokens during detokenization.
skip_special_tokens: bool = True,
# Whether to add spaces between special tokens during detokenization.
spaces_between_special_tokens: bool = True,
# Constrains the output to follow a given regular expression.
regex: Optional[str] = None,
# Do parallel sampling and return `n` outputs.
n: int = 1,
# Constrains the output to follow a given JSON schema.
# `regex` and `json_schema` cannot be set at the same time.
json_schema: Optional[str] = None,

## Penalties. See [Performance Implications on Penalties] section below for more informations.

# Float that penalizes new tokens based on their frequency in the generated text so far.
# Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to
# repeat tokens. Must be -2 <= value <= 2. Setting to 0 (default) will disable this penalty.
frequency_penalty: float = 0.0,
# Float that penalizes new tokens based on whether they appear in the generated text so far.
# Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat
# tokens. Must be -2 <= value <= 2. Setting to 0 (default) will disable this penalty.
presence_penalty: float = 0.0,
# Float that penalizes new tokens based on whether they appear in the prompt and the generated text
# so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to
# repeat tokens. Must be 0 <= value <= 2. Setting to 1 (default) will disable this penalty.
repetition_penalty: float = 1.0,
# Guides inference to generate at least this number of tokens by penalizing logits of tokenizer's
# EOS token and `stop_token_ids` to -inf, until the output token reaches given length.
# Note that any of the `stop` string can be generated before reaching `min_new_tokens`, as it is
# difficult to infer the correct token ID by given `stop` strings.
# Must be 0 <= value < max_new_tokens. Setting to 0 (default) will disable this penalty.
min_new_tokens: int = 0,
```

## Examples

### Normal
Launch a server
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Send a request
```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

### Streaming
Send a request and stream the output
```python
import requests, json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```

### Multi modal

Launch a server
```
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --chat-template chatml-llava
```

Download an image
```
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true
```

Send a request
```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

The `image_data` can be a file name, a URL, or a base64 encoded string. See also `python/sglang/srt/utils.py:load_image`.
Streaming is supported in a similar manner as [above](#streaming).

### Structured decoding (JSON, Regex)
You can specify a JSON schema or a regular expression to constrain the model output. The model output will be guaranteed to follow the given constraints.

```python
import json
import requests

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
print(response.json())

# Regular expression
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print(response.json())
```
