# Sampling Parameters in SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.
It is the low-level endpoint of the runtime.
If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API](../backend/openai_api_completions.ipynb).

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
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # LoRA related
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None
    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None
```

The `sampling_params` follows this format

```python
# The maximum number of output tokens
max_new_tokens: int = 128,
# Stop when hitting any of the strings in this list
stop: Optional[Union[str, List[str]]] = None,
# Stop when hitting any of the token_ids in this list
stop_token_ids: Optional[List[int]] = [],
# Sampling temperature
temperature: float = 1.0,
# Top-p sampling
top_p: float = 1.0,
# Top-k sampling
top_k: int = -1,
# Min-p sampling
min_p: float = 0.0,
# Whether to ignore EOS token
ignore_eos: bool = False,
# Whether to skip the special tokens during detokenization
skip_special_tokens: bool = True,
# Whether to add spaces between special tokens during detokenization
spaces_between_special_tokens: bool = True,
# Do parallel sampling and return `n` outputs.
n: int = 1,

## Structured Outputs
# Only one of the below three can be set for a request.

# Constrain the output to follow a given JSON schema.
json_schema: Optional[str] = None,
# Constrain the output to follow a given regular expression.
regex: Optional[str] = None,
# Constrain the output to follow a given EBNF grammar.
ebnf: Optional[str] = None,

## Penalties.

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


## Custom Parameters for Custom Logit Processor.
# A dictionary of custom parameters for the custom logit processor.
# The custom logit processor takes a list of dictionaries as input, where each
# dictionary is the custom parameters for one token in a batch of the input.
# See also python/sglang/srt/sampling/custom_logit_processor.py
custom_params: Optional[Dict[str, Any]] = None,
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

### Structured Outputs (JSON, Regex, EBNF)
You can specify a JSON schema, regular expression or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports two grammar backends:

- [Outlines](https://github.com/dottxt-ai/outlines) (default): Supports JSON schema and regular expression constraints.
- [XGrammar](https://github.com/mlc-ai/xgrammar): Supports JSON schema, regular expression, and EBNF constraints.
  - XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)

Initialize the XGrammar backend using `--grammar-backend xgrammar` flag
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --host 0.0.0.0 --grammar-backend [xgrammar|outlines] # xgrammar or outlines (default: outlines)
```

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w]+$"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

# JSON (works with both Outlines and XGrammar)
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

# Regular expression (Outlines backend only)
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

# EBNF (XGrammar backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
print(response.json())
```
### Custom Logit Processor
Launch a server with `--enable-custom-logit-processor` flag on.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --enable-custom-logit-processor
```

Define a custom logit processor that will always sample a specific token id.
```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            logits[i, :] = -float("inf")
            # Assign highest probability to the specified token
            logits[i, param_dict[key]] = 0.0
        return logits
```

Send a request
```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
print(response.json())
```
