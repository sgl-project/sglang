# Sampling Parameters in SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.

The `/generate` endpoint accepts the following arguments in the JSON format.

```python
@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Union[List[Dict], Dict] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # The start location of the prompt for return_logprob.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # The number of top logprobs to return.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
```

The `sampling_params` follows this format

```python
# The maximum number of output tokens
max_new_tokens: int = 16,
# Stop when hitting any of the strings in this list.
stop: Optional[Union[str, List[str]]] = None,
# Sampling temperature
temperature: float = 1.0,
# Top-p sampling
top_p: float = 1.0,
# Top-k sampling
top_k: int = -1,
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
            "max_new_tokens": 256,
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
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000
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
        "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this picture ASSISTANT:",
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
