## Sampling Parameters of SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.

The `/generate` endpoint accepts the following arguments in the JSON format.

```python
@dataclass
class GenerateReqInput:
    # The input prompt
    text: Union[List[str], str]
    # The image input
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params
    sampling_params: Union[List[Dict], Dict] = None
    # The request id
    rid: Optional[Union[List[str], str]] = None
    # Whether return logprobs of the prompts
    return_logprob: Optional[Union[List[bool], bool]] = None
    # The start location of the prompt for return_logprob
    logprob_start_len: Optional[Union[List[int], int]] = None
    # Whether to stream output
    stream: bool = False
```

The `sampling_params` follows this format

```python
class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 16,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        dtype: Optional[str] = None,
        regex: Optional[str] = None,
    ) -> None:
```

## Examples

### Normal
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

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

See [test_httpserver_llava.py](../test/srt/test_httpserver_llava.py).
