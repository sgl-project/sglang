## Sampling Parameters of SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.

The `/generate` endpoint accepts the following arguments in the JSON format.

```python
class GenerateReqInput:
    text: Union[List[str], str]
    image_data: Optional[Union[List[str], str]] = None
    sampling_params: Union[List[Dict], Dict] = None
    rid: Optional[Union[List[str], str]] = None
    return_logprob: Optional[Union[List[bool], bool]] = None
    logprob_start_len: Optional[Union[List[int], int]] = None
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
for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        data = json.loads(chunk.decode())
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```
