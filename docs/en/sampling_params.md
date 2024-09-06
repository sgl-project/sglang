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

## Performance Implications on Penalties

While you can apply penalties by supplying relevant `sampling_params`, this comes with some drawbacks.

These drawbacks will be applied to every single requests in the same batch, as penalizers also applies in batch.

### Latency

While we try to compute penalty algorithms through CUDA, it is still additional computation on top of the basic sampling logic. For detailed overhead, we recommend you to run your own benchmarks, but you can find samples below to get a glimpse.

### Memory

Since we compute penalty algorithms through CUDA, the logic stores relevant parameters on GPU. This is usually in a scale of `vocab_size` multiplied by `running_requests`.

You can run your own benchmark with desired parameters on your own hardware to make sure it's not OOMing before using.

Tuning `--mem-fraction-static` and/or `--max-running-requests` will help. See [here](hyperparameter_tuning.md#minor-tune---max-prefill-tokens---mem-fraction-static---max-running-requests) for more information.

### Benchmarks

All the benchmarks below were ran on NVIDIA H100 SXM5.

<details>

#### Baseline

Measured at [dc9d06d886151707f97d0b78095df9de262fd3c9](https://github.com/sgl-project/sglang/commit/dc9d06d886151707f97d0b78095df9de262fd3c9).

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     3000
Benchmark duration (s):                  66.11
Total input tokens:                      378633
Total generated tokens:                  775651
Total generated tokens (retokenized):    775118
Request throughput (req/s):              45.38
Input token throughput (tok/s):          5727.04
Output token throughput (tok/s):         11732.16
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   40881.94
Median E2E Latency (ms):                 43967.10
---------------Time to First Token----------------
Mean TTFT (ms):                          19884.75
Median TTFT (ms):                        14226.56
P99 TTFT (ms):                           47738.97
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          91.96
Median TPOT (ms):                        90.11
P99 TPOT (ms):                           308.54
---------------Inter-token Latency----------------
Mean ITL (ms):                           174.54
Median ITL (ms):                         58.56
P99 ITL (ms):                            440.18
==================================================
```

#### All Together

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512 --extra-request-body '{
  "frequency_penalty": 1.1,
  "presence_penalty": 1.1,
  "repetition_penalty": 0.1,
  "min_new_tokens": 5
}'

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     3000
Benchmark duration (s):                  78.35
Total input tokens:                      378633
Total generated tokens:                  775651
Total generated tokens (retokenized):    774756
Request throughput (req/s):              38.29
Input token throughput (tok/s):          4832.86
Output token throughput (tok/s):         9900.39
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   49017.68
Median E2E Latency (ms):                 52825.70
---------------Time to First Token----------------
Mean TTFT (ms):                          23892.60
Median TTFT (ms):                        18895.47
P99 TTFT (ms):                           57426.01
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          114.54
Median TPOT (ms):                        107.27
P99 TPOT (ms):                           293.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           205.68
Median ITL (ms):                         73.97
P99 ITL (ms):                            453.86
==================================================
```

#### Frequency Penalty

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512 --extra-request-body '{
    "frequency_penalty": 1.1
}'

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     3000
Benchmark duration (s):                  72.72
Total input tokens:                      378633
Total generated tokens:                  775651
Total generated tokens (retokenized):    774955
Request throughput (req/s):              41.26
Input token throughput (tok/s):          5206.84
Output token throughput (tok/s):         10666.51
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   45445.56
Median E2E Latency (ms):                 48960.39
---------------Time to First Token----------------
Mean TTFT (ms):                          22363.16
Median TTFT (ms):                        17125.02
P99 TTFT (ms):                           52920.95
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          104.71
Median TPOT (ms):                        98.30
P99 TPOT (ms):                           268.06
---------------Inter-token Latency----------------
Mean ITL (ms):                           191.60
Median ITL (ms):                         67.83
P99 ITL (ms):                            455.46
==================================================
```

#### Presence Penalty

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512 --extra-request-body '{
    "presence_penalty": 1.1
}'

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     3000
Benchmark duration (s):                  72.04
Total input tokens:                      378633
Total generated tokens:                  775651
Total generated tokens (retokenized):    775210
Request throughput (req/s):              41.64
Input token throughput (tok/s):          5255.98
Output token throughput (tok/s):         10767.18
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   44926.61
Median E2E Latency (ms):                 48302.88
---------------Time to First Token----------------
Mean TTFT (ms):                          22095.39
Median TTFT (ms):                        16740.93
P99 TTFT (ms):                           52554.03
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          103.54
Median TPOT (ms):                        97.37
P99 TPOT (ms):                           271.86
---------------Inter-token Latency----------------
Mean ITL (ms):                           189.86
Median ITL (ms):                         68.45
P99 ITL (ms):                            447.11
==================================================
```

#### Repetition Penalty

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512 --extra-request-body '{
    "repetition_penalty": 0.1
}'

============ Serving Benchmark Result ============
Backend: sglang
Traffic request rate: inf
Successful requests: 3000
Benchmark duration (s): 74.54
Total input tokens: 378633
Total generated tokens: 775651
Total generated tokens (retokenized): 766008
Request throughput (req/s): 40.24
Input token throughput (tok/s): 5079.36
Output token throughput (tok/s): 10405.35
----------------End-to-End Latency----------------
Mean E2E Latency (ms): 46530.38
Median E2E Latency (ms): 50302.65
---------------Time to First Token----------------
Mean TTFT (ms): 22603.47
Median TTFT (ms): 17167.08
P99 TTFT (ms): 54497.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms): 117.59
Median TPOT (ms): 101.79
P99 TPOT (ms): 320.04
---------------Inter-token Latency----------------
Mean ITL (ms): 195.26
Median ITL (ms): 69.51
P99 ITL (ms): 433.86
==================================================
```

#### Min New Tokens

The min new tokens penalizer computes until generation process reaches given `min_new_tokens`.

Dislike other penalizers, setting this to higher value will have more latency implications.

```
$ python3 -m sglang.bench_serving --backend sglang --port 8413 --dataset-name random --num-prompts 3000 --random-input 256 --random-output 512 --extra-request-body '{
    "min_new_tokens": 5
}'

============ Serving Benchmark Result ============
Backend: sglang
Traffic request rate: inf
Successful requests: 3000
Benchmark duration (s): 66.94
Total input tokens: 378633
Total generated tokens: 775651
Total generated tokens (retokenized): 775220
Request throughput (req/s): 44.81
Input token throughput (tok/s): 5656.13
Output token throughput (tok/s): 11586.90
----------------End-to-End Latency----------------
Mean E2E Latency (ms): 41888.55
Median E2E Latency (ms): 45354.16
---------------Time to First Token----------------
Mean TTFT (ms): 20866.91
Median TTFT (ms): 16219.79
P99 TTFT (ms): 49263.91
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms): 97.05
Median TPOT (ms): 89.76
P99 TPOT (ms): 233.50
---------------Inter-token Latency----------------
Mean ITL (ms): 179.17
Median ITL (ms): 55.08
P99 ITL (ms): 409.12
==================================================
```

</details>
