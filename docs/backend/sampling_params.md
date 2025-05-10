# Sampling Parameters

This doc describes the sampling parameters of the SGLang Runtime. It is the low-level endpoint of the runtime.

If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API](./openai_api_completions.ipynb).

## `/generate` Endpoint

The `/generate` endpoint accepts the following parameters in JSON format. For detailed usage, see the [native API doc](./native_api.ipynb).

| Argument               | Type/Default                                            | Description                                                                                                                                    |
|------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| text                   | `Optional[Union[List[str], str]] = None`                | The input prompt. Can be a single prompt or a batch of prompts.                                                                                |
| input_ids              | `Optional[Union[List[List[int]], List[int]]] = None`    | Alternative to `text`. Specify the input as token IDs instead of text.                                                                         |
| sampling_params        | `Optional[Union[List[Dict], Dict]] = None`              | The sampling parameters as described in the sections below.                                                                                    |
| return_logprob         | `Optional[Union[List[bool], bool]] = None`              | Whether to return log probabilities for tokens.                                                                                                |
| logprob_start_len      | `Optional[Union[List[int], int]] = None`                | If returning log probabilities, specifies the start position in the prompt. Default is "-1", which returns logprobs only for output tokens.   |
| top_logprobs_num       | `Optional[Union[List[int], int]] = None`                | If returning log probabilities, specifies the number of top logprobs to return at each position.                                               |
| stream                 | `bool = False`                                          | Whether to stream the output.                                                                                                                  |
| lora_path              | `Optional[Union[List[Optional[str]], Optional[str]]] = None`| Path to LoRA weights.                                                                                                                          |
| custom_logit_processor | `Optional[Union[List[Optional[str]], str]] = None`      | Custom logit processor for advanced sampling control. For usage see below.                                                                     |
| return_hidden_states   | `bool = False`                                          | Whether to return hidden states of the model. Note that each time it changes, the CUDA graph will be recaptured, which might lead to a performance hit. See the [examples](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states) for more information. |

## Sampling parameters

### Core parameters

| Argument        | Type/Default                                 | Description                                                                                                                                    |
|-----------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| max_new_tokens  | `int = 128`                                  | The maximum output length measured in tokens.                                                                                                  |
| stop            | `Optional[Union[str, List[str]]] = None`     | One or multiple [stop words](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop). Generation will stop if one of these words is sampled. |
| stop_token_ids  | `Optional[List[int]] = None`                 | Provide stop words in the form of token IDs. Generation will stop if one of these token IDs is sampled.                                        |
| temperature     | `float = 1.0`                                | [Temperature](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature) when sampling the next token. `temperature = 0` corresponds to greedy sampling, a higher temperature leads to more diversity. |
| top_p           | `float = 1.0`                                | [Top-p](https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p) selects tokens from the smallest sorted set whose cumulative probability exceeds `top_p`. When `top_p = 1`, this reduces to unrestricted sampling from all tokens. |
| top_k           | `int = -1`                                   | [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) randomly selects from the `k` highest-probability tokens. |
| min_p           | `float = 0.0`                                | [Min-p](https://github.com/huggingface/transformers/issues/27670) samples from tokens with probability larger than `min_p * highest_token_probability`. |

### Penalizers

| Argument           | Type/Default           | Description                                                                                                                                    |
|--------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| frequency_penalty  | `float = 0.0`          | Penalizes tokens based on their frequency in generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of penalization grows linearly with each appearance of a token. |
| presence_penalty   | `float = 0.0`          | Penalizes tokens if they appeared in the generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of the penalization is constant if a token occured. |
| min_new_tokens     | `int = 0`              | Forces the model to generate at least `min_new_tokens` until a stop word or EOS token is sampled. Note that this might lead to unintended behavior, for example, if the distribution is highly skewed towards these tokens. |

### Constrained decoding

Please refer to our dedicated guide on [constrained decoding](./structured_outputs.ipynb) for the following parameters.

| Argument     | Type/Default                    | Description                                                                                                                                    |
|--------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| json_schema  | `Optional[str] = None`          | JSON schema for structured outputs.                                                                                                            |
| regex        | `Optional[str] = None`          | Regex for structured outputs.                                                                                                                  |
| ebnf         | `Optional[str] = None`          | EBNF for structured outputs.                                                                                                                   |

### Other options

| Argument                      | Type/Default                    | Description                                                                                                                                    |
|-------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| n                             | `int = 1`                       | Specifies the number of output sequences to generate per request. (Generating multiple outputs in one request (n > 1) is discouraged; repeating the same prompts several times offers better control and efficiency.) |
| spaces_between_special_tokens | `bool = True`                   | Whether or not to add spaces between special tokens during detokenization.                                                                     |
| no_stop_trim                  | `bool = False`                  | Don't trim stop words or EOS token from the generated text.                                                                                    |
| continue_final_message        | `bool = False`                  | When enabled, the final assistant message is removed and its content is used as a prefill so that the model continues that message instead of starting a new turn. See [openai_chat_with_response_prefill.py](https://github.com/sgl-project/sglang/blob/main/examples/runtime/openai_chat_with_response_prefill.py) for examples. |
| ignore_eos                    | `bool = False`                  | Don't stop generation when EOS token is sampled.                                                                                               |
| skip_special_tokens           | `bool = True`                   | Remove special tokens during decoding.                                                                                                         |
| custom_params                 | `Optional[List[Optional[Dict[str, Any]]]] = None` | Used when employing `CustomLogitProcessor`. For usage, see below.                                                                              |
| thinking_budget               | `Optional[int] = None`          | The maximum number of reasoning tokens that can be generated for a request. |

## Examples

### Normal

Launch a server:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Send a request:

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

Detailed example in [send request](./send_request.ipynb).

### Streaming

Send a request and stream the output:

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

Detailed example in [openai compatible api](https://docs.sglang.ai/backend/openai_api_completions.html#id2).

### Multimodal

Launch a server:

```bash
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --chat-template chatml-llava
```

Download an image:

```bash
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true
```

Send a request:

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

Detailed example in [openai api vision](./openai_api_vision.ipynb).

### Structured Outputs (JSON, Regex, EBNF)

You can specify a JSON schema, regular expression or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports two grammar backends:

- [Outlines](https://github.com/dottxt-ai/outlines) (default): Supports JSON schema and regular expression constraints.
- [XGrammar](https://github.com/mlc-ai/xgrammar): Supports JSON schema, regular expression, and EBNF constraints.
  - XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md).

Initialize the XGrammar backend using `--grammar-backend xgrammar` flag:

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

Detailed example in [structured outputs](./structured_outputs.ipynb).

### Custom logit processor

Launch a server with `--enable-custom-logit-processor` flag on.

```bash
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

Send a request:

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

### Thinking Budget

Launch a server with `--reasoning-parser`.

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-8B --reasoning-parser qwen3
```

Send a request:

```python
import requests
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "9.11 and 9.8, which is greater?",
        "sampling_params": {
            "temperature": 0.3,
            "max_new_tokens": 256,
            "thinking_budget": 20,
        },
    },
)
print(response.json())
```
