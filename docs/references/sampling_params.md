# Sampling Parameters in SGLang Runtime
This doc describes the sampling parameters of the SGLang Runtime.
It is the low-level endpoint of the runtime.
If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API](https://docs.sglang.ai/backend/openai_api_completions.html).

## `/generate` Endpoint

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

## Sampling params

### Basic Sampling Parameters

* `max_new_tokens`: The maximum output length measured in tokens.
* `stop`: One or multiple [stop words](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#let_the_model_know_when_to_stop). Generation will stop if one of these words is sampled.
* `stop_token_ids`: Provide stop words in form of token ids. Generation will stop if one of these token ids is sampled.
* `temperature`: [Temperature](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) when sampling the next token. `temperature = 0` corresponds to greedy sampling, higher temperature leads to more diversity.
* `top_p`: [Top-p](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) selects tokens from the smallest sorted set whose cumulative probability exceeds `top_p`. When `top_p = 1`, this reduces to unrestricted sampling from all tokens.
* `top_k`: [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) randomly selects from the `k` highest-probability tokens.
* `min_p`: [Min-p](https://github.com/huggingface/transformers/issues/27670) samples from tokens with probability larger than `min_p * highest_token_probability`.

### Penalizers

To use penalizers you will need to `--disable-overlap`. Please note that this might degrade performance.

* `frequency_penalty`: Penalizes tokens based on their frequency in generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of penalization grows linearly with each appearance of a token.
* `presence_penalty`: Penalizes tokens if they appeared in the generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of the penalization is constant if a token occured.
* `repetition_penalty`: Penalizes tokens if they appeared in prompt or generation so far. Must be between `0` and `2` where numbers smaller than `1` encourage repeatment of tokens and numbers larger than `2` encourages sampling of new tokens. The penalization scales multiplicatively.
* `min_new_tokens`: Forces the model to generate at least `min_new_tokens` until a stop word or EOS token is sampled. Note that this might lead to unintended behavior for example if the distribution is highly skewed towards these tokens.

### TODO
* `spaces_between_special_tokens`: Whether or not to add spaces between special tokens during detokenization.
* `n`: `TODO`

### Constrained decoding

Please refer to our dedicated guide on [constrained decoding](https://docs.sglang.ai/backend/structured_outputs.html#Native-API-and-SGLang-Runtime-(SRT)) for the following parameters.

* `json_schema`
* `regex`
* `ebnf`

### Other options
* `no_stop_trim`: ?
* `ignore_eos`: Don't stop generation when EOS token is sampled.
* `skip_special_tokens`: Remove special tokens during decoding.
* `custom_params`: Used when employing `CustomLogitProcessor`. For usage see below.

## Examples

### Normal
*SV: This is described in  [native api doc](https://docs.sglang.ai/backend/native_api.html#).*

### Streaming
*SV: This section should be potentially moved to [native api doc](https://docs.sglang.ai/backend/native_api.html#) or have a section on its own.*

### Multi modal

*SV: This section should be potentially moved to [native api doc](https://docs.sglang.ai/backend/native_api.html#) or have a section on its own.*

### Structured Outputs (JSON, Regex, EBNF)
*SV: This section can be deleted, we have it in [structured outputs](https://docs.sglang.ai/backend/structured_outputs.html).*

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
