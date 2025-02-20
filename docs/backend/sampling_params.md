# Sampling Parameters

This doc describes the sampling parameters of the SGLang Runtime.
It is the low-level endpoint of the runtime.
If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API](https://docs.sglang.ai/backend/openai_api_completions.html).

## `/generate` Endpoint

The `/generate` endpoint accepts the following parameters in JSON format. For in detail usage see the [native api doc](https://docs.sglang.ai/backend/native_api.html).

* `prompt`: The input prompt. Can be a single prompt or a batch of prompts.
* `input_ids`: Alternative to `text`. Specify the input as token IDs instead of text.
* `sampling_params`: The sampling parameters as described in the sections below.
* `return_logprob`: Whether to return log probabilities for tokens.
* `logprob_start_len`: If returning log probabilities, specifies the start position in the prompt. Default is "-1" which returns logprobs only for output tokens.
* `top_logprobs_num`: If returning log probabilities, specifies the number of top logprobs to return at each position.
* `stream`: Whether to stream the output.
* `lora_path`: Path to LoRA weights.
* `custom_logit_processor`: Custom logit processor for advanced sampling control. For usage see below.

## Sampling params

### Core Parameters

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

### Constrained decoding

Please refer to our dedicated guide on [constrained decoding](https://docs.sglang.ai/backend/structured_outputs.html#Native-API-and-SGLang-Runtime-(SRT)) for the following parameters.

* `json_schema`
* `regex`
* `ebnf`

### Other options

* `n`: Specifies the number of output sequences to generate per request. (Generating multiple outputs in one request (n > 1) is discouraged; separate requests offer better control and efficiency.)
* `spaces_between_special_tokens`: Whether or not to add spaces between special tokens during detokenization.
* `no_stop_trim`: Don't trim stop words or EOS token from the generated text.
* `ignore_eos`: Don't stop generation when EOS token is sampled.
* `skip_special_tokens`: Remove special tokens during decoding.
* `custom_params`: Used when employing `CustomLogitProcessor`. For usage see below.


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
