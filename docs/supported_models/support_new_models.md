# How to Support New Models

This document explains how to add support for new language models and multimodal large language models (MLLMs) in
SGLang. It also covers how to test new models and register external implementations.

## How to Support a New Language Model

To support a new model in SGLang, you only need to add a single file under
the [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). You can learn
from existing model implementations and create a new file for your model. For most models, you should be able to find a
similar model to start with (e.g., starting from Llama). Also refer how
to [port a Model from vLLM to SGLang](#port-a-model-from-vllm-to-sglang)

## How to Support a New Multimodal Large Language Model

To support a new multimodal large language model (MLLM) in SGLang, there are several key components in addition to the
standard LLM support:

1. **Register your new model as multimodal**:
   Extend `is_multimodal_model`
   in [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561)
   to return `True` for your model.

2. **Register a new chat-template**:
   Only when your default chat-template is unable to accept images as input: Register a new chat template in [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) and the corresponding matching function.

3. **Multimodal Data Processor**:
   Define a new `Processor` class that inherits from `BaseMultimodalProcessor` and register this processor as your
   model’s dedicated processor.
   See [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)
   for more details.

4. **Handle Multimodal Tokens**:
   Implement a `pad_input_ids` function for your new model. In this function, multimodal tokens in the prompt should be
   expanded (if necessary) and padded with multimodal-data-hashes so that SGLang can recognize different multimodal data
   with `RadixAttention`.

5. **Handle Image Feature Extraction**:
   Implement a `get_image_feature` function for your new model, which extracts image features from raw image data and converts them into the embeddings used by the language model.

6. **Adapt to Vision Attention**:
   Adapt the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.

You can refer to [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) or
other mllm implementations. These models demonstrate how to correctly handle both multimodal and textual inputs.

## Testing and Debugging

Please note all your testing and benchmarking results in PR description.

### Interactive Debugging

For interactive debugging, compare the outputs of Hugging Face/Transformers and SGLang. The following two commands
should give the same text output and very similar prefill logits:

- Get the reference output:
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
  ```
- Get the SGLang output:
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [new model]
  ```

### Add the Model to the Test Suite

To ensure the new model is well maintained, add it to the test suite by including it in the `ALL_OTHER_MODELS` list in
the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py)
file, test the new model on your local machine and report the results on demonstrative benchmarks (GSM8K, MMLU, MMMU,
MMMU-Pro, etc.) in your PR. \\
For VLMs, also include a test in `test_vision_openai_server_{x}.py` (e.g. [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py), [test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)).


This is an example command to run to test a new model on your local machine:

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### Benchmark

- **(Required) MMMU**: follow MMMU benchmark [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) to get SGLang vs. HF Transformer accuracy comparison. The accuracy score from SGLang run should not be much lower than that from HF Transformer run. Similarly, follow https://docs.sglang.io/developer_guide/benchmark_and_profiling.html to get performance comparison: TTFT and throughput must meet or exceed baselines (e.g., HF Transformer).
- **(Optional) Other evals**: If you ran other evals, please note the results in PR description.

## Port a Model from vLLM to SGLang

The [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) is a valuable
resource, as vLLM covers many models. SGLang reuses vLLM’s interface and some layers, making it easier to port models
from vLLM to SGLang.

To port a model from vLLM to SGLang:

- Compare these two files for guidance:
    - [SGLang Llama Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
    - [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- The major differences include:
    - **Replace vLLM’s `Attention` with `RadixAttention`** (ensure you pass `layer_id` to `RadixAttention`).
    - **Replace vLLM’s `LogitsProcessor` with SGLang’s `LogitsProcessor`.**
    - **Replace the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.**
    - **Replace other vLLM layers** (such as `RMSNorm`, `SiluAndMul`) with SGLang layers.
    - **Remove `Sample`.**
    - **Change the `forward()` functions** and add a `forward_batch()` method.
    - **Add `EntryClass`** at the end.
    - **Ensure that the new implementation uses only SGLang components** and does not rely on any vLLM components.

Note: make sure you add your new model to the supported models list in the supported models documentation.

## Registering an External Model Implementation

In addition to the methods above, you can register your new model with the `ModelRegistry` before launching the server.
This allows you to integrate your model without modifying the source code.

For example:

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```

## Example: Implementing and Serving a Llama Wrapper Model

Below is an introductory, step-by-step walkthrough on how to implement a new model end-to-end in SGLang and then run it via the [Offline Engine](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb).

### Implementing Our Model

To keep things simple, this new model will be a simple wrapper around [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and our goal will be just to bias the output logits for each `forward` call by taking the square root of each individual logit.

Let's start by defining our model in a file called `llama_wrapper.py`.
The first step is to import the necessary libraries from SRT, which is SGLang's internal backend.

```python
# In the file `llama_wrapper.py`

import torch
from transformers import LlamaConfig
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.models.llama import LlamaForCausalLM
```

Next, we declare a new `class` for our model and have it inherit from `LlamaForCausalLM`, which allows our model to access `LlamaForCausalLM`'s predefined modules and layers, such as `LlamaAttention` and `LlamaMLP`.
Note that almost all model implementations take in `config` and `quant_config` as arguments for their `__init__` method; `config` and `quant_config` are passed in via [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219).
Because we have inherited from `LlamaForCausalLM`, we can pass our parameters directly to its constructor, which will set the member variables for us.

```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

Now, we want to define the `forward` method, which is what will be called at inference time.
Note that the signature for `forward` is essentially the same for any model; you can take a look at the other models defined in the [`models` directory](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) for references.
To see where exactly `forward` is called in the SGLang runtime's internals, take a look at [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) and [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724) in the [`ModelRunner` class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

We now call the `__call__` method for `self.model` (which is a member variable that `LlamaForCausalLM` defines in its `__init__` method), which eventually calls `LlamaForCausalLM`'s `forward` method.
After that, we feed the `hidden_states` into our model's `LogitsProcessor` (again defined in `LlamaForCausalLM`).

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

After receiving the logits for the next token, we can finally perform our biasing step.

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```
Now, our `LlamaWrapper` model is created and ready to be served!

### Serving Our Model Via SGLang's Offline Engine

The next step of this walkthrough involves hosting our new model offline, so that it can be served locally and without an HTTP server.

First, create a new file called `run.py`.
Now, we must ensure that SGLang's `ModelRegistry` can find our model.
To do this, we first download the model's configuration and weights from Huggingface.

```python
# In the file `run.py`

import asyncio
from functools import lru_cache
from huggingface_hub import snapshot_download
from llama_wrapper import LlamaWrapper # Make sure to import our new model!
import sglang as sgl
from sglang.srt.models.registry import ModelRegistry

# Make sure to request access to this model on Huggingface, then export your
# `HF_TOKEN` to download the model snapshot
llama_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./llama_ckpt",
)
```

Now that we have our model on disk, we want to point it to `LlamaWrapper` by changing the `architectures` field in `./llama_ckpt/config.json` to be `LlamaWrapper`.
That way, when we pass in the path of our model checkpoint to SGLang, it will know that we want to use "LlamaWrapper" instead of "LlamaForCausalLM" as our model.

```python
{
  "architectures": [
   #  "LlamaForCausalLM"
    "LlamaWrapper"
  ],
  ...
}
```

However, if we don't link our `LlamaWrapper` class to the "LlamaWrapper" registry keyword, then SGLang won't be able to find our model.
Thus, to register our `LlamaWrapper`, we want to follow the steps in the above section titled "Registering an External Model Implementation".

```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

Lastly, when we create our `Engine`, we just pass in the path to the local model directory.
Then, our `LlamaWrapper` is ready to be served; for this walkthrough, we will use SGLang `Engine`'s non-streaming asynchronous generation endpoint.

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```

Now, when we call `python run.py`, we will get the outputs of our newly created model!


## Documentation
Add to table of supported models in [generative_models.md](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/generative_models.md) or [multimodal_language_models.md](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/multimodal_language_models.md)

---

By following these guidelines, you can add support for new language models and multimodal large language models in
SGLang and ensure they are thoroughly tested and easily integrated into the system.
