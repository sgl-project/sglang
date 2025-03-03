# Supported Models

## Generative Models
- Llama / Llama 2 / Llama 3 / Llama 3.1 / Llama 3.2
- Mistral / Mixtral / Mistral NeMo / Mistral Small 3
- Gemma / Gemma 2
- Qwen / Qwen 2 / Qwen 2 MoE / Qwen 2 VL / Qwen 2.5 VL
- DeepSeek / DeepSeek 2 / [DeepSeek 3](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3)
- OLMoE
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --port=30000 --chat-template=chatml-llava`
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-72b-ov --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server.py)
- LLaVA 1.5 / 1.6 / NeXT
  - `python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000 --tp-size=1 --chat-template=llava_llama_3`
  - `python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server.py)
- Yi-VL
- StableLM
- Command-R
- DBRX
- Grok
- ChatGLM
- InternLM 2
- Exaone 3
- BaiChuan2
- MiniCPM / MiniCPM 3 / MiniCPMV
- XVERSE / XVERSE MoE
- SmolLM
- GLM-4
- Phi-3 / Phi-4
- Phi-3-Small
- IBM Granite 3

## Embedding Models

- LlamaEmbeddingModel
- Mistral embedding models
- Qwen embedding models
  - `python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding`

## Reward Models

- LlamaForSequenceClassification
  - `python -m sglang.launch_server --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --is-embedding`
- Gemma2ForSequenceClassification
  - `python -m sglang.launch_server --model-path Skywork/Skywork-Reward-Gemma-2-27B-v0.2 --is-embedding`
- InternLM2ForRewardModel
  - `python -m sglang.launch_server --model-path internlm/internlm2-7b-reward --is-embedding --trust-remote-code`
- Qwen2ForRewardModel
  - `python -m sglang.launch_server --model-path Qwen/Qwen2.5-Math-RM-72B --is-embedding --trust-remote-code --tp-size=4`
## How to Support a New Language Model

To support a new model in SGLang, you only need to add a single file under [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).
You can learn from existing model implementations and create new files for the new models.
For most models, you should be able to find a similar model to start with (e.g., starting from Llama).

## How to Support a New vLM

To support a new vision-language model (vLM) in SGLang, there are several key components in addition to the standard
LLM.

1. **Register your new model as multimodal**: Extend `is_multimodal_model` in [
   `model_config.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py) to
   return True for your model.
2. **Process Images**: Create a new `ImageProcessor` class that inherits from `BaseImageProcessor` and register this
   processor as your model's dedicated processor. See [
   `image_processor.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/image_processor.py)
   for more details.
3. **Handle Image Tokens**: Implement a `pad_input_ids` function for your new model, in which image tokens in the prompt
   should be expanded and replaced with image-hashes, so that SGLang can recognize different images for
   `RadixAttention`.
4. Replace Multi-headed `Attention` of ViT with SGLang's `VisionAttention`.

You can refer [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) or other
vLMs. These models demonstrate how to properly handle both visual and textual inputs.

You should test the new vLM locally against hf models. See [`mmmu`](https://github.com/sgl-project/sglang/tree/main/benchmark/mmmu) for an example.

### Test the correctness

#### Interactive debugging
For interactive debugging, you can compare the outputs of huggingface/transformers and SGLang.
The following two commands should give the same text output and very similar prefill logits.

- Get the reference output by `python3 scripts/playground/reference_hf.py --model [new model]`
- Get the SGLang output by `python3 -m sglang.bench_one_batch --correct --model [new model]`

#### Add the model to the test suite
To make sure the new model is well maintained in the future, it is better to add it to the test suite.
You can add it to the `ALL_OTHER_MODELS` list in the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py) and run the following command to test it.

For example, if the model is Qwen/Qwen2-1.5B
```
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### Port a model from vLLM to SGLang
Another valuable resource is the [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models). vLLM has extensive coverage of models, and SGLang reuses vLLM's interface and some layers to implement the models. This similarity makes it easy to port many models from vLLM to SGLang.

To port a model from vLLM to SGLang, you can compare these two files [SGLang Llama Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py) and [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). This comparison will help you understand how to convert a model implementation from vLLM to SGLang. The major difference is the replacement of Attention with RadixAttention. The other parts are almost identical. Specifically,
  - Replace vllm's `Attention` with `RadixAttention`. Note that you need to pass `layer_id` all the way to `RadixAttention`.
  - Replace vllm's `LogitsProcessor` with SGLang's `LogitsProcessor`.
  - Replace Multi-headed `Attention` of ViT with SGLang's `VisionAttention`.
  - Replace other vLLM layers with SGLang layers (e.g., `RMSNorm`, `SiluAndMul`).
  - Remove `Sample`.
  - Change `forward()` functions, and add `forward_batch`.
  - Add `EntryClass` at the end.
  - Please ensure the new implementation uses **only SGLang components and does not rely on any vLLM components**.

### Registering an external model implementation

In addition to the methods described above, you can also register your new model with the `ModelRegistry` before launching the server. This approach is useful if you want to integrate your model without needing to modify the source code.

Here is how you can do it:

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# for a single model, you can add it to the registry
ModelRegistry.models[model_name] = model_class

# for multiple models, you can imitate the import_model_classes() function in sglang/srt/models/registry.py
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

launch_server(server_args)
```
