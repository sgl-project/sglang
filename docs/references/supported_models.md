# Supported Models

## Generative Models
- Llama / Llama 2 / Llama 3 / Llama 3.1 / Llama 3.2
- Mistral / Mixtral / Mistral NeMo
- Gemma / Gemma 2
- Qwen / Qwen 2 / Qwen 2 MoE / Qwen 2 VL
- DeepSeek / DeepSeek 2
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
- MiniCPM / MiniCPM 3
- XVERSE / XVERSE MoE
- SmolLM
- GLM-4
- Phi-3-Small

## Embedding Models

- LlamaEmbeddingModel
- Mistral embedding models
- QWen embedding models
  - `python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding`

## Reward Models

- LlamaForSequenceClassification
  - `python -m sglang.launch_server --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --is-embedding`
- Gemma2ForSequenceClassification
  - `python -m sglang.launch_server --model-path Skywork/Skywork-Reward-Gemma-2-27B-v0.2 --is-embedding`
- InternLM2ForRewardModel
  - `python -m sglang.launch_server --model-path internlm/internlm2-7b-reward --is-embedding --trust-remote-code`

## How to Support a New Model

To support a new model in SGLang, you only need to add a single file under [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).
You can learn from existing model implementations and create new files for the new models.
For most models, you should be able to find a similar model to start with (e.g., starting from Llama).

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
  - Replace other vLLM layers with SGLang layers (e.g., `RMSNorm`, `SiluAndMul`).
  - Remove `Sample`.
  - Change `forward()` functions, and add `forward_batch`.
  - Add `EntryClass` at the end.
