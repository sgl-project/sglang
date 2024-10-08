# How to Support a New Model

To support a new model in SGLang, you only need to add a single file under [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).
You can learn from existing model implementations and create new files for the new models.
For most models, you should be able to find a similar model to start with (e.g., starting from Llama).

## Test the correctness

### Interactive debugging
For interactive debugging, you can compare the outputs of huggingface/transformers and SGLang.
The following two commands should give the same text output and very similar prefill logits.

- Get the reference output by `python3 scripts/playground/reference_hf.py --model [new model]`
- Get the SGLang output by `python3 -m sglang.bench_latency --correct --model [new model]`

### Add the model to the test suite
To make sure the new model is well maintained in the future, it is better to add it to the test suite.
You can add it to the `ALL_OTHER_MODELS` list in the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py) and run the following command to test it.

For example, if the model is Qwen/Qwen2-1.5B
```
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

## Port a model from vLLM to SGLang
Another valuable resource is the [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models). vLLM has extensive coverage of models, and SGLang reuses vLLM's interface and some layers to implement the models. This similarity makes it easy to port many models from vLLM to SGLang.

To port a model from vLLM to SGLang, you can compare these two files [SGLang Llama Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py) and [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). This comparison will help you understand how to convert a model implementation from vLLM to SGLang. The major difference is the replacement of Attention with RadixAttention. The other parts are almost identical. Specifically,
  - Replace vllm's `Attention` with `RadixAttention`. Note that you need to pass `layer_id` all the way to `RadixAttention`.
  - Replace vllm's `LogitsProcessor` with SGLang's `LogitsProcessor`.
  - Replace other vLLM layers with SGLang layers (e.g., `RMSNorm`, `SiluAndMul`).
  - Remove `Sample`.
  - Change `forward()` functions, and add `forward_batch`.
  - Add `EntryClass` at the end.

