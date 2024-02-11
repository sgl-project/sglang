## How to Support A New Model

To support a new model in SGLang, you only need to add a single file under [https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).

You can learn from existing model implementations and implement the new files for the new models. Most models are based on the transformer architecture so they are very similar.

Another good source is to learn from vLLM model implementations.
1. Compare these two files (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama2.py, https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). You can learn how to convert a model implementation from vLLM to SGLang. We need to replace PagedAttention with RadixAttention. The other parts are almost the same.
2. Convert models from vLLM to SGLang. https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
