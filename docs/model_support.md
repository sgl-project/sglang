## How to Support a New Model

To support a new model in SGLang, you only need to add a single file under [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).

You can learn from existing model implementations and create new files for the new models. Most models are based on the transformer architecture, making them very similar.

Another valuable resource is the vLLM model implementations. vLLM has extensive coverage of models, and SGLang has reused vLLM for most parts of the model implementations. This similarity makes it easy to port many models from vLLM to SGLang.

1. Compare these two files [SGLang LLaMA Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama2.py) and [vLLM LLaMA Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). This comparison will help you understand how to convert a model implementation from vLLM to SGLang. The major difference is the replacement of PagedAttention with RadixAttention. The other parts are almost identical.
2. Convert models from vLLM to SGLang by visiting the [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models).
