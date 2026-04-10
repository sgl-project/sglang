to guarantee the best accuracy along with efficiency, we should load tuned models from different paths.
    (as a contrary, loading only bf16 models and provides naive quantization is suboptimal)
    (tuned model weights should have been processed with techniques like GPTQ/AWQ)


check if 
    1. INT4 weights / INT8 weight can be found on huggingface for qwen3-30b-a3b, if so download
    2. sglang includes support for weite quantization with GPTQ/AWQ, if all provided
        we need a pipeline that produces these quantized weights for our model
