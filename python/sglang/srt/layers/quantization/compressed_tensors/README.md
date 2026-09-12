# quantization compressed_tensors module

To support compressed_tensors format quantization models, we adapted https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors into SGLang.


For practical purposes, we have only applied the compressed_tensors format of `w8a8_fp8`. If you have requirements for other formats, you can submit an issue through this [link](https://github.com/sgl-project/sglang/issues).

## ROCm W4A16 linear layers

Symmetric 4-bit group/channel checkpoints can use the optional ROCm vLLM GPTQ
operators (`gptq_gemm` with `use_v2_format`, and `gptq_shuffle`, as in vLLM 0.16).
Supported group sizes are 32, 64, 128, or channelwise; activation ordering is
unsupported, and each local weight partition must have K and N divisible by 128.
The kernel uses FP16 arithmetic. BF16 activations and scales are converted to
FP16, so models must stay within FP16's numerical range. Checkpoint scales that
overflow FP16 are rejected during loading. CUDA continues to use Marlin.
