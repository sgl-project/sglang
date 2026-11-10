# quantization compressed_tensors module

To support compressed_tensors format quantization models, we adapted https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors into SGLang.

Several formats are supported today, for both linear and MoE layers -- fp8 and
int8 w8a8, w8a16 fp8, int4 `pack-quantized`, and `nvfp4-pack-quantized` (native
W4A4 on Blackwell, weight-only FP4 Marlin below it). See `schemes/` for the full
set. `mxfp4-pack-quantized` linears are not supported yet: the dense FP4 Marlin
kernel is only instantiated for group_size 16 and cannot decode E8M0 scales. If
you have requirements for other formats, you can submit an issue through this
[link](https://github.com/sgl-project/sglang/issues).
