# quantization compressed_tensors module

To support compressed_tensors format quantization models, we adapted https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors into SGLang.

Several formats are supported today, for both linear and MoE layers -- fp8 and
int8 w8a8, w8a16 fp8, int4 `pack-quantized`, and `nvfp4-pack-quantized` (native
W4A4 on Blackwell, weight-only FP4 Marlin below it). See `schemes/` for the full
set. For RTN or other compressed-tensors exports with per-expert `weight_packed`
tensors, set `expert_layout: split` at the top level of `quantization_config`.
It is not necessary to set an AutoRound provider for this layout. Split-expert
`mxfp4-pack-quantized` MoE uses the following backends:

| Device | Expert execution | Selection |
| --- | --- | --- |
| NVIDIA SM100 (e.g. B200) | DeepGEMM MegaMoE, native MXFP4 W4A4 | `--moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe` on models with a MegaMoE forward |
| NVIDIA SM80+ (including A100 and H200) | FP4 Marlin, MXFP4 W4A16 | Default split-expert CUDA path |
| Intel XPU | Packed grouped GEMM, MXFP4 W4A16 | Default split-expert XPU path |

MegaMoE quantizes activations dynamically in groups of 32. It accepts MXFP4
weight-only metadata or dynamic MXFP4 activation metadata and requires gated
SiLU experts. The W4A16 paths keep BF16 activations even when the checkpoint
declares FP4 activations; the XPU kernel consumes the original E8M0 scale
bytes. These paths apply to routed experts, not to the entire model.
`mxfp8-quantized` dense layers
use the block-scaled GEMM on supported XPU devices and otherwise dequantize
their checkpoint weights to the model dtype for a dense linear on CUDA and
XPU. AutoRound `auto_round:llm_compressor` exports can use either automatic
compressed-tensors detection or `--quantization auto-round` for MXFP.

If you have requirements for other formats, you can submit an issue through this
[link](https://github.com/sgl-project/sglang/issues).
