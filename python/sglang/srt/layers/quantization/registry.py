# SPDX-License-Identifier: Apache-2.0
"""Declarative registry of quantization methods.

Every entry is a `"module.path:ClassName"` spec, resolved on first use by
`sglang.srt.layers.quantization.__init__`. This module deliberately imports
nothing: the tables can be read to answer "which methods exist" without
pulling in torch or any config implementation.

Adding a method means adding one line to the table below. Adding a new
*platform condition* additionally means teaching `_platform_conditions()` in
`__init__.py` about it, which is checked.

A registry key is not the same thing as a `--quantization` value.
`server_args.QUANTIZATION_CHOICES` is deliberately a *subset*: methods that
require an already-quantized checkpoint are reachable only through the
`quant_method` in that checkpoint's config, never from the CLI.
`blockwise_int8` is one -- `BlockInt8LinearMethod` asserts both a
`weight_block_size` and an int8-serialized checkpoint, and the former may only
be set when the latter is true, so `--quantization blockwise_int8` could not
work. Every CLI choice must appear here, but not the reverse.
"""

from __future__ import annotations

# Available on every platform. Iteration order is load-bearing: the
# `override_quantization_method` loop in `ModelConfig._verify_quantization`
# takes the first checkpoint match, so entries must stay in this order.
QUANTIZATION_METHOD_SPECS: dict[str, str] = {
    "fp8": "sglang.srt.layers.quantization.fp8:Fp8Config",  # MOE + linear online quantization
    "mxfp8": "sglang.srt.layers.quantization.fp8:Fp8Config",  # MOE + linear online quantization
    "blockwise_int8": "sglang.srt.layers.quantization.blockwise_int8:BlockInt8Config",
    # Modelopt has some online quantization support through ModelOptModelLoader.
    "modelopt": "sglang.srt.layers.quantization.modelopt_quant:ModelOptFp8Config",  # auto-detect, defaults to FP8
    "modelopt_fp8": "sglang.srt.layers.quantization.modelopt_quant:ModelOptFp8Config",
    "modelopt_fp4": "sglang.srt.layers.quantization.modelopt_quant:ModelOptFp4Config",
    "nvfp4_online": "sglang.srt.layers.quantization.nvfp4_online:NvFp4OnlineConfig",
    "modelopt_mixed": "sglang.srt.layers.quantization.modelopt_quant:ModelOptMixedPrecisionConfig",
    # Both are documented in quantization.md and support the compressed-tensors
    # `quant_method`.
    "w8a8_int8": "sglang.srt.layers.quantization.w8a8_int8:W8A8Int8Config",
    "w8a8_fp8": "sglang.srt.layers.quantization.w8a8_fp8:W8A8Fp8Config",
    "awq": "sglang.srt.layers.quantization.awq:AWQConfig",
    "awq_marlin": "sglang.srt.layers.quantization.awq:AWQMarlinConfig",
    "bitsandbytes": "sglang.srt.layers.quantization.bitsandbytes:BitsAndBytesConfig",
    "gguf": "sglang.srt.layers.quantization.gguf:GGUFConfig",
    "gptq": "sglang.srt.layers.quantization.gptq:GPTQConfig",
    "gptq_marlin": "sglang.srt.layers.quantization.gptq:GPTQMarlinConfig",
    "moe_wna16": "sglang.srt.layers.quantization.moe_wna16:MoeWNA16Config",  # custom loading logic for gptq/awq checkpoints (likely untested/unused)
    "compressed-tensors": "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors:CompressedTensorsConfig",  # for Ktransformers
    "w4afp8": "sglang.srt.layers.quantization.w4afp8:W4AFp8Config",
    "petit_nvfp4": "sglang.srt.layers.quantization.petit:PetitNvFp4Config",
    "quark": "sglang.srt.layers.quantization.quark.quark:QuarkConfig",  # AMD Quark quantizer (FP8 / MXFP4 / Int4FP8 etc.)
    "quark_mxfp4": "sglang.srt.layers.quantization.quark.quark:QuarkConfig",  # online MOE + linear quantization (incl. NVFP4 -> MXFP4 requantization)
    "auto-round": "sglang.srt.layers.quantization.auto_round:AutoRoundConfig",
    "auto-round-int8": "sglang.srt.layers.quantization.w8a8_int8:W8A8Int8Config",
    "modelslim": "sglang.srt.layers.quantization.modelslim.modelslim:ModelSlimConfig",  # for NPU
    "quark_int4fp8_moe": "sglang.srt.layers.quantization.quark_int4fp8_moe:QuarkInt4Fp8Config",
    "humming": "sglang.srt.layers.quantization.humming:HummingConfig",
    "mxfp_w4a8": "sglang.srt.layers.quantization.npu_mxfp4:Mxfp4W4A8Config",  # NPU W4A8 (MXFP4 weights + MXFP8 activations)
}

# Applied on top of the base table, in this order, when the named condition
# holds. Each is a plain dict update, so an existing key keeps its position and
# a new key appends -- matching what the imperative `.update()` calls did.
# Condition names are evaluated in `__init__.py`; keeping them as strings is
# what keeps this module import-free.
PLATFORM_OVERRIDE_SPECS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        # cpu/cuda/hip-gfx95/xpu. On XPU the OCP-MoE `Mxfp4Config` path is served
        # by the sgl-kernel-xpu grouped GEMM, which consumes the packed e2m1 +
        # ue8m0 g32 checkpoint layout directly. Backends without that kernel keep
        # the existing "unknown quantization method" error rather than falling
        # through to a bf16 upcast.
        "mxfp4_capable",
        {"mxfp4": "sglang.srt.layers.quantization.mxfp4:Mxfp4Config"},
    ),
    (
        "npu",
        {
            "gptq": "sglang.srt.layers.quantization.gptq:GPTQAscendConfig",
            # On NPU, `mxfp4` means single-level W4A4 MXFP4 for dense LLM. The
            # upstream `Mxfp4Config` OCP-MoE path is registered only under
            # `mxfp4_capable` (cpu/cuda/hip), so there is no collision here.
            "mxfp4": "sglang.srt.layers.quantization.npu_mxfp4_w4a4:Mxfp4W4A4Config",
        },
    ),
    (
        "xpu",
        {
            "gptq": "sglang.srt.layers.quantization.gptq:GPTQXPUConfig",
            "awq": "sglang.srt.layers.quantization.awq:AWQXPUConfig",
        },
    ),
    (
        "mps",
        {
            # Apple Silicon MLX backend -- on-the-fly quantization of fp16
            # weights at load time via mlx.nn.quantize. Only takes effect when
            # SGLANG_USE_MLX=1.
            "mlx_q4": "sglang.srt.layers.quantization.mlx:MlxQuantizationConfig",  # 4 bits, group_size=64 (mlx-community default)
            "mlx_q8": "sglang.srt.layers.quantization.mlx:MlxQuantizationConfig",  # 8 bits, group_size=64
        },
    ),
)

# The subset supported on CPU with AMX, with CPU-specific config classes where
# they differ. Consulted instead of the tables above once AMX is detected.
CPU_SUPPORTED_METHOD_SPECS: dict[str, str] = {
    "fp8": "sglang.srt.layers.quantization.fp8:Fp8Config",
    "w8a8_int8": "sglang.srt.layers.quantization.w8a8_int8:W8A8Int8Config",
    "compressed-tensors": "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors:CompressedTensorsConfig",
    "awq": "sglang.srt.layers.quantization.awq:AWQCPUConfig",
    "gptq": "sglang.srt.layers.quantization.gptq:CPUGPTQConfig",
    "mxfp4": "sglang.srt.layers.quantization.mxfp4:Mxfp4Config",
    "auto-round": "sglang.srt.layers.quantization.auto_round:AutoRoundConfig",
}


def all_method_names() -> list[str]:
    """Every method name registered on any platform, in table order.

    Spans platforms so that a caller can ask what the registry knows about
    without being told only what today's hardware activates.
    """
    names = list(QUANTIZATION_METHOD_SPECS)
    for _, overrides in PLATFORM_OVERRIDE_SPECS:
        names.extend(name for name in overrides if name not in names)
    names.extend(name for name in CPU_SUPPORTED_METHOD_SPECS if name not in names)
    return names


def all_config_class_specs() -> dict[str, str]:
    """Config class name -> spec, for every class any table can resolve to.

    Backs the package-level `__getattr__`, so `from ...quantization import
    Fp8Config` keeps working without importing all 28 config modules.
    """
    specs: dict[str, str] = {}
    tables = [QUANTIZATION_METHOD_SPECS, CPU_SUPPORTED_METHOD_SPECS]
    tables.extend(overrides for _, overrides in PLATFORM_OVERRIDE_SPECS)
    for table in tables:
        for spec in table.values():
            specs.setdefault(spec.rpartition(":")[2], spec)
    return specs
