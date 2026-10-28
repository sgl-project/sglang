# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 checkpoint names and layouts of the Quark and llm-compressor exports, shared by
the text and the multimodal model's loaders."""

from typing import Optional, Tuple

import torch


def quark_shared_experts_fusion_disable_reason(quant_config) -> Optional[str]:
    """Why a Quark checkpoint cannot fuse its shared expert into the routed MoE, or None."""
    if quant_config is None or quant_config.get_name() != "quark":
        return None
    # online MXFP4 requantization keeps the shared experts in the source's FP8 / MXFP8
    if quant_config.excluded_fp8_config is not None:
        return (
            "Shared experts stay in the checkpoint's FP8 precision while "
            "routed experts are requantized to MXFP4."
        )
    # a pre-quantized checkpoint may give its shared experts their own layer_quant_config
    layer_quant_config = quant_config.quant_config.get("layer_quant_config")
    for name, spec in (layer_quant_config or {}).items():
        if "shared_experts" not in name or not isinstance(spec, dict):
            continue
        weight_spec = spec.get("weight") or {}
        if weight_spec.get("dtype") != "fp4":
            return (
                f"Shared experts are serialized as {weight_spec.get('dtype')} "
                "while routed experts are MXFP4."
            )
    return None


def mxfp8_linear_scale_alias(name: str, params_dict: dict) -> str:
    """Newer Quark exports name the MXFP8 [1, 32] e8m0 scale of a linear layer
    `weight_scale`; Fp8LinearMethod registers it as `weight_scale_inv`. Only an
    unmatched name is redirected, so a `weight_scale` that is a real parameter
    (per-tensor FP8, NVFP4 experts) keeps its name."""
    if (
        name.endswith(".weight_scale")
        and name not in params_dict
        and name + "_inv" in params_dict
    ):
        return name + "_inv"
    return name


def remap_llm_compressor_nvfp4_expert(
    name: str, loaded_weight: torch.Tensor
) -> Tuple[str, torch.Tensor]:
    """Map llm-compressor `nvfp4-pack-quantized` expert tensors onto the
    ModelOpt names the NVFP4 MoE loaders expect.

    llm-compressor stores the per-tensor scale as `weight_global_scale`, the
    multiplier applied *before* the fp8 per-16 block scale was computed, so
    the dequantized weight is ``fp4 * weight_scale / weight_global_scale``.
    ModelOpt's `weight_scale_2` multiplies instead, hence the reciprocal.
    `input_global_scale` is only meaningful for static NVFP4 activations and
    is parked on the (discarded) `input_scale` slot.
    """
    if name.endswith(".weight_packed"):
        return name[: -len(".weight_packed")] + ".weight", loaded_weight
    if name.endswith(".weight_global_scale"):
        return (
            name[: -len(".weight_global_scale")] + ".weight_scale_2",
            1.0 / loaded_weight.to(torch.float32),
        )
    if name.endswith(".input_global_scale"):
        return name[: -len(".input_global_scale")] + ".input_scale", loaded_weight
    return name, loaded_weight
