# SPDX-License-Identifier: Apache-2.0
"""Native original Qwen-Image, bf16 resident DiT, FA, eager TP=1.

The ordinary constructor already declares the packed text QKV parameters. IPC
imports the finalized loader result into that exact schema, without repacking.
RoPE frequencies and modulation caches are unregistered process-local derived
state; existing forward paths rebuild their meta/empty values. timestep_zero is
a constructor-created local CUDA constant, never a shared checkpoint allocation.
"""

from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model

from . import common
from .common import build_meta as build_meta
from .common import load_ordinary as load_ordinary

ADAPTER_ID = "dit.qwen_image.transformer.v1"
PIPELINE_NAME = "QwenImagePipeline"
PIPELINE_MODULE = "sglang.multimodal_gen.runtime.pipelines.qwen_image"
MODEL_LABEL = "original Qwen-Image"
EXPECTED_CONFIG = {
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 60,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 768,
}


def supports_config(config):
    return config.get("_class_name") == "QwenImageTransformer2DModel" and all(
        config.get(key) == value for key, value in EXPECTED_CONFIG.items()
    )


def validate_supported(frozen, *, pipeline_name, attention):
    from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageArchConfig
    from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
        QwenImageTransformer2DModel,
    )

    recipe = frozen.thaw()
    if (
        pipeline_name != PIPELINE_NAME
        or recipe.component_name != "transformer"
        or recipe.model_cls is not QwenImageTransformer2DModel
    ):
        raise ValueError(
            "No weight-cache adapter for this pipeline/component/model/loader"
        )
    config = recipe.init_params["hf_config"]
    # The ordinary resolver consumes _class_name; the resolved native class
    # above is authoritative at this boundary.
    if not all(
        config.get(key) == value for key, value in EXPECTED_CONFIG.items()
    ) or any(not key.startswith("_") and key not in EXPECTED_CONFIG for key in config):
        raise ValueError(
            "Weight cache supports the original Qwen-Image architecture only"
        )
    arch = recipe.init_params["config"].arch_config
    for name, expected in EXPECTED_CONFIG.items():
        actual = getattr(arch, name, None)
        if name == "axes_dims_rope" and isinstance(actual, (list, tuple)):
            actual = list(actual)
        if actual != expected:
            raise ValueError(f"Unverified resolved Qwen-Image architecture: {name}")
    if (
        type(arch) is not QwenImageArchConfig
        or arch.zero_cond_t
        or getattr(arch, "use_additional_t_cond", False)
        or getattr(arch, "use_layer3d_rope", False)
    ):
        raise ValueError("Unverified Qwen-Image edit/layered/conditioning variant")
    common.validate_recipe(recipe, attention=attention, label="Qwen-Image")


def fingerprint_fields(frozen):
    return common.fingerprint_fields(frozen, adapter_id=ADAPTER_ID)


def finalize_after_import(model):
    # For this unquantized row, post_load_weights only leaves these constructor
    # flags disabled. Do not rerun quantization hooks on imported allocations.
    for block in model.transformer_blocks:
        if any(
            getattr(block, name)
            for name in (
                "_enable_nvfp4_resnorm_quant",
                "_fp8_img_attn_norm_quant",
                "_fp8_txt_attn_norm_quant",
                "_fp8_img_mlp_norm_quant",
                "_fp8_txt_mlp_norm_quant",
            )
        ):
            raise ValueError("Unverified quantized Qwen-Image derived state")
    return finalize_loaded_model(model)
