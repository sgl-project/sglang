# SPDX-License-Identifier: Apache-2.0
"""Verified initial representation: native Wan2.1 T2V 1.3B, bf16, FA, TP=1."""

import dataclasses

import torch

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    initialize_model_for_inference,
)
from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

ADAPTER_ID = "dit.wan2_1.transformer.v1"
EXPECTED_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "image_dim": None,
    "in_channels": 16,
    "num_attention_heads": 12,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "text_dim": 4096,
}


def is_wan_1_3b_config(config):
    return config.get("_class_name") == "WanTransformer3DModel" and all(
        config.get(key) == value for key, value in EXPECTED_CONFIG.items()
    )


def validate_supported(frozen, *, pipeline_name, attention):
    from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel

    recipe = frozen.thaw()
    args = recipe.server_args
    if (
        pipeline_name != "WanPipeline"
        or recipe.component_name != "transformer"
        or recipe.model_cls is not WanTransformer3DModel
    ):
        raise ValueError(
            "No weight-cache adapter for this pipeline/component/model/loader"
        )
    hf_config = recipe.init_params["hf_config"]
    if not all(hf_config.get(key) == value for key, value in EXPECTED_CONFIG.items()):
        raise ValueError("Weight cache supports the Wan2.1 T2V 1.3B architecture only")
    if any(not key.startswith("_") and key not in EXPECTED_CONFIG for key in hf_config):
        raise ValueError("Unverified Wan configuration extension")
    if (
        recipe.quant_spec.runtime_quant_config is not None
        or recipe.quant_spec.gguf_file is not None
        or recipe.quant_spec.post_load_hooks
    ):
        raise ValueError(
            "Weight cache Wan adapter does not support resolved quantization/post-load variants"
        )
    if recipe.quant_spec.param_dtype != torch.bfloat16:
        raise ValueError("Weight cache Wan adapter requires bf16 parameters")
    if attention != "fa" or args.attention_backend not in (None, "fa"):
        raise ValueError(
            "Weight cache Wan adapter currently supports FA attention only"
        )
    if args.attention_backend_config:
        raise ValueError(
            "Weight cache has not verified custom attention backend configuration"
        )
    arch = recipe.init_params["config"].arch_config
    if (
        arch.attention_type != "original"
        or arch.local_attn_size != -1
        or arch.sink_size != 0
    ):
        raise ValueError("Unverified Wan attention/causal variant")
    if recipe.component_starts_on_cpu or args.should_use_fsdp_for_component(
        "transformer"
    ):
        raise ValueError("Weight cache Wan adapter requires resident non-FSDP weights")
    for field in (
        "tp_size",
        "sp_degree",
        "cfg_parallel_degree",
        "dp_size",
        "num_gpus",
        "nnodes",
    ):
        if getattr(args, field) != 1:
            raise ValueError(f"Weight cache Wan adapter requires {field}=1")
    if (
        args.lora_path is not None
        or args.enable_torch_compile
        or args.enable_breakable_cuda_graph
    ):
        raise ValueError(
            "Weight cache initial Wan adapter requires eager inference without LoRA"
        )


def fingerprint_fields(frozen):
    recipe = frozen.thaw()
    return {
        "adapter": ADAPTER_ID,
        "model_cls": f"{recipe.model_cls.__module__}.{recipe.model_cls.__qualname__}",
        "config": dataclasses.asdict(recipe.init_params["config"]),
        "hf_config": recipe.init_params["hf_config"],
        "dtype": str(recipe.quant_spec.param_dtype),
        "attention": "fa",
        "quantization": None,
        "resident": True,
        "fsdp": False,
        "direct_gpu_weight_loading": recipe.weight_load_plan.load_full_state_dict_on_device,
    }


def load_ordinary(frozen):
    return TransformerLoader().load_prepared(
        frozen, attention_backend=AttentionBackendEnum.FA
    )[0]


def build_meta(frozen):
    recipe = frozen.thaw()
    with component_attn_backend_context_manager(
        AttentionBackendEnum.FA, component_name="transformer"
    ):
        model, _ = initialize_model_for_inference(
            recipe.model_cls,
            recipe.init_params,
            param_dtype=recipe.quant_spec.param_dtype,
        )
    return model.eval().requires_grad_(False)


def finalize_after_import(model):
    # Wan's post_load_weights is the audited base no-op; no weight transforms
    # may be repeated here. RoPE is computed from each request, not shared state.
    return finalize_loaded_model(model)
