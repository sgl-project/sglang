# SPDX-License-Identifier: Apache-2.0
"""Shared narrow DiT adapter mechanics; all loading stays in TransformerLoader."""

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
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def validate_recipe(recipe, *, attention, label):
    args = recipe.server_args
    if (
        recipe.quant_spec.runtime_quant_config is not None
        or recipe.quant_spec.gguf_file is not None
        or recipe.quant_spec.post_load_hooks
    ):
        raise ValueError(
            f"Weight cache {label} adapter does not support resolved quantization/post-load variants"
        )
    if recipe.quant_spec.param_dtype != torch.bfloat16:
        raise ValueError(f"Weight cache {label} adapter requires bf16 parameters")
    if attention != "fa" or args.attention_backend not in (None, "fa"):
        raise ValueError(
            f"Weight cache {label} adapter currently supports FA attention only"
        )
    if args.attention_backend_config:
        raise ValueError(
            "Weight cache has not verified custom attention backend configuration"
        )
    if recipe.component_starts_on_cpu or args.should_use_fsdp_for_component(
        "transformer"
    ):
        raise ValueError(
            f"Weight cache {label} adapter requires resident non-FSDP weights"
        )
    for field in (
        "tp_size",
        "sp_degree",
        "cfg_parallel_degree",
        "dp_size",
        "num_gpus",
        "nnodes",
    ):
        if getattr(args, field) != 1:
            raise ValueError(f"Weight cache {label} adapter requires {field}=1")
    if (
        args.lora_path is not None
        or args.enable_torch_compile
        or args.enable_breakable_cuda_graph
    ):
        raise ValueError(
            f"Weight cache initial {label} adapter requires eager inference without LoRA"
        )


def fingerprint_fields(frozen, *, adapter_id):
    recipe = frozen.thaw()
    return {
        "adapter": adapter_id,
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
