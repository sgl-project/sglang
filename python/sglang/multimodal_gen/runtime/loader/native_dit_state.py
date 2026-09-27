# SPDX-License-Identifier: Apache-2.0
"""Audited native DiT representations, independent of pipeline bindings.

Support rows are data, not complete per-model cache adapters. Only real schema
or derived-state differences need hooks. Other component loaders can implement
the same ComponentStateContract without inheriting DiT assumptions.
"""

import importlib
import json
from collections.abc import Callable

import msgspec
import torch

from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model

WAN_CONFIG = {
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
QWEN_IMAGE_CONFIG = {
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
MINIMAX_H3_CONFIG = {
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}


def _validate_runtime(recipe, *, attention, label):
    args = recipe.server_args
    if (
        recipe.quant_spec.runtime_quant_config is not None
        or recipe.quant_spec.gguf_file is not None
        or recipe.quant_spec.post_load_hooks
    ):
        raise ValueError(
            f"Weight cache {label} does not support resolved quantization/post-load variants"
        )
    if recipe.quant_spec.param_dtype != torch.bfloat16:
        raise ValueError(f"Weight cache {label} requires bf16 parameters")
    if attention != "fa" or args.attention_backend not in (None, "fa"):
        raise ValueError(f"Weight cache {label} currently supports FA attention only")
    if args.attention_backend_config:
        raise ValueError(
            "Weight cache has not verified custom attention backend configuration"
        )
    if recipe.component_starts_on_cpu or args.should_use_fsdp_for_component(
        recipe.component_name
    ):
        raise ValueError(f"Weight cache {label} requires resident non-FSDP weights")
    for field in (
        "tp_size",
        "sp_degree",
        "cfg_parallel_degree",
        "dp_size",
        "num_gpus",
        "nnodes",
    ):
        if getattr(args, field) != 1:
            raise ValueError(f"Weight cache {label} requires {field}=1")
    if (
        args.lora_path is not None
        or args.enable_torch_compile
        or args.enable_breakable_cuda_graph
    ):
        raise ValueError(f"Weight cache {label} requires eager inference without LoRA")


def _validate_resolved_fields(recipe, expected):
    arch = recipe.init_params["config"].arch_config
    for name, value in expected.items():
        actual = getattr(arch, name, None)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(
                f"Unverified resolved {recipe.model_cls.__name__} architecture: {name}"
            )


def _validate_wan(recipe):
    arch = recipe.init_params["config"].arch_config
    if (
        arch.attention_type != "original"
        or arch.local_attn_size != -1
        or arch.sink_size != 0
    ):
        raise ValueError("Unverified Wan attention/causal variant")


def _validate_qwen_image(recipe):
    from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageArchConfig

    _validate_resolved_fields(recipe, QWEN_IMAGE_CONFIG)
    arch = recipe.init_params["config"].arch_config
    if type(arch) is not QwenImageArchConfig or arch.zero_cond_t:
        raise ValueError("Unverified Qwen-Image edit/layered/conditioning variant")


def _validate_minimax_h3(recipe):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )

    _validate_resolved_fields(recipe, MINIMAX_H3_CONFIG)
    arch = recipe.init_params["config"].arch_config
    if (
        type(arch) is not MiniMaxH3DiTArchConfig
        or arch.adaln_curve_grid is not None
        or arch.adaln_affine_input_dim is not None
        or arch.checkpoint_uses_diffusers_layout
        or arch.has_gate_compress
        or recipe.server_args.minimax_h3_adaln_cache_path is not None
        or recipe.server_args.minimax_h3_adaln_online
        or any(key.startswith("adaln_") for key in recipe.init_params)
    ):
        raise ValueError("Unverified MiniMax-H3 layout/pruning/AdaLN cache variant")


def _unchanged_schema(model):
    return model


def _minimax_h3_schema(model):
    # Ordinary assign=True loading registers persistent FP32 RoPE as a parameter.
    model.rope.inv_freq = torch.nn.Parameter(model.rope.inv_freq, requires_grad=False)
    return model


def _finalize_qwen_image(model):
    # Packed QKV is already finalized. Never repeat a quantization/repacking hook.
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


def _finalize_minimax_h3(model):
    if (
        model._adaln_precomputed
        or model.adaln_cache is not None
        or model.adaln_t_table is not None
        or model.adaln_basis is not None
        or model.adaln_mean is not None
    ):
        raise ValueError("Unverified MiniMax-H3 derived AdaLN state")
    # For this representation the hook only validates FP32 projections / RoPE.
    model.post_load_weights()
    return finalize_loaded_model(model)


class NativeDiTStateContract(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    contract_id: str
    model_module: str
    model_name: str
    config_json: str
    validate_architecture: Callable
    adapt_meta_schema: Callable = _unchanged_schema
    finalize_after_import: Callable = finalize_loaded_model

    @property
    def expected_config(self):
        return json.loads(self.config_json)

    def supports_config(self, config):
        return config.get("_class_name") == self.model_name and all(
            config.get(key) == value for key, value in self.expected_config.items()
        )

    def matches_model(self, model_cls):
        if (model_cls.__module__, model_cls.__name__) != (
            self.model_module,
            self.model_name,
        ):
            return False
        return model_cls is getattr(
            importlib.import_module(self.model_module), self.model_name
        )

    def validate_supported(self, frozen, *, attention):
        recipe = frozen.thaw()
        if not self.matches_model(recipe.model_cls):
            raise ValueError("No weight-cache state contract for this resolved model")
        config = recipe.init_params["hf_config"]
        expected = self.expected_config
        if not all(config.get(key) == value for key, value in expected.items()) or any(
            not key.startswith("_") and key not in expected for key in config
        ):
            raise ValueError(
                f"Unverified {self.model_name} architecture/configuration extension"
            )
        _validate_runtime(recipe, attention=attention, label=self.model_name)
        self.validate_architecture(recipe)


WAN = NativeDiTStateContract(
    "native_dit.wan2_1.bf16.fa.v1",
    "sglang.multimodal_gen.runtime.models.dits.wanvideo",
    "WanTransformer3DModel",
    json.dumps(WAN_CONFIG, sort_keys=True),
    _validate_wan,
)
QWEN_IMAGE = NativeDiTStateContract(
    "native_dit.qwen_image.bf16.fa.v1",
    "sglang.multimodal_gen.runtime.models.dits.qwen_image",
    "QwenImageTransformer2DModel",
    json.dumps(QWEN_IMAGE_CONFIG, sort_keys=True),
    _validate_qwen_image,
    finalize_after_import=_finalize_qwen_image,
)
MINIMAX_H3 = NativeDiTStateContract(
    "native_dit.minimax_h3.fl2va.bf16.fa.v1",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3",
    "MiniMaxH3DiTModel",
    json.dumps(MINIMAX_H3_CONFIG, sort_keys=True),
    _validate_minimax_h3,
    adapt_meta_schema=_minimax_h3_schema,
    finalize_after_import=_finalize_minimax_h3,
)
CONTRACTS = (WAN, QWEN_IMAGE, MINIMAX_H3)


def for_model(model_cls):
    for contract in CONTRACTS:
        if contract.matches_model(model_cls):
            return contract
    raise ValueError("No weight-cache state contract for this resolved model")
