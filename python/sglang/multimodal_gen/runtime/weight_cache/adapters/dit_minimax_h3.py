# SPDX-License-Identifier: Apache-2.0
"""Native original FL2VA, mixed bf16/fp32 resident DiT, FA, eager TP=1.

Grouped checkpoint QKV is reordered by the ordinary loader exactly once. The
consumer imports that finalized layout. RoPE's persistent FP32 state becomes a
parameter under the ordinary assign=True loader; declare that finalized schema
on meta too. The lazy nonpersistent timestep frequency cache stays process-local.
"""

import torch

from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model

from . import common
from .common import load_ordinary as load_ordinary

ADAPTER_ID = "dit.minimax_h3.fl2va.v1"
PIPELINE_NAME = "MiniMaxH3Pipeline"
PIPELINE_MODULE = "sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline"
MODEL_LABEL = "native original MiniMax-H3 FL2VA"
SUPPORTS_SUBFOLDER = True
EXPECTED_CONFIG = {
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


def supports_config(config):
    return config.get("_class_name") == "MiniMaxH3DiTModel" and all(
        config.get(key) == value for key, value in EXPECTED_CONFIG.items()
    )


def validate_model_index(model_index):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
        MiniMaxH3ReleaseMetadata,
    )

    metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
    if metadata.partition != "fl2va":
        raise ValueError("Weight cache supports original MiniMax-H3 FL2VA only")


def validate_supported(frozen, *, pipeline_name, attention):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel

    recipe = frozen.thaw()
    if (
        pipeline_name != PIPELINE_NAME
        or recipe.component_name != "transformer"
        or recipe.model_cls is not MiniMaxH3DiTModel
    ):
        raise ValueError("No weight-cache adapter for this MiniMax-H3 model/loader")
    config = recipe.init_params["hf_config"]
    if not all(
        config.get(key) == value for key, value in EXPECTED_CONFIG.items()
    ) or any(not key.startswith("_") and key not in EXPECTED_CONFIG for key in config):
        raise ValueError("Weight cache supports original MiniMax-H3 architecture only")
    arch = recipe.init_params["config"].arch_config
    for name, expected in EXPECTED_CONFIG.items():
        actual = getattr(arch, name, None)
        if name == "patch_size" and isinstance(actual, (list, tuple)):
            actual = list(actual)
        if actual != expected:
            raise ValueError(f"Unverified resolved MiniMax-H3 architecture: {name}")
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
    common.validate_recipe(recipe, attention=attention, label="MiniMax-H3")


def fingerprint_fields(frozen):
    return common.fingerprint_fields(frozen, adapter_id=ADAPTER_ID)


def build_meta(frozen):
    model = common.build_meta(frozen)
    # Match load_model_from_full_model_state_dict, including registration kind.
    model.rope.inv_freq = torch.nn.Parameter(model.rope.inv_freq, requires_grad=False)
    return model


def finalize_after_import(model):
    if (
        model._adaln_precomputed
        or model.adaln_cache is not None
        or model.adaln_t_table is not None
        or model.adaln_basis is not None
        or model.adaln_mean is not None
    ):
        raise ValueError("Unverified MiniMax-H3 derived AdaLN state")
    # With all AdaLN variants excluded, this existing hook ONLY validates the
    # required FP32 projections and RoPE. It never transforms imported weights.
    model.post_load_weights()
    return finalize_loaded_model(model)
