# SPDX-License-Identifier: Apache-2.0
"""Verified initial representation: native Wan2.1 T2V 1.3B, bf16, FA, TP=1."""

from sglang.multimodal_gen.runtime.loader.utils import finalize_loaded_model

from . import common
from .common import build_meta as build_meta
from .common import load_ordinary as load_ordinary

ADAPTER_ID = "dit.wan2_1.transformer.v1"
PIPELINE_NAME = "WanPipeline"
PIPELINE_MODULE = "sglang.multimodal_gen.runtime.pipelines.wan_pipeline"
MODEL_LABEL = "Wan2.1 T2V 1.3B"
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


supports_config = is_wan_1_3b_config


def validate_supported(frozen, *, pipeline_name, attention):
    from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel

    recipe = frozen.thaw()
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
    common.validate_recipe(recipe, attention=attention, label="Wan")
    arch = recipe.init_params["config"].arch_config
    if (
        arch.attention_type != "original"
        or arch.local_attn_size != -1
        or arch.sink_size != 0
    ):
        raise ValueError("Unverified Wan attention/causal variant")


def fingerprint_fields(frozen):
    return common.fingerprint_fields(frozen, adapter_id=ADAPTER_ID)


def finalize_after_import(model):
    # Wan's post_load_weights is the audited base no-op; no weight transforms
    # may be repeated here. RoPE is computed from each request, not shared state.
    return finalize_loaded_model(model)
