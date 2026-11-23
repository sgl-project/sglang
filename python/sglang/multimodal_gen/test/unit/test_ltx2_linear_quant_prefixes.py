# SPDX-License-Identifier: Apache-2.0
"""LTX-2 DiT linears receive their checkpoint-relative module path.

Prefix-aware quantization configs (``--quantization-ignored-layers``,
per-layer checkpoint markers) select a method per linear from that path.
"""

import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25Config
from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2VideoTransformer3DModel,
)

TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "cross_attention_dim": 32,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "caption_channels": 32,
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 1,
}


def _linear_methods(quant_config):
    config = LTX25Config()
    for key, value in TINY.items():
        setattr(config.arch_config, key, value)
    config.arch_config.__post_init__()
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(
            config=config, hf_config={}, quant_config=quant_config
        )
    return {
        name: type(module.quant_method).__name__
        for name, module in model.named_modules()
        if isinstance(module, LinearBase)
    }


def test_quantization_ignored_layers_match_ltx2_module_paths(
    single_process_model_parallel,
):
    methods = _linear_methods(
        Fp8Config(ignored_layers=["transformer_blocks.0.attn1", "proj_out"])
    )
    for name in ("transformer_blocks.0.attn1.to_q", "proj_out"):
        assert methods[name] == "UnquantizedLinearMethod", name
    for name in ("transformer_blocks.0.attn2.to_q", "patchify_proj"):
        assert methods[name] == "Fp8LinearMethod", name


def test_without_ignored_layers_every_linear_keeps_the_global_method(
    single_process_model_parallel,
):
    """No ignore list: every linear built with the quant config stays quantized."""
    methods = _linear_methods(Fp8Config())
    quantized = {
        name: method
        for name, method in methods.items()
        if name.startswith("transformer_blocks.")
        or name in ("patchify_proj", "audio_patchify_proj", "proj_out")
    }
    assert len(quantized) > 10
    assert set(quantized.values()) == {"Fp8LinearMethod"}
