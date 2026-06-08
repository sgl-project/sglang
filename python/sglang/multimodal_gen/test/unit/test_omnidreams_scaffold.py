# SPDX-License-Identifier: Apache-2.0
"""Phase-0 scaffold checks for the OmniDreams DiT (CPU-only, no checkpoint).

Constructs the DiT on the meta device (no memory for ~2B params) and validates
the checkpoint-exact structure against an independently-derived authoritative
key fixture, plus the pre/post-fusion shapes, the 2-step flow-match sigmas, and
the 3D-RoPE layout.
"""

import os

import torch

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
    OmniDreamsPipelineConfig,
    warp_flow_match_sigmas,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
    ROPE_IS_NEOX_STYLE,
    OmniDreamsDiT,
    rope_dims,
)

_KEY_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "omnidreams_dit_keys.txt"
)


def _load_fixture_keys() -> set[str]:
    with open(_KEY_FIXTURE) as f:
        return {line.strip() for line in f if line.strip()}


def _build_meta_model() -> OmniDreamsDiT:
    with torch.device("meta"):
        return OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})


def test_state_dict_matches_authoritative_key_fixture():
    model = _build_meta_model()
    keys = set(model.state_dict().keys())
    expected = _load_fixture_keys()
    assert len(expected) == 570
    assert (
        keys == expected
    ), f"missing={sorted(expected - keys)} extra={sorted(keys - expected)}"


def test_unique_bias_is_crossattn_proj():
    model = _build_meta_model()
    biases = [k for k in model.state_dict() if k.endswith(".bias")]
    assert biases == ["crossattn_proj.0.bias"]


def test_pre_fusion_shapes():
    model = _build_meta_model()
    sd = model.state_dict()
    # x_embedder keeps the padding-mask channel pre-fusion: (16 + 1 + 1) * 2 * 2 = 72.
    assert tuple(sd["x_embedder.proj.1.weight"].shape) == (2048, 72)
    # HDMap embed: 16 * 2 * 2 = 64 in-features.
    assert tuple(sd["additional_patch_embedding.proj.1.weight"].shape) == (2048, 64)
    # Final layer pre-shuffle: patch_dim = 2*2*1*16 = 64.
    assert tuple(sd["final_layer.linear.weight"].shape) == (64, 2048)
    assert tuple(sd["crossattn_proj.0.weight"].shape) == (1024, 100352)


def test_post_load_weights_fuses_in_place():
    model = _build_meta_model()
    pre_keys = set(model.state_dict().keys())
    model.post_load_weights()
    sd = model.state_dict()
    # Padding-mask channels dropped: 72 -> 68.
    assert tuple(sd["x_embedder.proj.1.weight"].shape) == (2048, 68)
    # Shuffle fuse is a reorder; shape is preserved.
    assert tuple(sd["final_layer.linear.weight"].shape) == (64, 2048)
    # Fusion must not add or remove parameters.
    assert set(sd.keys()) == pre_keys
    assert model._is_padding_mask_fused and model._is_shuffle_op_fused


def test_two_step_flow_match_sigmas():
    sigmas = warp_flow_match_sigmas()
    assert len(sigmas) == 3
    assert abs(sigmas[0] - 1.0) < 1e-9
    assert abs(sigmas[1] - 0.8036) < 1e-3
    assert sigmas[2] == 0.0
    # The pipeline config exposes the same schedule.
    assert OmniDreamsPipelineConfig().denoising_sigmas() == sigmas


def test_rope_layout_neox_44_42_42():
    assert rope_dims(128) == (44, 42, 42)
    assert sum(rope_dims(128)) == 128
    assert ROPE_IS_NEOX_STYLE is True
