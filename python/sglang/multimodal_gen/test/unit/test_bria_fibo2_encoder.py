# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Fibo 2.0 text-encoder side.

The two failure modes guarded here both produce a model that runs and returns
wrong images without raising anything: using the reference layer indices
directly against the in-tree Qwen3-VL tuple (which is offset by one), and
building the encoder from the inherited Qwen3-VL-8B defaults instead of the 4B
values the checkpoint ships.
"""

import json
import math
import os

import pytest
import torch

from sglang.multimodal_gen.configs.models.encoders.bria_fibo2 import (
    PROMPT_TEMPLATE_ENCODE,
    SGLANG_HIDDEN_STATE_OFFSET,
    TEXT_ENCODER_OUT_LAYERS,
    BriaFibo2Qwen3VLArchConfig,
    BriaFibo2Qwen3VLConfig,
    build_caption_bundles,
    sglang_layer_indices,
    unique_hf_layers,
)

MODEL_CFG_DIR = os.environ.get("FIBO2_MODEL_CFG_DIR", "/workspace/dsh/fibo2/model_cfg")


def _load_text_encoder_config() -> dict:
    path = os.path.join(MODEL_CFG_DIR, "text_encoder_config.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------


def test_out_layers_match_the_reference():
    assert TEXT_ENCODER_OUT_LAYERS == (
        (9, 20, 31),
        (5, 16, 27),
        (7, 18, 29),
        (9, 20, 31),
        (11, 22, 33),
        (13, 24, 35),
    )


def test_fifteen_unique_layers_not_eighteen_slots():
    """Six bundles of three layers share one repeated bundle, so 15 not 18."""
    layers = unique_hf_layers()
    assert len(layers) == 15
    assert layers == [5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35]


def test_sglang_indices_are_shifted_by_one():
    assert SGLANG_HIDDEN_STATE_OFFSET == 1
    assert sglang_layer_indices() == (
        (8, 19, 30),
        (4, 15, 26),
        (6, 17, 28),
        (8, 19, 30),
        (10, 21, 32),
        (12, 23, 34),
    )


# ---------------------------------------------------------------------------
# R1: the layer offset. A one-hot state per layer identifies which layer a
# bundle actually read, so a missing shift changes the answer instead of quietly
# shifting the weights.
# ---------------------------------------------------------------------------


def _one_hot_states(hidden_size: int, num_layers: int, text_len: int = 2):
    """states[i] has RMS 1 and a single non-zero channel at position i."""
    states = []
    for layer in range(num_layers):
        state = torch.zeros(1, text_len, hidden_size)
        state[..., layer] = math.sqrt(hidden_size)
        states.append(state)
    return tuple(states)


@pytest.mark.parametrize("bundle_index", range(len(TEXT_ENCODER_OUT_LAYERS)))
def test_bundles_read_the_shifted_layer(bundle_index):
    hidden_size = 64  # wider than the layer count, so the marker is unambiguous
    num_layers = 36
    states = _one_hot_states(hidden_size, num_layers)

    bundles = build_caption_bundles(
        states, num_hidden_layers=num_layers, dtype=torch.float32
    )
    assert bundles.shape == (1, 6, 2, len(TEXT_ENCODER_OUT_LAYERS[0]) * hidden_size)

    bundle = bundles[0, bundle_index]
    expected_layers = TEXT_ENCODER_OUT_LAYERS[bundle_index]
    for block, hf_layer in enumerate(expected_layers):
        block_values = bundle[0, block * hidden_size : (block + 1) * hidden_size]
        marker = int(torch.argmax(block_values.abs()).item())
        # the in-tree tuple holds layer outputs, so the reference index is off by one
        assert marker == hf_layer - SGLANG_HIDDEN_STATE_OFFSET, (
            f"bundle {bundle_index} block {block} read layer {marker}, "
            f"expected {hf_layer - SGLANG_HIDDEN_STATE_OFFSET}"
        )


def test_bundle_values_are_unit_rms():
    hidden_size = 16
    num_layers = 36
    states = tuple(
        torch.randn(1, 3, hidden_size) * (layer + 1) for layer in range(num_layers)
    )
    bundles = build_caption_bundles(
        states, num_hidden_layers=num_layers, dtype=torch.float32
    )
    for block in range(3):
        block_values = bundles[..., block * hidden_size : (block + 1) * hidden_size]
        rms = block_values.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_bundles_cast_back_to_the_requested_dtype():
    states = _one_hot_states(64, 36)
    bundles = build_caption_bundles(states, num_hidden_layers=36, dtype=torch.bfloat16)
    assert bundles.dtype == torch.bfloat16


def test_layer_count_mismatch_fails_loudly():
    """If the tower ever adds the embedding entry, this must raise, not misread."""
    hidden_size, num_layers = 16, 36
    # 37 entries: the `transformers` convention, i.e. one too many for us
    states = tuple(torch.randn(1, 2, hidden_size) for _ in range(num_layers + 1))
    with pytest.raises(ValueError, match="one hidden state per decoder layer"):
        build_caption_bundles(states, num_hidden_layers=num_layers)


def test_bundle_rejects_wrong_cap_feat_dim():
    """3 x text hidden size must equal the transformer's cap_feat_dim."""
    config = BriaFibo2Qwen3VLConfig(
        arch_config=BriaFibo2Qwen3VLArchConfig(
            hidden_size=2560,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=36,
            cap_feat_dim=7000,
        )
    )
    with pytest.raises(ValueError, match="cap_feat_dim must be 3"):
        config.post_diffusers_config_update()


# ---------------------------------------------------------------------------
# R5: flatten the real 4B shape out of the checkpoint config
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_CFG_DIR, "text_encoder_config.json")),
    reason="model config dump not available",
)
def test_arch_is_flattened_from_the_real_text_config():
    config = BriaFibo2Qwen3VLConfig()
    config.update_model_arch(_load_text_encoder_config())
    config.post_diffusers_config_update()
    arch = config.arch_config

    # Qwen3-VL-4B-Instruct, not the inherited 8B defaults
    assert arch.hidden_size == 2560
    assert arch.num_hidden_layers == 36
    assert arch.num_attention_heads == 32
    assert arch.num_key_value_heads == 8
    assert arch.head_dim == 128
    assert arch.intermediate_size == 9728
    assert arch.vocab_size == 151936
    assert arch.rms_norm_eps == pytest.approx(1e-6)
    # rope_theta lives under `rope_parameters` in this checkpoint
    assert arch.rope_theta == pytest.approx(5_000_000.0)

    vision = arch.vision_config
    assert vision["depth"] == 24
    assert vision["hidden_size"] == 1024
    assert vision["out_hidden_size"] == 2560
    assert vision["spatial_merge_size"] == 2
    assert vision["patch_size"] == 16
    assert vision["deepstack_visual_indexes"] == [5, 11, 17]


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_CFG_DIR, "text_encoder_config.json")),
    reason="model config dump not available",
)
def test_bundle_width_matches_the_transformer_cross_attention_width():
    """3 x 2560 must be exactly the 7680 the DiT cross-attention reads."""
    from sglang.multimodal_gen.configs.models.dits.bria_fibo2 import (
        BriaFibo2ArchConfig,
    )

    config = BriaFibo2Qwen3VLConfig()
    config.update_model_arch(_load_text_encoder_config())
    config.post_diffusers_config_update()

    dit_arch = BriaFibo2ArchConfig()
    assert config.arch_config.cap_feat_dim == dit_arch.cap_feat_dim
    assert 3 * config.arch_config.hidden_size == dit_arch.cap_feat_dim


def test_prompt_template_matches_the_reference():
    assert (
        PROMPT_TEMPLATE_ENCODE
        == "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


def test_encoder_keeps_its_own_tokenizer_settings():
    """The Fibo encoder must not inherit JoyImage's prompt offsets."""
    arch = BriaFibo2Qwen3VLArchConfig()
    assert arch.max_sequence_length == 4096
    assert arch.tokenizer_kwargs["max_length"] == 4096
    assert not hasattr(arch, "prompt_template_encode_start_idx") or (
        arch.prompt_template_encode_start_idx == {"image": 34, "video": 91}
    )
