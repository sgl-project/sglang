# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.models.encoders.bagel_vit import (
    BagelImageEncoderArchConfig,
    BagelImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.bagel_vit import (
    _FA4_MIN_SEQUENCE_LENGTH,
    BagelImageEncoder,
    _SiglipAttention,
    _uses_sm100_fa4_crossover,
    preprocess_bagel_vit_image,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability


@pytest.fixture
def tiny_config() -> BagelImageEncoderConfig:
    return BagelImageEncoderConfig(
        arch_config=BagelImageEncoderArchConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            max_image_size=4,
            min_image_size=2,
            max_num_patches_per_side=2,
            position_embedding_rows=4,
            llm_hidden_size=6,
        )
    )


@pytest.fixture
def bagel_head_config() -> BagelImageEncoderConfig:
    return BagelImageEncoderConfig(
        arch_config=BagelImageEncoderArchConfig(
            hidden_size=72,
            intermediate_size=144,
            num_hidden_layers=1,
            num_attention_heads=1,
            patch_size=2,
            max_image_size=4,
            min_image_size=2,
            max_num_patches_per_side=2,
            position_embedding_rows=4,
            llm_hidden_size=6,
        )
    )


def test_preprocess_patch_order_and_position_ids(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    pixels = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    image = Image.fromarray(pixels)

    patches, position_ids, grid = preprocess_bagel_vit_image(image, tiny_config)

    assert grid == (1, 2)
    assert patches.shape == (2, 12)
    assert position_ids.tolist() == [0, 1]
    normalized = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    normalized = normalized * 2 - 1
    expected = normalized.reshape(3, 1, 2, 2, 2)
    expected = torch.einsum("chpwq->hwpqc", expected).reshape(2, 12)
    torch.testing.assert_close(patches, expected)


def test_forward_projects_to_llm_width(tiny_config: BagelImageEncoderConfig) -> None:
    torch.manual_seed(0)
    encoder = BagelImageEncoder(tiny_config).eval()
    patches = torch.randn(4, 12)
    position_ids = torch.tensor([0, 1, 2, 3])

    with set_forward_context(current_timestep=0, attn_metadata=None):
        output = encoder(patches, position_ids)

    assert output.shape == (4, 6)
    assert torch.isfinite(output).all()


def test_attention_matches_noncausal_sdpa(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    torch.manual_seed(1)
    encoder = BagelImageEncoder(tiny_config).eval()
    attention = encoder.vit_model.vision_model.encoder.layers[0].self_attn
    hidden_states = torch.randn(5, tiny_config.arch_config.hidden_size)
    query = attention.q_proj(hidden_states).view(1, 5, 2, 4)
    key = attention.k_proj(hidden_states).view(1, 5, 2, 4)
    value = attention.v_proj(hidden_states).view(1, 5, 2, 4)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=attention.head_dim**-0.5,
    )
    expected = attention.out_proj(expected.transpose(1, 2).reshape(5, -1))

    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = attention(hidden_states)

    assert attention.attn.backend == AttentionBackendEnum.TORCH_SDPA
    torch.testing.assert_close(actual, expected)


def test_attention_backend_support_is_limited_to_fa_and_sdpa(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    assert tiny_config.arch_config._supported_attention_backends == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }


def test_fa4_backend_uses_sdpa_below_validated_crossover(
    bagel_head_config: BagelImageEncoderConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2)
    attention = _SiglipAttention(bagel_head_config).eval()
    attention.attn.backend = AttentionBackendEnum.FA
    attention._use_sm100_fa4_crossover = True
    hidden_states = torch.randn(17, bagel_head_config.arch_config.hidden_size)
    query = attention.q_proj(hidden_states).view(1, 17, 1, 72)
    key = attention.k_proj(hidden_states).view(1, 17, 1, 72)
    value = attention.v_proj(hidden_states).view(1, 17, 1, 72)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=attention.head_dim**-0.5,
    )
    expected = attention.out_proj(expected.transpose(1, 2).reshape(17, -1))

    def fail_if_called(*_: object, **__: object) -> torch.Tensor:
        raise AssertionError("short FA4 path must use SDPA directly")

    monkeypatch.setattr(attention.attn, "forward", fail_if_called)

    output = attention(hidden_states)

    torch.testing.assert_close(output, expected)


def test_fa4_backend_runs_native_attention_at_crossover(
    tiny_config: BagelImageEncoderConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = BagelImageEncoder(tiny_config).eval()
    attention = encoder.vit_model.vision_model.encoder.layers[0].self_attn
    attention.attn.backend = AttentionBackendEnum.FA
    attention._use_sm100_fa4_crossover = True
    hidden_states = torch.randn(
        _FA4_MIN_SEQUENCE_LENGTH, tiny_config.arch_config.hidden_size
    )
    call_count = 0

    def fake_fa4(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal call_count
        del key, value
        call_count += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(attention.attn, "forward", fake_fa4)

    output = attention(hidden_states)

    assert call_count == 1
    assert output.shape == hidden_states.shape


def test_non_sm100_fa_backend_does_not_use_sm100_crossover(
    tiny_config: BagelImageEncoderConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _SiglipAttention(tiny_config).eval()
    attention.attn.backend = AttentionBackendEnum.FA
    attention._use_sm100_fa4_crossover = False
    hidden_states = torch.randn(17, tiny_config.arch_config.hidden_size)
    call_count = 0

    def fake_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal call_count
        del key, value
        call_count += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(attention.attn, "forward", fake_attention)

    output = attention(hidden_states)

    assert call_count == 1
    assert output.shape == hidden_states.shape


def test_sm100_fa_backend_enables_measured_crossover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(10, 0),
    )

    assert _uses_sm100_fa4_crossover(AttentionBackendEnum.FA)
    assert not _uses_sm100_fa4_crossover(AttentionBackendEnum.TORCH_SDPA)


def test_weight_loader_is_streaming_strict_and_meta_safe(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    with torch.device("meta"):
        encoder = BagelImageEncoder(tiny_config)
    expected = [
        (name, torch.full(tuple(parameter.shape), index + 1.0))
        for index, (name, parameter) in enumerate(encoder.named_parameters())
    ]

    loaded = encoder.load_weights(
        iter([("language_model.ignored", torch.ones(1)), *expected])
    )
    encoder.to("cpu")

    assert loaded == {name for name, _ in expected}
    assert not any(parameter.is_meta for parameter in encoder.parameters())
    with pytest.raises(ValueError, match="unexpected image encoder weights"):
        encoder.load_weights(
            [("vit_model.vision_model.unknown", torch.ones(1))], strict=True
        )


def test_forward_rejects_invalid_positions(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    encoder = BagelImageEncoder(tiny_config)
    with pytest.raises(ValueError, match="outside the checkpoint table"):
        encoder(torch.zeros(1, 12), torch.tensor([4]))
