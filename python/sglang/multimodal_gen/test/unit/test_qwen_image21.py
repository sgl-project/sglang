# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    _patchify,
    _unpatchify,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    collapse_image_slots,
)


def test_condition_slots_expand_to_actual_latent_grid():
    hidden = torch.randn(22, 8)
    ids = torch.tensor([1, 2] + [99] * 16 + [3, 4, 5, 6])
    collapsed, slots = collapse_image_slots(hidden, ids, 99)
    assert collapsed.shape == (7, 8)
    layout = build_layout(slots.tolist(), [(1, 4, 8), (1, 2, 2)], (4, 6, 6), "cpu")
    assert len(layout["image_indices"]) == 32
    assert len(layout["prefix_rope"]) == 38
    assert layout["segments"] == ((0, 2, False), (2, 34, True), (34, 38, False))
    torch.testing.assert_close(collapsed[slots][0], hidden[2])


def test_adjacent_image_slots_stay_distinct():
    layout = build_layout(
        [False, True, True, False], [(1, 2, 2), (1, 4, 2), (1, 2, 2)], (4, 6, 6), "cpu"
    )
    assert layout["segments"] == (
        (0, 1, False),
        (1, 5, True),
        (5, 13, True),
        (13, 14, False),
    )
    with pytest.raises(ValueError, match="slots"):
        build_layout([False], [(1, 2, 2), (1, 2, 2)], (4, 6, 6), "cpu")


def test_latent_pack_decode_contract():
    config = QwenImage21PipelineConfig()
    batch = SimpleNamespace(height=64, width=96)
    shape = config.prepare_latent_shape(batch, 2, 1)
    x = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    packed = config.maybe_pack_latents(x, 2, batch)
    assert packed.shape == (2, 24, 64)
    decoded = config.post_denoising_loop(packed, batch)
    torch.testing.assert_close(decoded[:, :, 0], x[:, 0])
    scale, shift = config.get_decode_scale_and_shift("cpu", torch.float32, None)
    torch.testing.assert_close(
        (decoded.float() - shift) * scale / scale + shift, decoded.float()
    )


def test_native_vae_roundtrip_shapes_and_checkpoint_names():
    ac = QwenImage21VAEArchConfig(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        temperal_downsample=(False, False, False, False),
    )
    model = AutoencoderKLQwenImage21(
        QwenImage21VAEConfig(arch_config=ac, use_tiling=False)
    ).eval()
    with torch.no_grad():
        moments = model._encode(torch.randn(1, 3, 1, 32, 64))
        assert moments.shape == (1, 8, 1, 2, 4)
        output = model._decode(moments[:, :4])
        assert output.shape == (1, 3, 1, 32, 64)
    assert model.state_dict()["encoder.conv_in.weight"].ndim == 4
    x = torch.randn(2, 3, 1, 8, 12)
    torch.testing.assert_close(_unpatchify(_patchify(x, 2), 2), x)


def test_architecture_derived_dimensions():
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(num_attention_heads=2, attention_head_dim=16)
    )
    assert config.hidden_size == 32


def test_registry_routes_local_checkpoint_and_preserves_legacy():
    assert (
        _get_config_info("Qwen/Qwen-Image-2.1").pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info(
            "/models/private", model_id="Qwen-Image-2.1"
        ).pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info("Qwen/Qwen-Image").pipeline_config_cls
        is not QwenImage21PipelineConfig
    )
