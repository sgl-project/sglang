# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 vision tower layout and native TP1 parity."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3OmniPipelineConfig,
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import (
    MiniMaxH3ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn import (
    VDNH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionTransformer,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.runtime_context import get_parallel


@pytest.mark.parametrize(
    "config_cls",
    [
        MiniMaxH3PipelineConfig,
        FastH3PipelineConfig,
        FastH3OmniPipelineConfig,
        PikaH3FlexPipelineConfig,
        VDNH3PipelineConfig,
    ],
)
def test_video_releases_replicate_vision(config_cls) -> None:
    assert config_cls().text_encoder_configs[0].vision_replicated


def test_image_release_keeps_sharded_vision() -> None:
    assert not MiniMaxH3ImagePipelineConfig().text_encoder_configs[0].vision_replicated


@pytest.mark.parametrize("switch", [True, False])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_encoder_builds_the_tower_its_config_selects(
    monkeypatch, switch, tp_size
) -> None:
    built = {}

    class RecordingModel(nn.Module):
        def __init__(self, arch, **kwargs) -> None:
            super().__init__()
            built.update(kwargs)
            self.language_model = nn.Module()

    monkeypatch.setattr(minimax_h3_qwen3vl, "Qwen3VLModel", RecordingModel)
    config = MiniMaxH3Qwen3VLConfig(vision_replicated=switch)
    arch = config.arch_config
    arch.text_config = SimpleNamespace(num_hidden_layers=arch.num_hidden_layers)
    arch.image_token_id = 1
    arch.video_token_id = 2

    with get_parallel().override(tp_size=tp_size, tp_rank=0, tp_group=None):
        encoder = minimax_h3_qwen3vl.MiniMaxH3Qwen3VLEncoder(config)

    assert ("model.visual.deepstack_merger_list" in encoder.layer_names) == (
        switch and tp_size > 1
    )
    assert built["vision_replicated"] is switch


@torch.inference_mode()
def test_replicated_video_tower_matches_native_tp1() -> None:
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=12,
        deepstack_visual_indexes=[0],
    )
    generator = torch.Generator().manual_seed(1234)
    grid = torch.tensor([[1, 2, 4], [2, 2, 2]])
    patches = torch.randn(16, 12, generator=generator)

    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        with get_parallel().override(tp_size=1, tp_rank=0, tp_group=None):
            native = Qwen3VLVisionTransformer(config)
            for param in native.parameters():
                param.uniform_(-0.25, 0.25, generator=generator)
            expected = native(patches, grid)

        for size, rank in [(1, 0), (2, 0), (2, 1)]:
            with get_parallel().override(tp_size=size, tp_rank=rank, tp_group=None):
                tower = Qwen3VLVisionTransformer(config, replicated=True)
                tower.load_state_dict(native.state_dict())
                group = "tp" if size == 1 else "replicated"
                assert tower.blocks[0].attn.qkv_proj.parallel_group == group
                actual = tower(patches, grid)

            torch.testing.assert_close(
                actual.pooler_output, expected.pooler_output, atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual.deepstack_features, expected.deepstack_features, atol=0, rtol=0
            )
