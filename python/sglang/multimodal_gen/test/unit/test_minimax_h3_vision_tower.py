# SPDX-License-Identifier: Apache-2.0
"""Which MiniMax-H3 releases build the reference vision tower (CPU, no weights)."""

from types import SimpleNamespace

import pytest
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


def _vision_switches(pipeline_config) -> tuple[bool, bool]:
    encoder_config = pipeline_config.text_encoder_configs[0]
    return (
        encoder_config.vision_replicated,
        encoder_config.vision_transformers_numerics,
    )


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
def test_video_releases_build_the_reference_vision_tower(config_cls) -> None:
    assert _vision_switches(config_cls()) == (True, True)


def test_image_release_keeps_its_vision_tower() -> None:
    assert _vision_switches(MiniMaxH3ImagePipelineConfig()) == (False, False)


@pytest.mark.parametrize("switch", [True, False])
def test_encoder_builds_the_tower_its_config_selects(monkeypatch, switch) -> None:
    built = {}

    class RecordingModel(nn.Module):
        def __init__(self, arch, **kwargs) -> None:
            super().__init__()
            built.update(kwargs)
            self.language_model = nn.Module()

    monkeypatch.setattr(minimax_h3_qwen3vl, "Qwen3VLModel", RecordingModel)
    config = MiniMaxH3Qwen3VLConfig(
        vision_replicated=switch, vision_transformers_numerics=switch
    )
    arch = config.arch_config
    arch.text_config = SimpleNamespace(num_hidden_layers=arch.num_hidden_layers)
    arch.image_token_id = 1
    arch.video_token_id = 2

    minimax_h3_qwen3vl.MiniMaxH3Qwen3VLEncoder(config)

    assert built["vision_replicated"] is switch
    assert built["vision_transformers_numerics"] is switch


@pytest.mark.parametrize("switch", [True, False])
def test_vision_tower_switches_select_layout_and_rounding(switch) -> None:
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
        deepstack_visual_indexes=[],
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0),
        global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA),
    ):
        tower = Qwen3VLVisionTransformer(
            config, replicated=switch, transformers_numerics=switch
        )

    group = "replicated" if switch else "tp"
    attention = tower.blocks[0].attn
    assert attention.qkv_proj.parallel_group == group
    assert attention.proj.parallel_group == group
    assert tower.transformers_numerics is switch
    assert tower.rotary_pos_emb.host_inv_freq is switch
    # the reference SDPA kernel, through the reproducible cuDNN plan
    assert attention._attention_impl.allow_cudnn_sdp is switch
    assert attention._attention_impl.cudnn_composite_sdp is switch
