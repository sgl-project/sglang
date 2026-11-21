from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImageEditPlusPipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    component_residency_strategies,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl_vision import (
    Qwen2_5VLVisionRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImageEditPipeline


def _vision_positions(device):
    grid = torch.arange(28, device=device)
    return torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_offloaded_vision_rope_matches_resident():
    """Qwen2.5-VL head size 80: CPU and CUDA pow round some inv_freq entries apart."""
    with torch.device("cpu"):
        offloaded = Qwen2_5VLVisionRotaryEmbedding(40)
    with torch.device("cuda"):
        resident = Qwen2_5VLVisionRotaryEmbedding(40)
    offloaded.recompute_on_device_change = True
    offloaded.cuda()

    positions = _vision_positions("cuda")
    expected = resident(positions)
    torch.testing.assert_close(offloaded(positions), expected, atol=0, rtol=0)
    torch.testing.assert_close(offloaded(positions), expected, atol=0, rtol=0)

    # --component-precisions.text_encoder bf16: each reload casts buffers too
    for _ in range(2):
        offloaded.cpu()
        offloaded.to("cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(offloaded(positions), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "pipeline_config_cls, rebuilds",
    [
        (QwenImageEditPipelineConfig, True),
        (QwenImageEditPlus_2511_PipelineConfig, True),
        (QwenImageEditPlusPipelineConfig, False),
    ],
)
def test_edit_pipeline_rebuilds_vision_rope_after_offload(
    pipeline_config_cls, rebuilds
):
    with torch.device("cpu"):
        rotary = Qwen2_5VLVisionRotaryEmbedding(40)
    _create_edit_stages(pipeline_config_cls, rotary)
    assert rotary.recompute_on_device_change is rebuilds


def _create_edit_stages(pipeline_config_cls, rotary):
    text_encoder = SimpleNamespace(
        model=SimpleNamespace(visual=SimpleNamespace(rotary_pos_emb=rotary))
    )
    pipeline = QwenImageEditPipeline.__new__(QwenImageEditPipeline)
    pipeline_config = pipeline_config_cls()
    # set from the checkpoint at load time
    pipeline_config.vae_config.arch_config.vae_scale_factor = 8
    server_args = SimpleNamespace(pipeline_config=pipeline_config)
    with (
        mock.patch.object(
            QwenImageEditPipeline,
            "get_module",
            lambda self, name, default=None: (
                text_encoder if name == "text_encoder" else default
            ),
        ),
        mock.patch.object(QwenImageEditPipeline, "add_standard_ti2i_stages"),
    ):
        pipeline.create_pipeline_stages(server_args)


def test_edit_pipeline_vision_rope_survives_bf16_offload_cycles():
    positions = _vision_positions("cpu")
    expected = Qwen2_5VLVisionRotaryEmbedding(40)(positions)
    encoder = torch.nn.Module()
    encoder.proj = torch.nn.Linear(4, 4)
    encoder.rotary_pos_emb = Qwen2_5VLVisionRotaryEmbedding(40)
    _create_edit_stages(QwenImageEditPlus_2511_PipelineConfig, encoder.rotary_pos_emb)

    strategy = ComponentOffloadStrategy()
    use = ComponentUse("text_encoding", "text_encoder", target_dtype=torch.bfloat16)
    with mock.patch.object(
        component_residency_strategies,
        "get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        for _ in range(2):
            # On CPU only, fp32 weights make every reload cast, as each
            # CPU->GPU reload does on a real device.
            encoder.proj.float()
            strategy.prepare_for_use(encoder, use, None)
            assert encoder.rotary_pos_emb.inv_freq.dtype == torch.bfloat16
            torch.testing.assert_close(
                encoder.rotary_pos_emb(positions), expected, atol=0, rtol=0
            )
            strategy.finish_use(encoder, use, None)
