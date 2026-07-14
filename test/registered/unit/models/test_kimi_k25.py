"""CPU coverage for Kimi-K2.5/K2.7 encoder-DP wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MoonViT3dTower:
    device = torch.device("cpu")
    merge_kernel_size = (2, 2)

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=2)
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
        )
        self.grid_thws = None

    def __call__(self, pixel_values, grid_thws):
        self.grid_thws = grid_thws
        # MoonViT3d returns a list of [tokens, merge_area, hidden] tensors.
        return [pixel_values.reshape(-1, 4, pixel_values.shape[-1])]


class _Projector:
    def __call__(self, image_embeds):
        return image_embeds


def _image_item(feature, grid_thw):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 1)],
        feature=feature,
        model_specific_data={"image_grid_thw": torch.tensor(grid_thw)},
    )


def test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1():
    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)

    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        output = run_dp_sharded_mrope_vision_model(
            tower, pixel_values, [[1, 2, 2]], rope_type="rope_2d_packed"
        )

    assert torch.equal(output, pixel_values.reshape(1, 4, 2))
    assert torch.equal(tower.grid_thws, torch.tensor([[1, 2, 2]]))


def test_dp_helper_uses_config_hidden_size_for_empty_moonvit3d_rank():
    class _GatherGroup:
        def all_gather(self, tensor, dim):
            return torch.cat([torch.ones_like(tensor), tensor], dim=dim)

    tower = _MoonViT3dTower()
    parallel = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=1,
        attn_tp_group=_GatherGroup(),
    )

    with patch("sglang.srt.multimodal.mm_utils.get_parallel", return_value=parallel):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            torch.randn(4, 2),
            [[1, 2, 2]],
            rope_type="rope_2d_packed",
        )

    assert output.shape == (1, 4, 2)
    assert tower.grid_thws is None


def test_kimi_k25_encoder_dp_selects_packed_moonvit_contract():
    model = KimiK25ForConditionalGeneration.__new__(KimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.use_data_parallel = True
    model.vision_tower = _MoonViT3dTower()
    model.mm_projector = _Projector()
    items = [_image_item(torch.randn(4, 2), [[1, 2, 2]])]
    sharded_embeddings = torch.randn(1, 2)

    with patch(
        "sglang.srt.models.kimi_k25.run_dp_sharded_mrope_vision_model",
        return_value=sharded_embeddings,
    ) as run_dp:
        output = model.get_image_feature(items)

    assert output is sharded_embeddings
    assert run_dp.call_args.kwargs == {"rope_type": "rope_2d_packed"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
