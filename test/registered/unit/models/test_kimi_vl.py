"""CPU-only coverage for Kimi-VL encoder parallelism wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration
from sglang.srt.models.kimi_vl_moonvit import (
    MoonVitEncoderLayer,
    multihead_attention,
)
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VisionTower:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def __call__(
        self, pixel_values, image_grid_hws=None, max_seqlen=None, grid_hw=None
    ):
        image_grid_hws = grid_hw if image_grid_hws is None else image_grid_hws
        self.calls.append((pixel_values, image_grid_hws))
        # One merged token per image. The real MoonViT returns a list here.
        return [
            torch.full((1, 4, 2), index + 1.0)
            for index in range(image_grid_hws.shape[0])
        ]


class _Projector:
    def __init__(self):
        self.input = None

    def __call__(self, image_features):
        self.input = image_features
        return image_features


class _GridRecordingVisionTower:
    def __init__(self):
        self.grid_thw = None

    def __call__(self, pixel_values, grid_hw, max_seqlen=None):
        self.grid_thw = grid_hw
        self.max_seqlen = max_seqlen
        return pixel_values


def _bare_model(*, use_data_parallel: bool):
    """Build just enough of Kimi-VL to exercise ``get_image_feature``."""
    model = KimiVLForConditionalGeneration.__new__(KimiVLForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
    model.use_data_parallel = use_data_parallel
    model.vision_tower = _VisionTower()
    model.multi_modal_projector = _Projector()
    return model


def _image_item(feature, grid_hws):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 1)],
        feature=feature,
        model_specific_data={"image_grid_hws": torch.tensor(grid_hws)},
    )


class TestKimiVLEncoderParallelism(CustomTestCase):
    def test_moonvit_uses_tensor_parallel_layers(self):
        # At TP=1, sharded layers retain the checkpoint's original tensor
        # shapes while still exercising the same code path used by multi-GPU.
        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
        ):
            layer = MoonVitEncoderLayer(
                num_heads=2,
                hidden_dim=8,
                mlp_dim=16,
                prefix="vision_tower.encoder.blocks.0",
                use_tensor_parallel=True,
            )

        self.assertIsInstance(layer.wqkv, QKVParallelLinear)
        self.assertIsInstance(layer.wo, RowParallelLinear)
        self.assertIsInstance(layer.mlp.fc0, ColumnParallelLinear)
        self.assertIsInstance(layer.mlp.fc1, RowParallelLinear)
        self.assertEqual(layer.wqkv.weight.shape, (24, 8))
        self.assertEqual(layer.wo.weight.shape, (8, 8))
        self.assertEqual(layer.mlp.fc0.weight.shape, (16, 8))
        self.assertEqual(layer.mlp.fc1.weight.shape, (8, 16))

    def test_encoder_dp_uses_existing_mrope_sharding_helper(self):
        model = _bare_model(use_data_parallel=True)
        items = [
            _image_item(torch.randn(4, 2), [[2, 2]]),
            _image_item(torch.randn(8, 2), [[4, 2]]),
        ]
        sharded_features = torch.randn(3, 4, 2)

        with patch(
            "sglang.srt.models.kimi_vl.run_dp_sharded_mrope_vision_model",
            return_value=sharded_features,
        ) as run_dp:
            output = model.get_image_feature(items)

        run_dp.assert_called_once()
        vision_tower, pixel_values, grid_hws = run_dp.call_args.args
        self.assertIs(vision_tower, model.vision_tower)
        self.assertEqual(pixel_values.shape, (12, 2))
        self.assertEqual(grid_hws, [[2, 2], [4, 2]])
        self.assertEqual(run_dp.call_args.kwargs, {"rope_type": "rope_2d"})
        self.assertIs(model.multi_modal_projector.input, sharded_features)
        self.assertIs(output, sharded_features)
        self.assertEqual(model.vision_tower.calls, [])

    def test_default_path_preserves_existing_feature_wiring(self):
        model = _bare_model(use_data_parallel=False)
        items = [
            _image_item(torch.randn(4, 2), [[2, 2]]),
            _image_item(torch.randn(8, 2), [[4, 2]]),
        ]

        output = model.get_image_feature(items)

        self.assertEqual(len(model.vision_tower.calls), 1)
        pixel_values, image_grid_hws = model.vision_tower.calls[0]
        self.assertEqual(pixel_values.shape, (12, 2))
        self.assertEqual(image_grid_hws.tolist(), [[2, 2], [4, 2]])
        self.assertEqual(model.multi_modal_projector.input.shape, (2, 4, 2))
        self.assertIs(output, model.multi_modal_projector.input)

    def test_encoder_dp_keeps_moonvit_grid_metadata_on_vision_device(self):
        vision_tower = _GridRecordingVisionTower()
        pixel_values = torch.randn(4, 2)

        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
        ):
            output = run_dp_sharded_mrope_vision_model(
                vision_tower,
                pixel_values,
                [[2, 2]],
                rope_type="rope_2d",
            )

        self.assertIs(output, pixel_values)
        self.assertEqual(vision_tower.grid_thw.device, pixel_values.device)
        self.assertEqual(vision_tower.grid_thw.tolist(), [[2, 2]])
        self.assertEqual(vision_tower.max_seqlen, 4)

    def test_encoder_dp_tp1_concatenates_moonvit_image_outputs(self):
        vision_tower = _VisionTower()
        pixel_values = torch.randn(4, 2)

        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
        ):
            output = run_dp_sharded_mrope_vision_model(
                vision_tower,
                pixel_values,
                [[2, 2]],
                rope_type="rope_2d",
            )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 4, 2))

    def test_moonvit_attention_accepts_precomputed_max_seqlen(self):
        q = torch.randn(4, 2, 4, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        fake_output = torch.randn_like(q)

        with patch(
            "sglang.srt.models.kimi_vl_moonvit.flash_attn_varlen_func",
            return_value=fake_output,
        ) as flash_attn:
            output = multihead_attention(q, q, q, cu_seqlens, cu_seqlens, max_seqlen=4)

        self.assertTrue(torch.equal(output, fake_output.flatten(start_dim=-2)))
        self.assertEqual(flash_attn.call_args.args[5:7], (4, 4))


if __name__ == "__main__":
    unittest.main()
