import sys
import unittest
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _VisionStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dtype_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.parameter_lookups = 0

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        self.parameter_lookups += 1
        return super().parameters(recurse=recurse)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return features.flatten(start_dim=2) + 1


class _ProjectorStub(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * 10


class TestDeepseekVL2ImageFeatures(CustomTestCase):
    def test_global_and_local_views_keep_layout_tokens_in_order(self) -> None:
        item = SimpleNamespace(
            feature=torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1),
            images_spatial_crop=torch.tensor([[[2, 1]]]),
        )
        expected_by_global_view_pos = {
            "head": [10, -1, -2, 20, 30, -1],
            "tail": [20, 30, -1, -2, 10, -1],
        }

        for global_view_pos, expected in expected_by_global_view_pos.items():
            with self.subTest(global_view_pos=global_view_pos):
                image_features = DeepseekVL2ForCausalLM.build_image_features(
                    items=[item],
                    vision=_VisionStub(),
                    projector=_ProjectorStub(),
                    image_newline=torch.tensor([-1.0]),
                    view_seperator=torch.tensor([-2.0]),
                    global_view_pos=global_view_pos,
                )

                torch.testing.assert_close(
                    image_features,
                    torch.tensor(expected, dtype=torch.float32).unsqueeze(1),
                )

    def test_vision_dtype_is_read_once_per_batch(self) -> None:
        item = SimpleNamespace(
            feature=torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1),
            images_spatial_crop=torch.tensor([[[1, 1]]]),
        )
        vision = _VisionStub()

        DeepseekVL2ForCausalLM.build_image_features(
            items=[item, item],
            vision=vision,
            projector=_ProjectorStub(),
            image_newline=torch.tensor([-1.0]),
            view_seperator=torch.tensor([-2.0]),
            global_view_pos="head",
        )

        self.assertEqual(vision.parameter_lookups, 1)

    def test_missing_timm_error_suppresses_import_cause(self) -> None:
        with mock.patch.dict(sys.modules, {"timm": None}):
            with self.assertRaisesRegex(ImportError, "Please install timm") as error:
                DeepseekVL2ForCausalLM._init_vision_module(
                    vision_config=mock.sentinel.vision_config,
                    quant_config=None,
                )

        self.assertIsNone(error.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
