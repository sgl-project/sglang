import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
