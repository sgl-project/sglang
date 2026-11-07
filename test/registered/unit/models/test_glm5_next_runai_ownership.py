import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models import glm5_next
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PREFIX = "model.layers.0.self_attn"


def _stream_through_one_buffer(weights):
    buffer = torch.empty(max(w.numel() for _, w in weights))
    for name, weight in weights:
        view = buffer[: weight.numel()].view_as(weight)
        view.copy_(weight)
        setattr(view, RUNAI_STREAMER_TENSOR_ATTR, True)
        yield name, view


class TestGlm5NextRunaiOwnership(CustomTestCase):
    @torch.no_grad()
    def test_fused_qkv_a_proj_keeps_a_copy_of_the_first_projection(self):
        q_a = torch.full((3, 4), 1.0)
        kv_a = torch.full((3, 4), 2.0)
        fused = nn.Parameter(torch.full((6, 4), torch.nan), requires_grad=False)
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=0),
            num_fused_shared_experts=0,
            quant_config=None,
            fuse_qkv_a_proj=True,
            named_parameters=lambda: iter(
                [(f"{PREFIX}.fused_qkv_a_proj_with_mqa.weight", fused)]
            ),
        )

        with patch.object(glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"):
            glm5_next.Glm5NextForConditionalGeneration.load_weights(
                model,
                _stream_through_one_buffer(
                    [
                        (f"{PREFIX}.q_a_proj.weight", q_a),
                        (f"{PREFIX}.kv_a_proj_with_mqa.weight", kv_a),
                    ]
                ),
            )

        torch.testing.assert_close(fused.data, torch.cat([q_a, kv_a]))


if __name__ == "__main__":
    unittest.main()
