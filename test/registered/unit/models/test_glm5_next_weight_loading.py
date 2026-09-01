import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, _param, loaded_weight):
        self.loaded = loaded_weight


class TestGlm5NextWeightLoading(unittest.TestCase):
    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_runai_streamed_fused_qkv_a_proj_owns_buffer(self, post_load):
        fused_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=0, num_hidden_layers=1),
            num_fused_shared_experts=0,
            quant_config=None,
            fuse_qkv_a_proj=True,
            named_parameters=lambda: iter(
                [
                    (
                        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
                        fused_param,
                    )
                ]
            ),
        )

        staging_buffer = torch.tensor([1, 2], dtype=torch.float32)

        def streamed_weights():
            q_view = staging_buffer[:]
            setattr(q_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.q_a_proj.weight", q_view

            staging_buffer.copy_(torch.tensor([3, 4], dtype=torch.float32))
            kv_view = staging_buffer[:]
            setattr(kv_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_view

        Glm5NextForConditionalGeneration.load_weights(model, streamed_weights())

        self.assertEqual(fused_param.loaded.tolist(), [1, 2, 3, 4])
        post_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
