import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.runai_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNvFp4OnlineRunaiStreamer(unittest.TestCase):
    def test_pending_fp8_weight_owns_streamed_buffer(self):
        layer = SimpleNamespace(
            w2_weight_scale=object(),
            w2_weight_scale_2=object(),
        )
        param = torch.nn.Parameter(torch.empty(1))
        loaded_calls = []
        dequantized_weights = []

        def original_weight_loader(*args, **kwargs):
            loaded_calls.append((args, kwargs))

        def fake_dequantizer(weight, _scale, _device):
            dequantized_weights.append(weight.clone())
            return torch.zeros((2, 16), dtype=torch.bfloat16)

        def fake_quantize(weight):
            return (
                torch.zeros_like(weight, dtype=torch.uint8),
                torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
                torch.tensor(1.0),
            )

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            side_effect=fake_quantize,
        ):
            loader = ModelOptNvFp4OnlineFusedMoEMethod.get_online_weight_loader(
                layer,
                original_weight_loader,
                layer_log_name="test",
                fp8_dequantizer=fake_dequantizer,
            )

            staging_buffer = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
            streamed_weight = staging_buffer[:]
            setattr(streamed_weight, RUNAI_STREAMER_TENSOR_ATTR, True)
            loader(
                param,
                streamed_weight,
                "experts.0.w2.weight",
                shard_id="w2",
                expert_id=None,
            )

            staging_buffer.copy_(torch.tensor([9.0, 9.0], dtype=torch.float8_e4m3fn))
            streamed_scale = staging_buffer[:]
            setattr(streamed_scale, RUNAI_STREAMER_TENSOR_ATTR, True)
            loader(
                param,
                streamed_scale,
                "experts.0.w2.weight_scale",
                shard_id="w2",
                expert_id=None,
            )

        self.assertEqual(dequantized_weights[0].to(torch.float32).tolist(), [1.0, 2.0])
        self.assertEqual(len(loaded_calls), 3)


if __name__ == "__main__":
    unittest.main()
