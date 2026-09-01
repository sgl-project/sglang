import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestQwen38Nvfp4FusionPolicy(unittest.TestCase):
    @staticmethod
    def _run(
        *,
        num_tokens: int,
        forward_mode: ForwardMode,
        full_prefill_graph: bool,
    ):
        gate_up = torch.ones((num_tokens, 4))
        fused = torch.full((num_tokens, 2), 2.0)
        eager = torch.full((num_tokens, 2), 3.0)
        mlp = SimpleNamespace(
            _enable_silu_fp4_quant_fusion=True,
            gate_up_proj=mock.Mock(return_value=(gate_up, None)),
            _silu_fp4_quant_fused=mock.Mock(return_value=fused),
            act_fn=mock.Mock(return_value=eager),
            down_proj=mock.Mock(side_effect=lambda value: (value, None)),
        )

        with mock.patch(
            "sglang.srt.models.qwen2_moe.check_cuda_graph_backend",
            return_value=full_prefill_graph,
        ):
            output = Qwen2MoeMLP.forward(
                mlp,
                torch.zeros((num_tokens, 2)),
                SimpleNamespace(forward_mode=forward_mode),
            )
        return mlp, output

    def test_full_graph_prefill_falls_back_from_fused_quantization(self):
        mlp, output = self._run(
            num_tokens=128,
            forward_mode=ForwardMode.EXTEND,
            full_prefill_graph=True,
        )

        mlp._silu_fp4_quant_fused.assert_not_called()
        mlp.act_fn.assert_called_once()
        torch.testing.assert_close(output, torch.full((128, 2), 3.0))

    def test_other_prefill_backends_keep_fused_quantization(self):
        mlp, output = self._run(
            num_tokens=28,
            forward_mode=ForwardMode.EXTEND,
            full_prefill_graph=False,
        )

        mlp._silu_fp4_quant_fused.assert_called_once()
        mlp.act_fn.assert_not_called()
        torch.testing.assert_close(output, torch.full((28, 2), 2.0))

    def test_decode_keeps_fused_quantization(self):
        mlp, output = self._run(
            num_tokens=1,
            forward_mode=ForwardMode.DECODE,
            full_prefill_graph=True,
        )

        mlp._silu_fp4_quant_fused.assert_called_once()
        mlp.act_fn.assert_not_called()
        torch.testing.assert_close(output, torch.full((1, 2), 2.0))


if __name__ == "__main__":
    unittest.main()
