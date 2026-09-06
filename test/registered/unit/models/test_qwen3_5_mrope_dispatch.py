"""CPU-only tests for Qwen3.5 attention preparation dispatch."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen3_5MropeDispatch(CustomTestCase):
    def _run_self_attention(self, positions):
        prepared = tuple(object() for _ in range(4))
        attention_output = torch.ones(2, 4)
        gated_output = torch.full((2, 4), 2.0)
        projected_output = torch.full((2, 4), 3.0)
        layer = SimpleNamespace(
            attn_output_gate=True,
            forward_prepare_cuda_fused=mock.Mock(return_value=prepared),
            forward_prepare_fused_gate=mock.Mock(),
            forward_prepare_native=mock.Mock(return_value=prepared),
            forward_prepare_npu=mock.Mock(),
            attn=mock.Mock(return_value=attention_output),
            o_proj=mock.Mock(return_value=(projected_output, None)),
        )
        hidden_states = torch.zeros(2, 4)
        forward_batch = object()

        with (
            mock.patch.multiple(
                qwen3_5,
                _is_cuda=True,
                _is_hip=False,
                _is_xpu=False,
                _is_cpu=False,
                _is_npu=False,
            ),
            mock.patch.object(
                qwen3_5,
                "fused_sigmoid_mul",
                return_value=gated_output,
            ) as sigmoid_mul,
        ):
            output = Qwen3_5AttentionDecoderLayer.self_attention(
                layer,
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        layer.attn.assert_called_once_with(*prepared[:3], forward_batch)
        sigmoid_mul.assert_called_once_with(attention_output, prepared[3], inplace=True)
        layer.o_proj.assert_called_once_with(gated_output)
        layer.forward_prepare_fused_gate.assert_not_called()
        layer.forward_prepare_npu.assert_not_called()
        self.assertIs(output, projected_output)
        return layer

    def test_mrope_positions_use_native_preparation(self):
        positions = torch.arange(6).reshape(3, 2)

        layer = self._run_self_attention(positions)

        layer.forward_prepare_native.assert_called_once_with(
            positions=positions,
            hidden_states=mock.ANY,
        )
        layer.forward_prepare_cuda_fused.assert_not_called()

    def test_text_positions_use_cuda_fused_preparation(self):
        positions = torch.arange(2)

        layer = self._run_self_attention(positions)

        layer.forward_prepare_cuda_fused.assert_called_once_with(
            positions=positions,
            hidden_states=mock.ANY,
        )
        layer.forward_prepare_native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
