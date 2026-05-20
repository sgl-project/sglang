"""Unit tests for Qwen3.5 empty hidden state handling in local compute paths."""

import types
import unittest

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5GatedDeltaNet,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SentinelCalled(RuntimeError):
    pass


def _raise(name):
    raise _SentinelCalled(f"{name} was called for an empty hidden_states tensor")


def _empty_hidden_states():
    return torch.empty((0, 64), dtype=torch.float32)


class TestQwen3_5EmptyTensor(CustomTestCase):
    def test_gated_delta_net_skips_input_proj_on_empty(self):
        """GatedDeltaNet returns early before _forward_input_proj on empty input."""
        layer = object.__new__(Qwen3_5GatedDeltaNet)
        layer._forward_input_proj = types.MethodType(
            lambda self, hidden_states: _raise("_forward_input_proj"),
            layer,
        )

        output = Qwen3_5GatedDeltaNet.forward(layer, _empty_hidden_states(), object())

        self.assertEqual(output.shape, torch.Size([0, 64]))

    def test_self_attention_skips_qkv_proj_on_empty(self):
        """self_attention returns early before qkv_proj on empty input."""
        layer = object.__new__(Qwen3_5AttentionDecoderLayer)
        layer.qkv_proj = types.MethodType(
            lambda self, hidden_states: _raise("qkv_proj"),
            layer,
        )
        positions = torch.empty((0,), dtype=torch.long)

        output = Qwen3_5AttentionDecoderLayer.self_attention(
            layer, positions, _empty_hidden_states(), object()
        )

        self.assertEqual(output.shape, torch.Size([0, 64]))


if __name__ == "__main__":
    unittest.main()
