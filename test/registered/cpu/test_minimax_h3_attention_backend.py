# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    _get_subblock_sparse_attention_runner,
    _sm90_sparse_attention,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RecordingAttentionImpl:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordingBackend:
    @staticmethod
    def get_impl_cls():
        return _RecordingAttentionImpl

    @staticmethod
    def get_enum():
        return "recording"


class TestMiniMaxH3AttentionBackend(CustomTestCase):
    def test_forwards_layer_prefix_to_attention_impl(self):
        attention = SimpleNamespace(
            num_heads=8,
            head_dim=128,
            softmax_scale=128**-0.5,
            prefix="blocks.7.attn",
        )

        MiniMaxH3Attention._set_attention_backend(attention, _RecordingBackend())

        self.assertEqual(attention._attention_impl.kwargs["prefix"], "blocks.7.attn")

    def test_sm90_dispatch_is_resolved_once_per_device(self):
        """Repeated layers must not query static device properties on the hot path."""
        _get_subblock_sparse_attention_runner.cache_clear()
        self.addCleanup(_get_subblock_sparse_attention_runner.cache_clear)

        with patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ) as get_capability:
            first = _get_subblock_sparse_attention_runner(torch.device("cuda:0"))
            second = _get_subblock_sparse_attention_runner(torch.device("cuda:0"))

        self.assertIs(first, _sm90_sparse_attention)
        self.assertIs(second, first)
        get_capability.assert_called_once_with(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main(verbosity=3)
