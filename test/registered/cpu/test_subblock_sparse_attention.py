# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    _get_subblock_sparse_attention_runner,
    _sm90_sparse_attention,
    _sm100_sparse_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-b-test-cpu")


class TestSubBlockSparseAttentionDispatch(CustomTestCase):
    def setUp(self):
        _get_subblock_sparse_attention_runner.cache_clear()
        self.addCleanup(_get_subblock_sparse_attention_runner.cache_clear)

    def test_dispatch_is_resolved_once_per_device(self):
        device = torch.device("cuda:0")
        with patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ) as get_capability:
            first = _get_subblock_sparse_attention_runner(device)
            second = _get_subblock_sparse_attention_runner(device)

        self.assertIs(first, _sm90_sparse_attention)
        self.assertIs(second, first)
        get_capability.assert_called_once_with(device)

    def test_dispatches_sm100(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            runner = _get_subblock_sparse_attention_runner(device)

        self.assertIs(runner, _sm100_sparse_attention)

    def test_rejects_unsupported_compute_capability(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(
                RuntimeError,
                "supports compute capability 9.0 or 10.0;.*10.3 device",
            ):
                _get_subblock_sparse_attention_runner(device)


if __name__ == "__main__":
    unittest.main(verbosity=3)
