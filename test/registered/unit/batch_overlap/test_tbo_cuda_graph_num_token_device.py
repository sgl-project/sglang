"""Regression: the TBO cuda-graph plugin must allocate its children
``num_token_non_padded`` buffer on the model device, not CPU.

``ForwardBatch.num_token_non_padded`` is a scalar tensor on the model device
(see ``ForwardBatch.compute``, which does ``.to(device, ...)``). The eager TBO
split path already honors this -- ``compute_tbo_children_num_token_non_padded_raw``
moves the tensor to ``get_server_args().device`` -- but
``TboCudaGraphRunnerPlugin`` preallocated its persistent buffer with a bare
``torch.zeros((2,), dtype=torch.int32)``, leaving it on CPU.

During TBO cuda-graph replay that CPU buffer becomes the child's
``num_token_non_padded`` and is compared against a device ``arange`` in the
padded-region mask (``_mask_topk_ids_padded_region`` /
``_zero_topk_weights_padded_region``), raising a device-mismatch error
(an illegal memory access on HIP/aiter MoE).

CPU-only: a ``meta`` device makes the buffer placement observable without a GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.batch_overlap.two_batch_overlap as tbo
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboCudaGraphRunnerPlugin,
    TboForwardBatchPreparer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTboCudaGraphNumTokenDevice(CustomTestCase):
    def test_plugin_buffer_on_model_device(self):
        # Use 'meta' so the configured device differs from the implicit CPU
        # default; a bare torch.zeros() would leave the buffer on CPU and fail.
        fake_args = SimpleNamespace(device="meta")
        with patch.object(tbo, "get_server_args", lambda: fake_args):
            plugin = TboCudaGraphRunnerPlugin()

        buf = plugin._tbo_children_num_token_non_padded
        self.assertEqual(tuple(buf.shape), (2,))
        self.assertEqual(buf.dtype, torch.int32)
        self.assertEqual(buf.device.type, "meta")

    def test_graph_and_eager_paths_agree_on_device(self):
        # Both the preallocated cuda-graph buffer and the eager split tensor must
        # land on the same (model) device, matching ForwardBatch's contract.
        fake_args = SimpleNamespace(device="meta")
        with patch.object(tbo, "get_server_args", lambda: fake_args):
            eager = (
                TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                    tbo_split_token_index=3, num_token_non_padded=8
                )
            )
            plugin = TboCudaGraphRunnerPlugin()

        self.assertEqual(
            eager.device.type,
            plugin._tbo_children_num_token_non_padded.device.type,
        )

    def test_eager_split_values(self):
        # value_a = min(split, n); value_b = max(0, n - split). Computed on CPU
        # so the values are materializable.
        fake_args = SimpleNamespace(device="cpu")
        with patch.object(tbo, "get_server_args", lambda: fake_args):
            eager = (
                TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                    tbo_split_token_index=3, num_token_non_padded=8
                )
            )
        self.assertEqual(eager.dtype, torch.int32)
        self.assertEqual(eager.tolist(), [3, 5])

    def test_replay_updates_persistent_child_count_views(self):
        fake_args = SimpleNamespace(device="cpu")
        with patch.object(tbo, "get_server_args", lambda: fake_args):
            plugin = TboCudaGraphRunnerPlugin()

        child_a, child_b = plugin._tbo_children_num_token_non_padded
        plugin.replay_prepare(
            forward_mode=ForwardMode.DECODE,
            bs=8,
            num_token_non_padded=5,
            spec_info=None,
        )
        self.assertEqual((child_a.item(), child_b.item()), (4, 1))

        plugin.replay_prepare(
            forward_mode=ForwardMode.DECODE,
            bs=8,
            num_token_non_padded=8,
            spec_info=None,
        )
        self.assertEqual((child_a.item(), child_b.item()), (4, 4))


if __name__ == "__main__":
    unittest.main()
