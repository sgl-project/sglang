"""CPU checks: a DeepEP low-latency layer narrower than the shared buffer dispatches in the buffer's layout.

The process has one low-latency buffer, laid out for its creator's expert count (a speculative target's), and DeepEP
leaves zero-initialized signalling state in that layout. A draft MoE with fewer experts maps its experts onto the
buffer's local slots instead of running a second layout over the same state; a wider layer is refused.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.token_dispatcher import deepep  # noqa: E402

Impl = deepep._DeepEPDispatcherImplLowLatency


def _layer(num_experts, buffer_experts, ranks=8):
    layer = SimpleNamespace(
        num_experts=num_experts,
        group=SimpleNamespace(size=lambda: ranks),
        overlap_args=None,
    )
    patch = mock.patch.object(
        deepep.DeepEPBuffer, "layout_num_experts", return_value=buffer_experts
    )
    return layer, patch


class TestDeepEPLowLatencyBufferLayout(CustomTestCase):
    def test_a_layer_of_the_buffers_width_keeps_its_ids(self):
        layer, patch = _layer(384, 384)
        with patch:
            self.assertIsNone(Impl._buffer_layout(layer))

    def test_a_narrower_layer_maps_each_expert_to_its_ranks_local_slot(self):
        """128 draft experts over 8 ranks (16 local) in a 384-expert buffer (48 local): same rank, same local slot."""
        layer, patch = _layer(128, 384)
        with patch:
            layout = Impl._buffer_layout(layer)
        self.assertEqual(layout, (384, 16, 48))
        ids = torch.tensor([[0, 15, 16, 127, -1]])
        mapped = Impl._to_buffer_ids(ids, layout)
        self.assertEqual(mapped.tolist(), [[0, 15, 48, 7 * 48 + 15, -1]])
        self.assertTrue(torch.equal(mapped[mapped >= 0] // 48, ids[ids >= 0] // 16))

    def test_a_wider_layer_is_refused(self):
        """More local experts than the buffer's layout holds would overrun its per-expert state."""
        layer, patch = _layer(384, 128)
        with (
            patch,
            self.assertRaisesRegex(NotImplementedError, "no more local experts"),
        ):
            Impl._buffer_layout(layer)

    def test_combine_overlap_on_a_narrower_layer_is_refused(self):
        layer, patch = _layer(128, 384)
        layer.overlap_args = object()
        with patch, self.assertRaisesRegex(NotImplementedError, "overlap"):
            Impl._buffer_layout(layer)


if __name__ == "__main__":
    unittest.main()
