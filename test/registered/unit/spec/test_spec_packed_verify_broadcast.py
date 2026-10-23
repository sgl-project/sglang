# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_SPEC_PACKED_VERIFY_BROADCAST``: the verify result in one collective.

``_alloc_verify_outputs`` returns ``predict`` / ``accept_index`` /
``num_correct_drafts`` with the same values either way; switched on they are
views of one buffer, so ``_broadcast_verify_outputs`` issues one collective.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_utils import (
    _alloc_verify_outputs,
    _broadcast_verify_outputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WIDTH = 4


class _RecordingGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.calls = []

    def broadcast(self, tensor, src=0):
        self.calls.append((tensor.data_ptr(), tensor.numel(), src))
        return tensor


class TestVerifyOutputs(CustomTestCase):
    def test_same_values_and_one_buffer(self):
        for bs in (1, 3, 8):
            with self.subTest(bs=bs):
                with envs.SGLANG_SPEC_PACKED_VERIFY_BROADCAST.override(False):
                    ref = _alloc_verify_outputs(bs * WIDTH, bs, WIDTH, "cpu")
                with envs.SGLANG_SPEC_PACKED_VERIFY_BROADCAST.override(True):
                    got = _alloc_verify_outputs(bs * WIDTH, bs, WIDTH, "cpu")
                self.assertIsNone(ref[0])
                packed = got[0]
                self.assertEqual(packed.numel(), bs * WIDTH + bs * WIDTH + bs)
                for r, g in zip(ref[1:3], got[1:3]):
                    self.assertEqual((r.shape, r.dtype), (g.shape, g.dtype))
                    self.assertTrue(torch.equal(r, g))
                    self.assertTrue(g.is_contiguous())
                self.assertEqual(got[3].shape, ref[3].shape)
                base = packed.data_ptr()
                sizes = [t.numel() * t.element_size() for t in got[1:]]
                offsets = [t.data_ptr() - base for t in got[1:]]
                self.assertEqual(offsets, [0, sizes[0], sizes[0] + sizes[1]])

    def test_one_collective_when_packed(self):
        with envs.SGLANG_SPEC_PACKED_VERIFY_BROADCAST.override(True):
            packed, predict, accept_index, num_correct = _alloc_verify_outputs(
                2 * WIDTH, 2, WIDTH, "cpu"
            )
        group = _RecordingGroup(world_size=8)
        _broadcast_verify_outputs(group, packed, predict, accept_index, num_correct)
        self.assertEqual(group.calls, [(packed.data_ptr(), packed.numel(), 0)])

        group = _RecordingGroup(world_size=8)
        _broadcast_verify_outputs(group, None, predict, accept_index, num_correct)
        self.assertEqual(
            [c[0] for c in group.calls],
            [predict.data_ptr(), accept_index.data_ptr(), num_correct.data_ptr()],
        )

        group = _RecordingGroup(world_size=1)
        _broadcast_verify_outputs(group, packed, predict, accept_index, num_correct)
        self.assertEqual(group.calls, [])


if __name__ == "__main__":
    unittest.main()
