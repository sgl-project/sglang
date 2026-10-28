"""Cover strided Mamba state detection and staging for HiCache."""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestStridedStateDetection(CustomTestCase):
    def test_contiguous_slots_are_not_strided(self):
        for shape in ((8, 4), (8, 4, 3), (1, 5)):
            with self.subTest(shape=shape):
                self.assertFalse(
                    MambaPoolHost._slots_are_strided(torch.zeros(shape)),
                    "a contiguous slot array must take the direct path",
                )

    def test_envelope_strided_slots_are_detected(self):
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.zeros(num_slots * envelope)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        self.assertTrue(MambaPoolHost._slots_are_strided(view))

    def test_empty_tensor_is_not_strided(self):
        self.assertFalse(MambaPoolHost._slots_are_strided(torch.zeros((0, 4))))

    def test_staging_round_trip_preserves_slot_contents(self):
        """Gather and scatter preserve selected slots without changing their neighbors."""
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.arange(num_slots * envelope, dtype=torch.float32)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        indices = torch.tensor([4, 1, 3])

        staged = view.index_select(0, indices)
        self.assertTrue(staged.is_contiguous())
        for row, slot in enumerate(indices.tolist()):
            self.assertTrue(torch.equal(staged[row], view[slot]))

        dst = torch.zeros_like(raw)
        dst_view = torch.as_strided(
            dst, size=(num_slots, per_slot), stride=(envelope, 1)
        )
        dst_view.index_copy_(0, indices, staged)
        for slot in indices.tolist():
            self.assertTrue(torch.equal(dst_view[slot], view[slot]))
        # A partial transfer must leave unselected slots untouched.
        for slot in set(range(num_slots)) - set(indices.tolist()):
            self.assertTrue(torch.all(dst_view[slot] == 0))


if __name__ == "__main__":
    unittest.main()
