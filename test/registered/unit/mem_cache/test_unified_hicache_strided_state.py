"""`MambaPoolHost` must notice an envelope-strided device state view.

Every state-transfer kernel in `pool_host/mamba.py` addresses a slot as
``ptr + index * item_size``, i.e. it assumes the slot stride equals the slot's
own size. The unified memory pool stores conv/SSM state ENVELOPE-strided: one
slot's stride spans every state tensor of every layer, so slot `i` does not
start at `i * numel_per_slot`. The mis-addressing stays inside the buffer, so
it corrupts silently instead of faulting -- which is why the predicate that
routes those views through the contiguous staging path is worth pinning.

    python -m pytest test/registered/unit/mem_cache/test_unified_hicache_strided_state.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestStridedStateDetection(CustomTestCase):
    def test_contiguous_slots_are_not_strided(self):
        """A plain per-slot array is what the kernels already handle."""
        for shape in ((8, 4), (8, 4, 3), (1, 5)):
            with self.subTest(shape=shape):
                self.assertFalse(
                    MambaPoolHost._slots_are_strided(torch.zeros(shape)),
                    "a contiguous slot array must take the direct path",
                )

    def test_envelope_strided_slots_are_detected(self):
        """One slot's stride spanning a wider envelope is the unified layout."""
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.zeros(num_slots * envelope)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        self.assertTrue(MambaPoolHost._slots_are_strided(view))

    def test_empty_tensor_is_not_strided(self):
        """No slots, nothing to address -- must not divide by or index slot 0."""
        self.assertFalse(MambaPoolHost._slots_are_strided(torch.zeros((0, 4))))

    def test_staging_round_trip_preserves_slot_contents(self):
        """The property the staging path relies on: gathering the wanted slots
        out of a strided view and scattering them back is the identity, so the
        kernel can run against a contiguous copy in between."""
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
        # Slots outside the index set must be untouched, or a partial backup
        # would clobber a neighbour's envelope.
        for slot in set(range(num_slots)) - set(indices.tolist()):
            self.assertTrue(torch.all(dst_view[slot] == 0))


if __name__ == "__main__":
    unittest.main()
