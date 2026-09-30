"""Slot arithmetic for the QSA pending index-K ring under N compression groups.

The ring holds ``num_groups`` groups of ``compress_ratio`` slots per request, so a
request owns ``compress_ratio * num_groups`` slots and a verify window up to that
width maps every position to a distinct slot. ``num_groups == 1`` is the historical
single-group layout, and these tests pin the new arithmetic to it bit-for-bit.
"""

import unittest

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    pending_ring_slot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4


def _legacy_slots(requests, positions, ratio):
    """The pre-parameterization formula, kept as the ``num_groups == 1`` oracle."""
    return requests * ratio + positions % ratio


def _reqs_like(positions, req):
    return torch.full_like(positions, req, dtype=torch.long)


class TestPendingRingSlot(unittest.TestCase):
    def test_single_group_matches_legacy_formula(self):
        """The safety net: N=1 must reproduce the old slots exactly."""
        positions = torch.arange(-8, 48, dtype=torch.long)
        for req in (1, 2, 7):
            with self.subTest(req=req):
                requests = _reqs_like(positions, req)
                got = pending_ring_slot(
                    requests, positions, compress_ratio=RATIO, num_groups=1
                )
                want = _legacy_slots(requests, positions, RATIO)
                self.assertTrue(
                    torch.equal(got, want),
                    f"N=1 diverged for req={req}: {got.tolist()} != {want.tolist()}",
                )

    def test_window_within_capacity_is_collision_free(self):
        """A window of exactly ``ratio * num_groups`` positions maps injectively."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(100, 100 + capacity, dtype=torch.long)
                slots = pending_ring_slot(
                    _reqs_like(positions, 3),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertEqual(slots.unique().numel(), capacity)

    def test_window_beyond_capacity_reuses_slots(self):
        """The documented limit: capacity + 1 positions cannot all be distinct."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(100, 100 + capacity + 1, dtype=torch.long)
                slots = pending_ring_slot(
                    _reqs_like(positions, 3),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertLess(slots.unique().numel(), capacity + 1)

    def test_requests_own_disjoint_slot_ranges(self):
        """Request r owns exactly ``[r * ratio * N, (r + 1) * ratio * N)``."""
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                positions = torch.arange(0, 64, dtype=torch.long)
                seen = {}
                for req in (1, 2, 3):
                    slots = pending_ring_slot(
                        _reqs_like(positions, req),
                        positions,
                        compress_ratio=RATIO,
                        num_groups=num_groups,
                    )
                    self.assertGreaterEqual(int(slots.min()), req * capacity)
                    self.assertLess(int(slots.max()), (req + 1) * capacity)
                    seen[req] = set(slots.tolist())
                self.assertFalse(seen[1] & seen[2])
                self.assertFalse(seen[2] & seen[3])

    def test_group_boundary_neighbours_land_in_adjacent_groups(self):
        """``k * ratio +/- 1`` straddle a group boundary, not the same slot."""
        for num_groups in (2, 4):
            with self.subTest(num_groups=num_groups):
                boundary = RATIO * num_groups * 3  # a group boundary, far from 0
                positions = torch.tensor(
                    [boundary - 1, boundary, boundary + 1], dtype=torch.long
                )
                slots = pending_ring_slot(
                    _reqs_like(positions, 1),
                    positions,
                    compress_ratio=RATIO,
                    num_groups=num_groups,
                )
                self.assertEqual(slots.unique().numel(), 3)


class TestBuildPendingRingSlots(unittest.TestCase):
    def _build(self, positions, reqs, lengths, num_groups, is_extend):
        positions = torch.tensor(positions, dtype=torch.long)
        return build_pending_ring_slots(
            token_to_batch_idx=torch.arange(positions.numel()),
            req_pool_indices=torch.tensor(reqs, dtype=torch.long),
            sequence_lengths=torch.tensor(lengths, dtype=torch.long),
            logical_positions=positions,
            compress_ratio=RATIO,
            num_groups=num_groups,
            is_extend=is_extend,
        )

    def test_defaults_to_single_group(self):
        """Omitting ``num_groups`` keeps the historical layout."""
        positions = [0, 1, 2, 3]
        got = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(4),
            req_pool_indices=torch.tensor([1, 1, 1, 1]),
            sequence_lengths=torch.tensor([8, 8, 8, 8]),
            logical_positions=torch.tensor(positions),
            compress_ratio=RATIO,
            is_extend=False,
        )
        want = _legacy_slots(torch.tensor([1, 1, 1, 1]), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(got, want))

    def test_extend_dump_stays_in_the_inert_region(self):
        """Non-pending extend tokens dump into rows ``[0, ratio)``.

        Request slot 0 is never allocated, so that region is inert. It must stay
        disjoint from every allocated request's range at any N.
        """
        # request 1, sequence length 8 -> pending tail starts at position 8.
        positions = [0, 3, 4, 8, 9]
        lengths = [8] * len(positions)
        reqs = [1] * len(positions)
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                slots = self._build(positions, reqs, lengths, num_groups, True)
                capacity = RATIO * num_groups
                dumped = slots[:3]  # positions 0, 3, 4 are before the pending tail
                pending = slots[3:]  # positions 8, 9 are the pending group
                self.assertTrue(bool((dumped < RATIO).all()))
                self.assertTrue(bool((pending >= capacity).all()))

    def test_extend_pending_tail_matches_plain_formula(self):
        """Pending tokens use the same slots whether or not extend is set."""
        positions = [8, 9, 10, 11]
        for num_groups in (1, 2, 4):
            with self.subTest(num_groups=num_groups):
                extended = self._build(positions, [1] * 4, [8] * 4, num_groups, True)
                plain = self._build(positions, [1] * 4, [8] * 4, num_groups, False)
                self.assertTrue(torch.equal(extended, plain))


class TestBuildGroupRingSlots(unittest.TestCase):
    def _build(self, group_ends, reqs, num_groups):
        return build_group_ring_slots(
            req_pool_indices=torch.tensor(reqs, dtype=torch.long),
            group_end_positions=torch.tensor(group_ends, dtype=torch.long),
            sequence_ids=torch.arange(len(group_ends)),
            compress_ratio=RATIO,
            num_groups=num_groups,
        )

    def test_members_are_oldest_first(self):
        """Column k holds the k-th oldest member of the group."""
        # group ending at 7 spans positions 4..7, oldest first.
        slots = self._build([7], [1], num_groups=1)
        self.assertEqual(slots.shape, (1, RATIO))
        positions = [4, 5, 6, 7]
        want = _legacy_slots(torch.tensor([1] * 4), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(slots[0], want))

    def test_all_members_share_one_ring_group(self):
        """Members of one group share a group index even when the end is unaligned.

        The extend producer always emits ``group_end = blocks * ratio + ratio - 1``
        (aligned), but the graph producer feeds ``lengths - 1``, which is only
        aligned on the boundary rows. Deriving the group from ``group_end`` rather
        than from each member's own position keeps a group together either way.
        """
        for num_groups in (2, 4):
            with self.subTest(num_groups=num_groups):
                capacity = RATIO * num_groups
                group_ends = [RATIO * 4 - 1, RATIO * 4, RATIO * 4 + 1]
                slots = self._build(group_ends, [1] * 3, num_groups)
                for row, end in enumerate(group_ends):
                    group = (end // RATIO) % num_groups
                    lo = capacity + group * RATIO
                    hi = lo + RATIO
                    self.assertTrue(
                        bool(((slots[row] >= lo) & (slots[row] < hi)).all()),
                        f"row {row} (end={end}) straddled ring groups: "
                        f"{slots[row].tolist()}",
                    )

    def test_single_group_matches_legacy_formula(self):
        """N=1 regression oracle for the group builder."""
        group_ends = [7, 11, 15]
        slots = self._build(group_ends, [1, 2, 3], num_groups=1)
        for row, end in enumerate(group_ends):
            positions = list(range(end - RATIO + 1, end + 1))
            want = _legacy_slots(
                torch.tensor([row + 1] * RATIO), torch.tensor(positions), RATIO
            )
            self.assertTrue(torch.equal(slots[row], want))

    def test_members_clamp_at_zero(self):
        """A group at the very start clamps negative members to position 0."""
        slots = self._build([1], [1], num_groups=1)
        positions = [0, 0, 0, 1]
        want = _legacy_slots(torch.tensor([1] * 4), torch.tensor(positions), RATIO)
        self.assertTrue(torch.equal(slots[0], want))


if __name__ == "__main__":
    unittest.main()
