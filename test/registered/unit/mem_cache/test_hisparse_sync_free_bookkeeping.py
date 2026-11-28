"""Lossless checks for the sync-free HiSparse scheduler-side swap-in bookkeeping.

The overlap scheduler prepares decode step N+1 while forward N runs, so any
device->host sync on that path stalls the CPU until the in-flight graph ends.
The DSv4 branch of ``HiSparseCoordinator.map_last_loc_to_buffer`` used
``torch.any`` + boolean-mask compaction; it is now computed for every row,
inactive rows rewriting the value already stored. The backup-position selection
of ``_eager_backup_previous_token`` moved into ``_select_backup_positions``
(its list goes to the device through a pinned non-blocking copy).

These tests keep the previous implementations as references and assert the new
code yields the identical backup positions / flag transitions and bit-identical
``req_device_buffer_token_locs`` and ``full_to_hisparse_device_index_mapping``.
They run on CPU: the DSv4 update is driven through the real coordinator method
bound to a state object built without the GPU pools.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    _select_backup_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

COMPRESS_RATIO = 4
PAGE_SIZE = 4
LAYER_NUM = 3


# --------------------------------------------------------------------------
# Pre-fix reference implementations (verbatim logic).
# --------------------------------------------------------------------------
def _old_select_backup_positions(
    seq_lens_cpu, req_pool_indices_cpu, skip_first_backup, compress_ratio
):
    backup_indices = []
    for i in range(len(seq_lens_cpu)):
        req_idx = int(req_pool_indices_cpu[i])
        if skip_first_backup[req_idx]:
            skip_first_backup[req_idx] = False
            continue
        if (int(seq_lens_cpu[i]) - 1) % compress_ratio == 0:
            backup_indices.append(i)
    return backup_indices


def _old_map_last_loc_dsv4(state, seq_lens, out_cache_loc, req_pool_indices):
    active_reqs = seq_lens % state.compress_ratio == 0
    if not torch.any(active_reqs):
        return
    active_seq_lens = seq_lens[active_reqs]
    active_out_cache_loc = out_cache_loc[active_reqs]
    active_req_pool_indices = req_pool_indices[active_reqs]
    compressed_seq_lens = active_seq_lens // state.compress_ratio
    reserved_positions = (compressed_seq_lens - 1).clamp(max=state.device_buffer_size)
    reserved_buffer_loc = state.req_to_device_buffer[
        active_req_pool_indices, reserved_positions
    ]
    state.req_device_buffer_token_locs[
        :, active_req_pool_indices, state.device_buffer_size
    ] = reserved_buffer_loc.to(torch.int32)
    compressed_locs = state.token_to_kv_pool_allocator.get_last_loc_compressed(
        active_out_cache_loc
    )
    state.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
        reserved_buffer_loc
    )


# --------------------------------------------------------------------------
# Coordinator state without GPU pools.
# --------------------------------------------------------------------------
def _make_state(*, device_buffer_size, num_slots, num_groups, seed):
    """A ``HiSparseCoordinator`` instance carrying only the fields the DSv4
    update reads, with randomized (but reproducible) table contents."""
    g = torch.Generator().manual_seed(seed)
    padded = device_buffer_size + PAGE_SIZE
    st = HiSparseCoordinator.__new__(HiSparseCoordinator)
    st.compress_ratio = COMPRESS_RATIO
    st.device_buffer_size = device_buffer_size
    st.req_to_device_buffer = torch.randint(
        1, 1 << 20, (num_slots, padded), dtype=torch.int64, generator=g
    )
    st.req_device_buffer_token_locs = torch.randint(
        -1, 1 << 20, (LAYER_NUM, num_slots, padded), dtype=torch.int32, generator=g
    )
    # DSv4 allocator: compressed index of a full loc is (loc - 3) // ratio and
    # the mapping carries a trailing -1 sentinel (index -1 lookups).
    st.token_to_kv_pool_allocator = SimpleNamespace(
        get_last_loc_compressed=lambda locs: (locs - 3) // COMPRESS_RATIO
    )
    mapping = torch.cat(
        [
            torch.randint(0, 1 << 20, (num_groups,), dtype=torch.int64, generator=g),
            torch.tensor([-1], dtype=torch.int64),
        ]
    )
    st.mem_pool_device = SimpleNamespace(full_to_hisparse_device_index_mapping=mapping)
    return st


def _clone_state(st):
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.__dict__.update(st.__dict__)
    c.req_to_device_buffer = st.req_to_device_buffer.clone()
    c.req_device_buffer_token_locs = st.req_device_buffer_token_locs.clone()
    c.mem_pool_device = SimpleNamespace(
        full_to_hisparse_device_index_mapping=(
            st.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
        )
    )
    return c


def _batch(seq_lens, groups, req_slots):
    """Decode batch: request ``i`` sits in compressed group ``groups[i]`` and its
    newest token occupies full loc ``4 * group + (seq_len - 1) % 4``, so an
    active row (``seq_len % 4 == 0``) maps to its own group while an inactive
    row maps to the previous group (possibly another request's, or -1)."""
    seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    groups = torch.tensor(groups, dtype=torch.int64)
    out_cache_loc = groups * COMPRESS_RATIO + (seq_lens - 1) % COMPRESS_RATIO
    return seq_lens, out_cache_loc, torch.tensor(req_slots, dtype=torch.int64)


class TestDsv4MapLastLocLossless(CustomTestCase):
    NUM_SLOTS = 8
    NUM_GROUPS = 64
    BUFFER_SIZES = (16, 8)

    def _assert_same(self, device_buffer_size, seq_lens, groups, req_slots, seed=0):
        base = _make_state(
            device_buffer_size=device_buffer_size,
            num_slots=self.NUM_SLOTS,
            num_groups=self.NUM_GROUPS,
            seed=seed,
        )
        old, new = _clone_state(base), _clone_state(base)
        sl, loc, rp = _batch(seq_lens, groups, req_slots)
        _old_map_last_loc_dsv4(old, sl, loc, rp)
        HiSparseCoordinator._map_last_loc_to_buffer_dsv4(
            new, seq_lens=sl, out_cache_loc=loc, req_pool_indices=rp
        )
        self.assertTrue(
            torch.equal(
                old.req_device_buffer_token_locs, new.req_device_buffer_token_locs
            )
        )
        self.assertTrue(
            torch.equal(
                old.mem_pool_device.full_to_hisparse_device_index_mapping,
                new.mem_pool_device.full_to_hisparse_device_index_mapping,
            )
        )
        # Untouched tables stay untouched.
        self.assertTrue(
            torch.equal(base.req_to_device_buffer, new.req_to_device_buffer)
        )
        return base, new

    def test_all_active(self):
        for size in self.BUFFER_SIZES:
            self._assert_same(size, [4, 8, 400, 36], [3, 7, 9, 20], [0, 5, 2, 7])

    def test_none_active_is_a_no_op(self):
        for size in self.BUFFER_SIZES:
            base, new = self._assert_same(
                size, [5, 6, 7, 1, 2, 3], [3, 7, 9, 0, 1, 40], [0, 5, 2, 7, 1, 3]
            )
            self.assertTrue(
                torch.equal(
                    base.req_device_buffer_token_locs, new.req_device_buffer_token_locs
                )
            )
            self.assertTrue(
                torch.equal(
                    base.mem_pool_device.full_to_hisparse_device_index_mapping,
                    new.mem_pool_device.full_to_hisparse_device_index_mapping,
                )
            )

    def test_mixed_including_aliasing_inactive_rows(self):
        # Row 1 (inactive, group 8) maps to group 7 == row 0's active target;
        # row 3 (inactive, seq_len < ratio, group 0) maps to index -1 (sentinel).
        for size in self.BUFFER_SIZES:
            self._assert_same(
                size,
                [32, 33, 4, 2, 800, 13],
                [7, 8, 20, 0, 30, 31],
                [4, 1, 6, 0, 2, 7],
            )

    def test_single_request(self):
        for size in self.BUFFER_SIZES:
            self._assert_same(size, [12], [5], [3])
            self._assert_same(size, [13], [5], [3])

    def test_empty_batch(self):
        for size in self.BUFFER_SIZES:
            self._assert_same(size, [], [], [])

    def test_randomized_steps(self):
        g = torch.Generator().manual_seed(1234)
        for step in range(40):
            n = int(torch.randint(1, self.NUM_SLOTS + 1, (1,), generator=g))
            req_slots = torch.randperm(self.NUM_SLOTS, generator=g)[:n].tolist()
            groups = torch.randperm(self.NUM_GROUPS, generator=g)[:n].tolist()
            seq_lens = torch.randint(1, 200, (n,), generator=g).tolist()
            size = self.BUFFER_SIZES[step % len(self.BUFFER_SIZES)]
            self._assert_same(size, seq_lens, groups, req_slots, seed=step)


class TestBackupPositionSelection(CustomTestCase):
    def _run_both(self, seq_lens, req_slots, skip_pattern, num_slots=8):
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        req_cpu = torch.tensor(req_slots, dtype=torch.int64)
        old_flags = [False] * num_slots
        for s in skip_pattern:
            old_flags[s] = True
        new_flags = list(old_flags)
        old_pos = _old_select_backup_positions(
            seq_lens_cpu, req_cpu, old_flags, COMPRESS_RATIO
        )
        new_pos = _select_backup_positions(
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices_cpu=req_cpu,
            skip_first_backup=new_flags,
            compress_ratio=COMPRESS_RATIO,
        )
        self.assertEqual(old_pos, new_pos)
        self.assertEqual(old_flags, new_flags)
        return new_pos, new_flags

    def test_patterns(self):
        seq_lens = [5, 6, 9, 13, 1, 17, 18, 21]
        req_slots = [0, 1, 2, 3, 4, 5, 6, 7]
        for skip in ([], [0], [2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [7], [4, 1]):
            self._run_both(seq_lens, req_slots, skip)

    def test_skip_flag_flips_exactly_once(self):
        # First decode step after staging: skipped and the flag is cleared;
        # the next aligned step backs up. A skipped request's flag is cleared
        # even when the step was not aligned.
        pos, flags = self._run_both([9, 10], [3, 5], [3, 5])
        self.assertEqual(pos, [])
        self.assertEqual(flags, [False] * 8)
        pos, flags = self._run_both([13, 14], [3, 5], [])
        self.assertEqual(pos, [0])

    def test_untouched_slots_keep_their_flag(self):
        # A staged request not in this batch must still skip its first step.
        _, flags = self._run_both([9], [0], [0, 6])
        self.assertEqual(flags[6], True)
        self.assertEqual(flags[0], False)

    def test_empty_batch(self):
        pos, _ = self._run_both([], [], [2])
        self.assertEqual(pos, [])


if __name__ == "__main__":
    unittest.main()
