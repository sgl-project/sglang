import unittest

import torch

from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4
PAGE = 16
LAYERS = 2
HEADS, HDIM = 1, 8
SLOTS = 8 * PAGE
REQ_ROWS = 6


class _FullKV:
    """Stands in for full_kv_pool; the QSA state is what is under test."""

    def __init__(self):
        self.k = torch.zeros(SLOTS, 4)

    def get_cpu_copy(self, indices, req_pool_index=None):
        return self.k[indices].clone()

    def load_cpu_copy(self, data, indices, req_pool_index=None):
        self.k[indices] = data


def _pool() -> QSATokenToKVPool:
    # __init__ needs a Mamba pool and device buffers; only the QSA layout matters here.
    pool = object.__new__(QSATokenToKVPool)
    pool.full_kv_pool = _FullKV()
    pool.mamba_pool = None
    pool._mamba_translate = lambda ids: ids
    pool.qsa_compress_ratio = RATIO
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(SLOTS // RATIO, HEADS, HDIM) for _ in range(LAYERS)
    ]
    pool.qsa_key_state_buffer_pool = [
        torch.zeros(REQ_ROWS * RATIO, HEADS, HDIM) for _ in range(LAYERS)
    ]
    pool.qsa_rope_position_buffer = torch.zeros(REQ_ROWS * RATIO, 3, dtype=torch.int64)
    return pool


def _pages(*pages):
    return torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in pages])


def _fill(pool, slots, row, value):
    for buf in pool.qsa_compressed_k_buffer_pool:
        buf[torch.unique(slots // RATIO)] = value
    ring = torch.arange(row * RATIO, (row + 1) * RATIO)
    for buf in pool.qsa_key_state_buffer_pool:
        buf[ring] = value
    pool.qsa_rope_position_buffer[ring] = int(value)


class TestQSARetractionBackup(CustomTestCase):
    def test_resumed_request_keeps_its_qsa_state(self):
        """A request retracted from rows/pages that another request then reuses
        must resume with its own compressed index keys and pending ring."""
        pool = _pool()
        old_slots = _pages(1, 2)
        _fill(pool, old_slots, row=1, value=1.0)
        backup = pool.get_cpu_copy(old_slots, req_pool_index=1)

        new_slots = _pages(4, 5)
        _fill(pool, new_slots, row=3, value=2.0)  # left by another request
        pool.load_cpu_copy(backup, new_slots, req_pool_index=3)

        ring = torch.arange(3 * RATIO, 4 * RATIO)
        for buf in pool.qsa_compressed_k_buffer_pool:
            self.assertTrue(torch.all(buf[torch.unique(new_slots // RATIO)] == 1.0))
        for buf in pool.qsa_key_state_buffer_pool:
            self.assertTrue(torch.all(buf[ring] == 1.0))
        self.assertTrue(torch.all(pool.qsa_rope_position_buffer[ring] == 1))

    def test_groups_follow_request_order_when_pages_are_reordered(self):
        pool = _pool()
        old_slots = _pages(1, 5)
        for value, page in ((10.0, 1), (11.0, 5)):
            for buf in pool.qsa_compressed_k_buffer_pool:
                buf[torch.unique(_pages(page) // RATIO)] = value
        backup = pool.get_cpu_copy(old_slots, req_pool_index=1)

        pool.load_cpu_copy(backup, _pages(7, 2), req_pool_index=2)

        for value, page in ((10.0, 7), (11.0, 2)):
            for buf in pool.qsa_compressed_k_buffer_pool:
                self.assertTrue(
                    torch.all(buf[torch.unique(_pages(page) // RATIO)] == value)
                )

    def test_partial_last_group_and_distinct_ring_values(self):
        """A request whose length is not a multiple of the ratio, moved to a
        different row and reordered pages; every group, ring member and RoPE
        axis has its own value."""
        pool = _pool()
        n = PAGE + RATIO - 1  # one full page plus a partial group
        old_slots = torch.cat([_pages(1), _pages(5)])[:n]
        new_slots = torch.cat([_pages(4), _pages(2)])[:n]
        old_groups = old_slots[::RATIO] // RATIO
        new_groups = new_slots[::RATIO] // RATIO
        for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
            for g, slot in enumerate(old_groups):
                buf[slot] = 100 * layer + g + 1
        ring_old = torch.arange(1 * RATIO, 2 * RATIO)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            buf[ring_old] = (
                (torch.arange(RATIO) + 50 * layer + 1).view(-1, 1, 1).float()
            )
        pool.qsa_rope_position_buffer[ring_old] = torch.arange(RATIO * 3).view(RATIO, 3)

        backup = pool.get_cpu_copy(old_slots, req_pool_index=1)
        pool.load_cpu_copy(backup, new_slots, req_pool_index=4)

        for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
            for g, slot in enumerate(new_groups):
                self.assertTrue(torch.all(buf[slot] == 100 * layer + g + 1))
        ring_new = torch.arange(4 * RATIO, 5 * RATIO)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            want = (torch.arange(RATIO) + 50 * layer + 1).view(-1, 1, 1).float()
            self.assertTrue(torch.equal(buf[ring_new], want.expand_as(buf[ring_new])))
        self.assertTrue(
            torch.equal(
                pool.qsa_rope_position_buffer[ring_new],
                torch.arange(RATIO * 3).view(RATIO, 3),
            )
        )

    def test_pp_stage_without_qsa_layers(self):
        """With no local QSA layer there are no QSA buffers and no ring to carry."""
        pool = _pool()
        pool.qsa_compressed_k_buffer_pool = []
        pool.qsa_key_state_buffer_pool = []
        ring = torch.arange(2 * RATIO, 3 * RATIO)
        pool.qsa_rope_position_buffer[ring] = 7

        backup = pool.get_cpu_copy(_pages(1), req_pool_index=1)
        pool.load_cpu_copy(backup, _pages(4), req_pool_index=2)

        self.assertIsNone(backup[1][1])
        self.assertTrue(torch.all(pool.qsa_rope_position_buffer[ring] == 7))

    def test_empty_indices(self):
        pool = _pool()
        empty = torch.empty(0, dtype=torch.int64)
        pool.load_cpu_copy(
            pool.get_cpu_copy(empty, req_pool_index=1), empty, req_pool_index=2
        )


if __name__ == "__main__":
    unittest.main()
