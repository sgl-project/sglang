import unittest

import torch

from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

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


class TestQSARetractionBackup(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
