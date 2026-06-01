"""Crash regression for PR #25889: stale cached_loc after register_mapping().

Bug:
  DeepSeekV4TokenToKVPool caches the full→SWA index translation in
  self.cached_loc (when SGLANG_OPT_CACHE_SWA_TRANSLATION=True).
  register_mapping() was replacing full_to_swa_index_mapping without
  clearing cached_loc, so a subsequent set_swa_key_buffer_radix_fused
  call would use stale SWA indices and, if those indices exceed the
  current pool size, raise a RuntimeError (OOB tensor access).

Crash scenario reproduced here:
  Pass 1 (large SWA pool, size=8): first SWA layer primes
    cached_loc = [4, 5, 6, 7].
  register_mapping() called with new mapping (pre-fix: cache not cleared).
  Pass 2 (smaller SWA pool, size=4): same SWA layer finds cached_loc is
    not None → uses stale [4, 5, 6, 7] → OOB write on size-4 pool →
    RuntimeError.

Fix: register_mapping() sets self.cached_loc = None so the next call
recomputes with the fresh mapping.

Run with:
    python -m pytest test/registered/unit/mem_cache/test_dsv4_stale_loc_crash.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_NUM_HEADS = 2
_HEAD_DIM = 8
_SWA_LARGE = 8  # size-8 pool used during pass 1
_SWA_SMALL = 4  # size-4 pool used during pass 2 (simulates post-HiCache loadback)


class _SWAPoolMock:
    """Minimal SWA pool mock whose set_key_buffer_fused does a real tensor write.

    A write with OOB swa_loc raises RuntimeError, reproducing the crash.
    """

    def __init__(self, size: int):
        self.buf = torch.zeros(size, _NUM_HEADS, _HEAD_DIM)

    def set_key_buffer_fused(
        self, local_layer_id: int, swa_loc: torch.Tensor, cache_k: torch.Tensor
    ) -> None:
        n = swa_loc.numel()
        self.buf[swa_loc.long()] = cache_k[:n].reshape(n, _NUM_HEADS, _HEAD_DIM)


def _build_pool(
    mapping: torch.Tensor,
    swa_pool: _SWAPoolMock,
    start_layer: int = 0,
) -> DeepSeekV4TokenToKVPool:
    """Create a minimal DeepSeekV4TokenToKVPool via __new__, bypassing __init__."""
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.cached_loc = None
    pool._should_cache_swa = True
    pool.start_layer = start_layer
    pool.full_to_swa_index_mapping = mapping
    pool.swa_kv_pool = swa_pool
    # _swa_local_layer_id: map global SWA layer id → local index 0 for this test.
    pool._swa_local_layer_id = lambda lid: 0
    return pool


def _mapping(indices: list, size: int = 16) -> torch.Tensor:
    m = torch.full((size,), -1, dtype=torch.int64)
    for i, v in enumerate(indices):
        m[i] = v
    return m


class TestDSV4StaleLocCrash(CustomTestCase):
    """
    Two paired tests that together constitute the crash regression for #25889.

    test_crash_without_fix:  reproduces the RuntimeError that occurs when
                             register_mapping() does NOT clear cached_loc.
    test_fix_prevents_crash: same sequence with the fixed register_mapping()
                             → no error, correct SWA slots written.
    """

    def setUp(self):
        # raw_loc: full-pool indices for 4 tokens.
        self.raw_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        # cache_k: synthetic key data for 4 tokens.
        self.cache_k = torch.ones(4, _NUM_HEADS, _HEAD_DIM)
        # SWA layer id > start_layer (0), so cached_loc is only reset by the
        # "is None" branch, NOT by the "layer_id == start_layer" branch.
        self.swa_layer_id = 1
        # mapping_v1: raw_loc [0,1,2,3] → SWA slots [4,5,6,7] (valid for size-8 pool).
        self.mapping_v1 = _mapping([4, 5, 6, 7])
        # mapping_v2: same raw_loc → SWA slots [0,1,2,3] (valid for size-4 pool).
        self.mapping_v2 = _mapping([0, 1, 2, 3])

    def test_crash_without_fix(self):
        """Without the fix, stale cached_loc [4,5,6,7] causes RuntimeError on
        the size-4 pool used in pass 2."""
        large_pool = _SWAPoolMock(_SWA_LARGE)
        pool = _build_pool(self.mapping_v1, large_pool, start_layer=0)

        # Pass 1: swa_layer_id=1, cached_loc is None → compute and cache [4,5,6,7].
        pool.set_swa_key_buffer_radix_fused(
            self.swa_layer_id, self.raw_loc, self.cache_k
        )
        self.assertEqual(pool.cached_loc.tolist(), [4, 5, 6, 7], "Pass 1: cache primed")

        # PRE-FIX register_mapping: replace mapping WITHOUT clearing cached_loc.
        pool.full_to_swa_index_mapping = self.mapping_v2

        # Pass 2 on a smaller pool (size=4). cached_loc is still [4,5,6,7].
        # OOB write → RuntimeError.
        pool.swa_kv_pool = _SWAPoolMock(_SWA_SMALL)
        with self.assertRaises((RuntimeError, IndexError)):
            pool.set_swa_key_buffer_radix_fused(
                self.swa_layer_id, self.raw_loc, self.cache_k
            )

    def test_fix_prevents_crash(self):
        """With the fix, register_mapping() clears cached_loc. Pass 2 recomputes
        [0,1,2,3] from mapping_v2 and writes to the correct size-4 pool slots."""
        large_pool = _SWAPoolMock(_SWA_LARGE)
        pool = _build_pool(self.mapping_v1, large_pool, start_layer=0)

        # Pass 1: prime cache → cached_loc = [4,5,6,7].
        pool.set_swa_key_buffer_radix_fused(
            self.swa_layer_id, self.raw_loc, self.cache_k
        )
        self.assertEqual(pool.cached_loc.tolist(), [4, 5, 6, 7])

        # FIXED register_mapping: clears cached_loc.
        pool.register_mapping(self.mapping_v2)
        self.assertIsNone(
            pool.cached_loc, "Fix: cached_loc cleared by register_mapping"
        )

        # Pass 2 on the smaller pool: cached_loc is None → recompute with mapping_v2.
        small_pool = _SWAPoolMock(_SWA_SMALL)
        pool.swa_kv_pool = small_pool
        pool.set_swa_key_buffer_radix_fused(
            self.swa_layer_id, self.raw_loc, self.cache_k
        )

        self.assertEqual(
            pool.cached_loc.tolist(), [0, 1, 2, 3], "Fresh indices after fix"
        )
        # Verify data landed in the correct SWA slots [0-3], not the stale [4-7].
        self.assertTrue(
            small_pool.buf[0:4].abs().sum().item() > 0,
            "Correct SWA slots 0-3 received data",
        )


if __name__ == "__main__":
    unittest.main()
