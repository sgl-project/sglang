"""Regression test for PR #25889: DeepSeekV4TokenToKVPool.register_mapping()
must clear cached_loc.

Bug scenario (pre-fix):
  During a forward pass, the first SWA layer (layer_id == start_layer) computes
  and caches `cached_loc` via translate_loc_from_full_to_swa(). If HiCache then
  loads back SWA KV from host, it calls register_mapping() to install the new
  full->swa index mapping. Before the fix, register_mapping() only stored the
  new tensor but did NOT clear cached_loc. Subsequent SWA layers (layer_id >
  start_layer) saw `cached_loc is not None` and returned the stale translation,
  silently writing KV to wrong SWA pool slots.

Fix (PR #25889): add `self.cached_loc = None` in register_mapping().

Test structure:
  - test_stale_without_fix:        shows the stale-return bug using a replica
                                   of the pre-fix logic
  - test_correct_with_fix:         verifies the fixed logic returns fresh values
  - test_actual_pool_register_mapping: exercises the real production method
                                   directly (bypassing the full __init__)

Run with:
    python -m pytest test/manual/core/test_dsv4_cached_loc_invalidation.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Minimal stub that replicates DeepSeekV4TokenToKVPool's caching pattern.
# Used to demonstrate both sides of the bug without constructing the full pool.
# ---------------------------------------------------------------------------


class _DSV4CacheStub:
    """Stripped-down replica of the caching logic in DeepSeekV4TokenToKVPool."""

    start_layer = 3

    def __init__(self, device):
        self.device = device
        self.cached_loc = None
        self.full_to_swa_index_mapping = None

    # --- pre-fix version ---
    def register_mapping_buggy(self, mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = mapping
        # BUG: cached_loc not cleared → stale on next mid-forward call

    # --- post-fix version (PR #25889) ---
    def register_mapping_fixed(self, mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = mapping
        self.cached_loc = None  # THE FIX

    def _translate(self, raw_loc: torch.Tensor) -> torch.Tensor:
        return self.full_to_swa_index_mapping[raw_loc]

    def get_swa_loc(self, layer_id: int, raw_loc: torch.Tensor) -> torch.Tensor:
        """Exact replica of set_swa_key_buffer_radix_fused caching branch."""
        if layer_id == self.start_layer or self.cached_loc is None:
            self.cached_loc = self._translate(raw_loc)
        return self.cached_loc


def _make_mapping(indices, values, size=32, device="cpu"):
    m = torch.zeros(size, dtype=torch.int64, device=device)
    m[indices] = torch.tensor(values, dtype=torch.int64, device=device)
    return m


class TestDSV4CachedLocBugAndFix(CustomTestCase):
    """Shows the pre-fix bug and verifies the post-fix behaviour."""

    def setUp(self):
        self.device = get_device()
        self.raw_loc = torch.tensor([0, 1, 2, 3], device=self.device)
        self.mapping_v1 = _make_mapping(
            [0, 1, 2, 3], [10, 11, 12, 13], device=self.device
        )
        self.mapping_v2 = _make_mapping(
            [0, 1, 2, 3], [20, 21, 22, 23], device=self.device
        )

    def test_stale_without_fix(self):
        """Without the fix, register_mapping() mid-forward leaves a stale cached_loc.

        Sequence:
          1. start_layer: cached_loc = translate(mapping_v1) = [10..13]
          2. HiCache load-back: register_mapping(mapping_v2)  ← buggy, no clear
          3. start_layer+1:     layer_id != start_layer, cached_loc is not None
                                → returns stale [10..13], NOT the correct [20..23]
        """
        stub = _DSV4CacheStub(self.device)
        stub.register_mapping_buggy(self.mapping_v1)

        # Forward pass — first SWA layer primes the cache.
        loc = stub.get_swa_loc(stub.start_layer, self.raw_loc)
        self.assertEqual(loc.tolist(), [10, 11, 12, 13])

        # HiCache load-back installs new mapping (buggy path).
        stub.register_mapping_buggy(self.mapping_v2)
        self.assertIsNotNone(stub.cached_loc, "Bug: cached_loc not cleared")

        # Next SWA layer — should use mapping_v2 but returns mapping_v1.
        loc_next = stub.get_swa_loc(stub.start_layer + 1, self.raw_loc)
        self.assertEqual(
            loc_next.tolist(),
            [10, 11, 12, 13],
            "Confirms the bug: stale cached_loc [10..13] returned instead of [20..23]",
        )

    def test_correct_with_fix(self):
        """With the fix, register_mapping() clears cached_loc; next layer recomputes.

        Same sequence as test_stale_without_fix but using the fixed register_mapping.
        """
        stub = _DSV4CacheStub(self.device)
        stub.register_mapping_fixed(self.mapping_v1)

        # Forward pass — first SWA layer primes the cache.
        loc = stub.get_swa_loc(stub.start_layer, self.raw_loc)
        self.assertEqual(loc.tolist(), [10, 11, 12, 13])

        # HiCache load-back installs new mapping (fixed path).
        stub.register_mapping_fixed(self.mapping_v2)
        self.assertIsNone(
            stub.cached_loc, "Fix: cached_loc cleared by register_mapping"
        )

        # Next SWA layer — recomputes with mapping_v2.
        loc_next = stub.get_swa_loc(stub.start_layer + 1, self.raw_loc)
        self.assertEqual(
            loc_next.tolist(),
            [20, 21, 22, 23],
            "Fix works: fresh translation [20..23] from new mapping",
        )


class TestDSV4ActualPoolRegisterMapping(CustomTestCase):
    """Exercises the production DeepSeekV4TokenToKVPool.register_mapping() directly.

    Uses __new__ to bypass the complex __init__ (which needs full GPU pool setup)
    and tests only the register_mapping / cached_loc contract.
    """

    def test_register_mapping_clears_cached_loc(self):
        device = get_device()

        # Bypass full __init__ — only the fields register_mapping touches matter.
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        old_loc = torch.tensor([10, 11, 12, 13], device=device)
        pool.cached_loc = old_loc
        pool.full_to_swa_index_mapping = None

        new_mapping = torch.arange(64, dtype=torch.int64, device=device)
        pool.register_mapping(new_mapping)

        self.assertIsNone(pool.cached_loc, "register_mapping must clear cached_loc")
        self.assertIs(pool.full_to_swa_index_mapping, new_mapping)

    def test_register_mapping_clears_none_cached_loc(self):
        """Idempotent when cached_loc is already None."""
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.cached_loc = None
        pool.full_to_swa_index_mapping = None

        mapping = torch.arange(16, dtype=torch.int64, device=get_device())
        pool.register_mapping(mapping)

        self.assertIsNone(pool.cached_loc)
        self.assertIs(pool.full_to_swa_index_mapping, mapping)


if __name__ == "__main__":
    unittest.main()
