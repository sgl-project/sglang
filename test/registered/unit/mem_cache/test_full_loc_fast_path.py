# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routing tests for the composite write paths (`UnifiedSWAKVPool`,
`HybridLinearKVPool`).

All write-location info travels in the attention metadata (`KVWriteLoc`); the
pools hold none and never translate — the write loc reaching `set_kv_buffer` is
always PHYSICAL. Two routing contracts are pinned here:

1. Full-attention. The full-physical loc is carried in `KVWriteLoc.full_loc`
   (from `ForwardBatch.out_cache_loc_full_physical`) and written directly.
   `UnifiedSWAKVPool` asserts it's present (the unified memory pool always precomputes
   it); `HybridLinearKVPool` falls back to `loc` for a static (non-shared) pool,
   where `loc` is itself already physical.
2. SWA. The swa-physical loc rides the backend `swa_out_cache_loc` rail
   (`KVWriteLoc.swa_loc`) and is written directly.

Pure dispatch tests: the inner sub-pools are recording stubs, so no GPU / real
buffers are needed (CPU CI).

    python -m pytest test/registered/unit/mem_cache/test_full_loc_fast_path.py -v
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _loc_info(virtual_loc, swa_phys=None, full_phys=None):
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    return KVWriteLoc(virtual_loc, swa_phys, full_phys)


class _RecordingPool:
    """Stub sub-pool that records the `loc` and kwargs passed to `set_kv_buffer`."""

    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        self.calls.append((loc, kwargs))


class TestUnifiedSWARouting(unittest.TestCase):
    """`UnifiedSWAKVPool.set_kv_buffer` routing: full layers write the full-physical
    `full_loc`; SWA layers write the swa-physical `swa_loc`. Both come from the
    write metadata; the pool never translates."""

    def _make_bare_pool(self):
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        # Bypass the heavy __init__; set only the attributes set_kv_buffer reads.
        pool = object.__new__(UnifiedSWAKVPool)
        pool.full_kv_pool = _RecordingPool()
        pool.swa_kv_pool = _RecordingPool()
        # layer 0 -> full attention; layer 1 -> SWA. (pool_layer_id, is_swa)
        pool.layers_mapping = {0: (0, False), 1: (0, True)}
        return pool

    def test_full_layer_writes_full_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        swa_phys = torch.tensor([1, 2, 0], dtype=torch.int64)
        full_phys = torch.tensor([3, 4, 5], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)  # full layer
        pool.set_kv_buffer(
            layer,
            _loc_info(virtual_loc, swa_phys, full_phys),
            torch.zeros(3, 4, 8),
            torch.zeros(3, 4, 8),
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, kwargs = pool.full_kv_pool.calls[0]
        # Forward the full-physical tensor from the write metadata, NOT the
        # virtual loc. No `already_physical` — the pool only ever gets physical.
        self.assertIs(forwarded, full_phys)
        self.assertIsNot(forwarded, virtual_loc)
        self.assertNotIn("already_physical", kwargs)

    def test_full_layer_requires_full_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        swa_phys = torch.tensor([1, 2, 0], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        # No full_loc precomputed -> fail loud (the unified memory pool must precompute
        # out_cache_loc_full_physical) rather than write a virtual loc as physical.
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(
                layer,
                _loc_info(virtual_loc, swa_phys),
                torch.zeros(3, 4, 8),
                torch.zeros(3, 4, 8),
            )

    def test_swa_layer_writes_swa_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        swa_phys = torch.tensor([1, 2, 0], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=1)  # SWA layer
        pool.set_kv_buffer(
            layer,
            _loc_info(virtual_loc, swa_phys),
            torch.zeros(3, 4, 8),
            torch.zeros(3, 4, 8),
        )

        self.assertEqual(len(pool.swa_kv_pool.calls), 1)
        forwarded, kwargs = pool.swa_kv_pool.calls[0]
        # SWA write rides the backend rail: forward the swa-physical loc directly.
        self.assertIs(forwarded, swa_phys)
        self.assertNotIn("already_physical", kwargs)
        # Full pool untouched for an SWA layer.
        self.assertEqual(len(pool.full_kv_pool.calls), 0)

    def test_swa_layer_requires_swa_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=1)  # SWA layer
        # No swa_loc bundled -> the rail contract is violated; must assert
        # rather than silently writing wrong (un-translated) locations.
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(
                layer,
                _loc_info(virtual_loc, None),
                torch.zeros(3, 4, 8),
                torch.zeros(3, 4, 8),
            )


class TestHybridLinearFullLocRouting(unittest.TestCase):
    """`HybridLinearKVPool.set_kv_buffer` (non-MLA) writes the full-physical
    `full_loc` from the write metadata when present (unified memory pool), else the
    already-physical `loc` (static pool). No translate, no `already_physical`."""

    def _make_bare_pool(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _RecordingPool()
        pool.use_mla = False
        pool.full_attention_layer_id_mapping = {0: 0}
        return pool

    def test_writes_full_loc_from_write_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        full_phys = torch.tensor([2, 3, 4], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(virtual_loc, full_phys=full_phys),
            torch.zeros(3, 4, 8),
            torch.zeros(3, 4, 8),
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, kwargs = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, full_phys)
        self.assertIsNot(forwarded, virtual_loc)
        self.assertNotIn("already_physical", kwargs)

    def test_falls_back_to_loc_when_absent(self):
        # Static (non-shared) pool: no full_loc bundled; `loc` is already
        # physical, so write it directly.
        pool = self._make_bare_pool()
        phys_loc = torch.tensor([7, 8, 9], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(phys_loc),
            torch.zeros(3, 4, 8),
            torch.zeros(3, 4, 8),
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, kwargs = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, phys_loc)
        self.assertNotIn("already_physical", kwargs)


class _RecordingMLAPool(_RecordingPool):
    """Also records the model-level MLA entry points."""

    def __init__(self):
        super().__init__()
        self.mla_set_calls = []
        self.mla_get_calls = []

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        self.mla_set_calls.append(loc)

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        self.mla_get_calls.append(loc)
        return None, None


class TestHybridLinearMLARouting(unittest.TestCase):
    """MLA-side routing contracts of `HybridLinearKVPool`:

    - `set_kv_buffer` (MLA branch) mirrors the MHA branch — write the
      pre-translated `KVWriteLoc.full_loc` when present (unified pool, where it
      carries the DENSE loc), else the raw `loc` (static pool, already physical).
    - `set_mla_kv_buffer` / `get_mla_kv_buffer` receive VIRTUAL locs and apply
      `_full_translate` exactly once (identity for a static pool)."""

    def _make_bare_pool(self, translate=None):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _RecordingMLAPool()
        pool.use_mla = True
        pool.full_attention_layer_id_mapping = {0: 0}
        pool._full_translate = translate if translate is not None else (lambda x: x)
        return pool

    def test_mla_writes_full_loc_from_write_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        dense_phys = torch.tensor([21, 24, 27], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(virtual_loc, full_phys=dense_phys),
            torch.zeros(3, 1, 8),
            None,
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, _ = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, dense_phys)
        self.assertIsNot(forwarded, virtual_loc)

    def test_mla_falls_back_to_loc_when_absent(self):
        pool = self._make_bare_pool()
        phys_loc = torch.tensor([7, 8, 9], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(phys_loc),
            torch.zeros(3, 1, 8),
            None,
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, _ = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, phys_loc)

    def test_set_mla_kv_buffer_translates_exactly_once(self):
        calls = []

        def translate(ids):
            calls.append(ids)
            return ids + 100

        pool = self._make_bare_pool(translate=translate)
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        layer = types.SimpleNamespace(layer_id=0)

        pool.set_mla_kv_buffer(
            layer, virtual_loc, torch.zeros(3, 1, 6), torch.zeros(3, 1, 2)
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(pool.full_kv_pool.mla_set_calls), 1)
        self.assertTrue(
            torch.all(pool.full_kv_pool.mla_set_calls[0] == virtual_loc + 100)
        )

    def test_get_mla_kv_buffer_translates_exactly_once(self):
        calls = []

        def translate(ids):
            calls.append(ids)
            return ids + 100

        pool = self._make_bare_pool(translate=translate)
        virtual_loc = torch.tensor([4, 5], dtype=torch.int64)
        layer = types.SimpleNamespace(layer_id=0)

        pool.get_mla_kv_buffer(layer, virtual_loc)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(pool.full_kv_pool.mla_get_calls), 1)
        self.assertTrue(
            torch.all(pool.full_kv_pool.mla_get_calls[0] == virtual_loc + 100)
        )


if __name__ == "__main__":
    unittest.main()
