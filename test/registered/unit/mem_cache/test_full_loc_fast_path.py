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
   (from `ForwardMetadata.out_cache_loc_full_physical`) and written directly.
   `UnifiedSWAKVPool` asserts it's present (the unified memory pool always precomputes
   it); `HybridLinearKVPool` falls back to `loc` for a static (non-shared) pool,
   where `loc` is itself already physical.
2. SWA. The swa-physical loc rides the backend `swa_out_cache_loc` slot
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
    """`UnifiedSWAKVPool.set_kv_buffer` routing: full layers write `full_loc`
    when present (triton's capture-stable buffer), else the rebound generic
    `loc` -- the same id space once the loc is rebound; SWA layers write the swa-physical
    `swa_loc`, which has no fallback (a different id space). The pool never
    translates."""

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

    def test_full_layer_falls_back_to_generic_loc(self):
        """Bug regression: fa3 x unified-SWA crashed at gpt-oss
        cuda-graph capture because every backend except triton bundles the
        2-arg KVWriteLoc(loc, swa) and the full-layer door demanded an explicit
        full_loc. Once the loc is rebound the generic `loc` IS the full-side kernel-facing id
        (rebind_write_loc runs at ForwardBatch construction),
        so the door must fall back to it -- the pool still never translates."""
        pool = self._make_bare_pool()
        rebound_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        swa_phys = torch.tensor([1, 2, 0], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(rebound_loc, swa_phys),
            torch.zeros(3, 4, 8),
            torch.zeros(3, 4, 8),
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, kwargs = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, rebound_loc)
        self.assertNotIn("already_physical", kwargs)

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
        # SWA write rides the backend slot: forward the swa-physical loc directly.
        self.assertIs(forwarded, swa_phys)
        self.assertNotIn("already_physical", kwargs)
        # Full pool untouched for an SWA layer.
        self.assertEqual(len(pool.full_kv_pool.calls), 0)

    def test_swa_layer_requires_swa_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=1)  # SWA layer
        # No swa_loc bundled -> the write-loc contract is violated; must assert
        # rather than silently writing wrong (un-translated) locations.
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(
                layer,
                _loc_info(virtual_loc, None),
                torch.zeros(3, 4, 8),
                torch.zeros(3, 4, 8),
            )


class TestUnifiedSWATombstoneClamp(unittest.TestCase):
    """`UnifiedSWAKVPool.translate_loc_from_full_to_swa` must clamp tombstoned
    ids to the reserved padding sink (0).

    A token whose swa page was freed carries -1 in `virtual_to_physical`. Before
    the clamp, that produced a negative id, which a captured graph stores at a
    negative offset from the buffer base. The composite allocator's method of
    the same name already clamped; this path did not.
    """

    def _make_bare_pool(self, page_size, v2p, multiplier=1):
        from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        # A real sub-allocator (not a stand-in): the translation reads its v2p
        # table, and the pool reaches it through the allocator's own method.
        swa_allocator = object.__new__(MultiEndedAllocator)
        # `page_size` is the WIDENED (DCP) surface, `pool_page_size` the physical
        # rows per page; equal at dcp_size == 1, which is what this fixture is.
        swa_allocator.page_size = page_size
        swa_allocator.pool_page_size = page_size
        swa_allocator.virtual_to_physical = v2p
        swa_allocator.kernel_page_multiplier = multiplier
        pool = object.__new__(UnifiedSWAKVPool)
        pool._swa_allocator = swa_allocator
        return pool

    def test_tombstoned_id_lands_on_sink(self):
        for ps, mult in ((1, 1), (4, 1), (4, 6)):
            v2p = torch.tensor([0, -1, 2], dtype=torch.int64)
            pool = self._make_bare_pool(ps, v2p, multiplier=mult)
            # Virtual ids covering the tombstoned page (index 1) and a live one.
            kv_indices = torch.tensor([0, ps, 2 * ps], dtype=torch.int64)
            out = pool.translate_loc_from_full_to_swa(kv_indices)
            self.assertEqual(out.dtype, torch.int64)
            self.assertTrue(
                bool((out >= 0).all().item()),
                f"tombstoned swa id stayed negative at page_size={ps}, "
                f"multiplier={mult}: {out}",
            )
            self.assertEqual(int(out[1].item()), 0)


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
      carries the kernel-facing loc), else the raw `loc` (static pool, already physical).
    - `set_mla_kv_buffer` / `get_mla_kv_buffer` forward `loc` untouched:
      writes are kernel-facing since the ForwardBatch rebind, and read
      indices are translated at their production sites."""

    def _make_bare_pool(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _RecordingMLAPool()
        pool.use_mla = True
        pool.full_attention_layer_id_mapping = {0: 0}
        return pool

    def test_mla_writes_full_loc_from_write_loc(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        kernel_phys = torch.tensor([21, 24, 27], dtype=torch.int64)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            _loc_info(virtual_loc, full_phys=kernel_phys),
            torch.zeros(3, 1, 8),
            None,
        )

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded, _ = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, kernel_phys)
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

    def test_set_mla_kv_buffer_door_never_translates(self):
        """Physical-loc contract: the write door forwards `loc` UNTOUCHED.
        The translate happens exactly once at ForwardBatch construction
        (rebind_write_loc, kernel-facing-first); a door that translated
        again would double-translate every unified MLA write. Deleting the
        forward (or re-adding a door translate) turns this red."""
        pool = self._make_bare_pool()
        loc = torch.tensor([107, 108, 109], dtype=torch.int64)
        layer = types.SimpleNamespace(layer_id=0)

        pool.set_mla_kv_buffer(layer, loc, torch.zeros(3, 1, 6), torch.zeros(3, 1, 2))

        self.assertEqual(len(pool.full_kv_pool.mla_set_calls), 1)
        self.assertIs(pool.full_kv_pool.mla_set_calls[0], loc)

    def test_get_mla_kv_buffer_door_never_translates(self):
        """Kernel-facing contract, read side: `loc` is a read-index tensor
        already translated at its production site
        (fetch_mha_one_shot_kv_indices / prepare_chunked_kv_indices); the
        door forwards it UNTOUCHED -- a re-added door translate would
        double-translate every unified MLA prefix read."""
        pool = self._make_bare_pool()
        loc = torch.tensor([104, 105], dtype=torch.int64)
        layer = types.SimpleNamespace(layer_id=0)

        pool.get_mla_kv_buffer(layer, loc)

        self.assertEqual(len(pool.full_kv_pool.mla_get_calls), 1)
        self.assertIs(pool.full_kv_pool.mla_get_calls[0], loc)


class TestMlaWriteDoorsUnderDcp(unittest.TestCase):
    """Which MLA write door is DCP-aware, and which refuses.

    `set_mla_kv_buffer` resolves the owner rule inside its kernel, so it owns
    the DCP write. `set_kv_buffer` (the combined latent+rope row) cannot: the
    two backends that could reach it disagree on the loc space -- flashinfer's
    `k_rope is None` branch passes a WIDENED loc, the Triton backend one it
    already collapsed -- so there is no single correct translation. It used to
    select `loc % dcp_size == dcp_rank` and then write WITHOUT dividing, i.e.
    widened ids straight into a rank-local buffer. Refusing is the contract;
    a re-added masked-but-undivided write is what this guards."""

    def _bare_mla_pool(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool = object.__new__(MLATokenToKVPool)
        pool.size = 64
        pool.page_size = 1
        pool.kernel_page_blocks = 1
        pool.start_layer = 0
        pool.dtype = torch.float16
        pool.store_dtype = torch.float16
        pool.dsa_kv_cache_store_fp8 = False
        pool.kv_buffer = [torch.zeros((65, 1, 8), dtype=torch.float16)]
        return pool

    def test_set_kv_buffer_refuses_under_dcp(self):
        from sglang.srt.runtime_context import get_parallel

        pool = self._bare_mla_pool()
        layer = types.SimpleNamespace(layer_id=0)
        loc = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        cache_k = torch.ones((4, 1, 8), dtype=torch.float16)

        with get_parallel().override(
            dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=1
        ):
            with self.assertRaises(AssertionError) as cm:
                pool.set_kv_buffer(layer, _loc_info(loc), cache_k, None)
        self.assertIn("set_mla_kv_buffer", str(cm.exception))
        # Nothing was written on the way to refusing.
        self.assertTrue(bool((pool.kv_buffer[0] == 0).all()))

    def test_set_kv_buffer_still_writes_without_dcp(self):
        pool = self._bare_mla_pool()
        layer = types.SimpleNamespace(layer_id=0)
        loc = torch.tensor([3, 5], dtype=torch.int64)
        cache_k = torch.ones((2, 1, 8), dtype=torch.float16)

        pool.set_kv_buffer(layer, _loc_info(loc), cache_k, None)

        self.assertTrue(bool((pool.kv_buffer[0][3] == 1).all()))
        self.assertTrue(bool((pool.kv_buffer[0][5] == 1).all()))
        self.assertTrue(bool((pool.kv_buffer[0][4] == 0).all()))


if __name__ == "__main__":
    unittest.main()
