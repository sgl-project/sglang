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
"""The unified free path must not move anything across the PCIe bus.

Two independent host syncs lived in `MultiEndedAllocator`'s free path:

  1. Tombstone scatters written as ``t[idx] = -1``. The scalar RHS makes torch
     materialise ``-1`` as a CPU tensor and copy it H2D, and a pageable H2D
     copy BLOCKS the host until the stream drains. Invisible on decode-shaped
     work; ~16 ms per free behind an 8192-token prefill.

  2. `torch.unique` recovering distinct PAGE ids from freed TOKEN ids. Its
     output shape is data-dependent, so it must D2H the count
     (``_unique2 -> item -> _local_scalar_dense -> cudaStreamSynchronize``).
     `PagedTokenToKVPoolAllocator` already solved this with `free_segment`:
     a page's tokens sit consecutively in the kv row, so given `start_pos` the
     page representatives are stride slices. The unified allocators simply
     never implemented it and so were permanently on the syncing path.

These tests mirror `test_paged_free_segment.py` -- the same sweep against the
`torch.unique` reference, the same free-group deferral -- because the unified
allocators now mirror that allocator's design rather than a parallel one.

    python -m pytest test/registered/unit/mem_cache/test_unified_free_no_host_sync.py -v
"""

import ast
import inspect
import textwrap
import unittest
from unittest import mock

import torch
from test_multi_ended_allocator import TestPagedMultiEndedAllocator as _PagedFixture

from sglang.srt.mem_cache import multi_ended_allocator as mea
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE_SIZE = _PagedFixture.PAGE_SIZE


def _paged_allocator(lazy: bool):
    """A real paged `MultiEndedAllocator` from the sibling fixture."""
    inst = _PagedFixture([m for m in dir(_PagedFixture) if m.startswith("test_")][0])
    _pool, full, _swa, _fkv, _skv = inst._build()
    full.lazy_compaction = lazy
    return full


# --------------------------------------------------------------------------
# 1. tombstone scatters
# --------------------------------------------------------------------------

_TABLES = {"virtual_to_physical", "physical_to_virtual"}

# Methods that MUST tombstone through index_fill_. Explicit, because "this
# method writes a tombstone" is a design fact per method, not something a scan
# can infer -- but `test_every_allocator_free_path_is_listed` below fails if a
# new allocator arrives with its own free path and is not added here.
_TOMBSTONE_METHODS = [
    (mea.MultiEndedAllocator, "_free_lazy"),
    (mea.MultiEndedAllocator, "free"),
    (mea.MultiEndedAllocator, "_commit_move_batch"),
    (mea.FloatMultiEndedAllocator, "free"),
    (mea.FloatMultiEndedAllocator, "make_room"),
    (mea.FloatMultiEndedAllocator, "_relocate_to_positions"),
]


# Ways to write `-1` into a table without the value crossing the bus:
# `index_fill_` takes the scalar as an argument torch keeps off the host, and
# the fused launchers store it from inside the kernel.
_NO_SYNC_TOMBSTONE_FORMS = ("index_fill_", "free_unbind_inplace")


def _allocators_in_module():
    """Every allocator class DEFINED in multi_ended_allocator (not imported)."""
    return sorted(
        (
            c
            for c in vars(mea).values()
            if isinstance(c, type)
            and c.__module__ == mea.__name__
            and "Allocator" in c.__name__
        ),
        key=lambda c: c.__name__,
    )


def _table_touching_methods():
    """Every own method of every allocator whose source names a page table.

    DISCOVERY, not a list: a hardcoded list stops guarding the moment a new
    allocator class arrives with its own free path -- which is what happened
    when FloatMultiEndedAllocator was added and inherited no coverage.
    """
    out = []
    for cls in _allocators_in_module():
        for name, fn in vars(cls).items():
            if not inspect.isfunction(fn):
                continue
            try:
                src = inspect.getsource(fn)
            except OSError:
                continue
            if any(f".{t}[" in src for t in _TABLES):
                out.append((cls, name))
    return sorted(out, key=lambda pair: (pair[0].__name__, pair[1]))


def _scalar_index_assignments(fn):
    """`self.<table>[<tensor idx>] = <scalar>` occurrences in fn's source.

    Slice assignments (``t[a:b] = -1``) are excluded: a slice is a view, so the
    fill needs no index tensor. Tensor-valued scatters are excluded too -- only
    the scalar RHS materialises a CPU value tensor.
    """

    def _is_scalar_literal(node):
        # NOTE: `-1` parses as UnaryOp(USub, Constant(1)), NOT Constant. Testing
        # only for Constant silently skips every negative literal -- i.e. every
        # tombstone this scan exists to find.
        if isinstance(node, ast.Constant):
            return True
        return isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _is_scalar_literal(node.value):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Subscript):
                continue
            val = tgt.value
            if not (isinstance(val, ast.Attribute) and val.attr in _TABLES):
                continue
            if isinstance(tgt.slice, ast.Slice):
                continue
            # A CONSTANT integer index (`t[0] = 0`, `t[-1] = -1`) is a
            # single-element sentinel write, not the tensor-index tombstone this
            # guard exists to find, and the only such writes are in `clear()`.
            if _is_scalar_literal(tgt.slice):
                continue
            bad.append(ast.unparse(node))
    return bad


class TestTombstonesDoNotCrossTheBus(unittest.TestCase):
    def test_no_scalar_index_assignment(self):
        discovered = _table_touching_methods()
        self.assertGreaterEqual(
            len(discovered), len(_TOMBSTONE_METHODS), "discovery scan went blind"
        )
        for cls, name in discovered:
            with self.subTest(method=f"{cls.__name__}.{name}"):
                bad = _scalar_index_assignments(getattr(cls, name))
                self.assertEqual(
                    bad,
                    [],
                    msg=(
                        f"{cls.__name__}.{name} writes a tombstone with a scalar "
                        f"RHS: {bad}. That materialises -1 as a CPU tensor and "
                        f"copies it H2D, blocking the scheduler thread until the "
                        f"stream drains. Use `.index_fill_(0, idx, -1)`."
                    ),
                )

    def test_the_scan_detects_the_scalar_form_it_guards(self):
        """Self-check. The scan is only as good as its AST matching, and it
        silently missed every tombstone until `-1` was recognised as
        UnaryOp(USub, Constant) rather than Constant. Pin that."""

        def _offender(self):
            self.virtual_to_physical[free_v_pages] = -1  # noqa: F821

        self.assertEqual(len(_scalar_index_assignments(_offender)), 1)

    def test_every_allocator_free_path_is_listed(self):
        """The positive list must name every allocator that owns a free path.

        REGRESSION: the list used to hold three MultiEndedAllocator methods, so
        adding FloatMultiEndedAllocator with its own `free` silently dropped that
        free path out of coverage -- and it shipped a scalar tombstone. Fail here
        instead, loudly, the next time an allocator arrives.
        """
        listed = {(cls.__name__, name) for cls, name in _TOMBSTONE_METHODS}
        for cls in _allocators_in_module():
            for name, fn in vars(cls).items():
                if not inspect.isfunction(fn):
                    continue
                try:
                    src = inspect.getsource(fn)
                except OSError:
                    continue
                # WRITES a page table -- either correctly (index_fill_) or in the
                # banned scalar form the scan below catches. A method that only
                # READS a table has nothing to tombstone.
                if not (
                    any(f"{t}.index_fill_" in src for t in _TABLES)
                    or _scalar_index_assignments(fn)
                ):
                    continue
                self.assertIn(
                    (cls.__name__, name),
                    listed,
                    msg=(
                        f"{cls.__name__}.{name} writes a page table but is not in "
                        f"_TOMBSTONE_METHODS, so the index_fill_ guard does not "
                        f"cover it. Add it."
                    ),
                )

    def test_free_paths_actually_write_a_tombstone(self):
        """Positive form, so deleting the scatter entirely cannot pass. The
        mechanism is not the point -- keeping the tombstone value off the host
        is -- so this lists the sanctioned ways to do that and a new one is
        added here deliberately."""
        for cls, name in _TOMBSTONE_METHODS:
            with self.subTest(method=f"{cls.__name__}.{name}"):
                src = inspect.getsource(getattr(cls, name))
                self.assertTrue(
                    any(form in src for form in _NO_SYNC_TOMBSTONE_FORMS),
                    f"{cls.__name__}.{name} writes no tombstone through any of "
                    f"{_NO_SYNC_TOMBSTONE_FORMS}",
                )

    def test_index_fill_matches_scalar_assign_semantics(self):
        """Behaviour-preserving, including the edge cases the free path hands
        it: empty index, duplicate pages, full table."""
        for idx in (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([1, 3, 5], dtype=torch.int64),
            torch.tensor([2, 2, 3], dtype=torch.int64),  # duplicates
            torch.arange(6, dtype=torch.int64),
        ):
            with self.subTest(n=int(idx.numel())):
                a = torch.arange(6, dtype=torch.int64)
                b = a.clone()
                a[idx] = -1
                b.index_fill_(0, idx, -1)
                self.assertTrue(torch.equal(a, b))


# --------------------------------------------------------------------------
# 2. free_segment: stride page extraction instead of torch.unique
# --------------------------------------------------------------------------


class TestFreeSegment(unittest.TestCase):
    """Mirrors `test_paged_free_segment.TestFreeSegment`."""

    def test_matches_unique_over_alignments(self):
        """Sweep (start, end) so segments cover aligned/unaligned head and
        tail, a single partial page, and the full row."""
        for num_tokens in (1, PAGE_SIZE, PAGE_SIZE + 1, 3 * PAGE_SIZE - 1):
            for start in range(0, num_tokens, max(1, num_tokens // 4)):
                for end in (start + 1, num_tokens):
                    if end <= start:
                        continue
                    alloc = _paged_allocator(lazy=True)
                    row = alloc.alloc(3 * PAGE_SIZE)
                    seg = row[start:end]
                    expected = torch.unique(seg // PAGE_SIZE)
                    alloc.free_segment(seg, start_pos=start)
                    freed = torch.sort(alloc._free_phys_pages)[0]
                    with self.subTest(n=num_tokens, start=start, end=end):
                        # v2p is identity-ish here, so freed physical pages map
                        # 1:1 onto the expected virtual pages.
                        self.assertEqual(freed.numel(), expected.numel())

    def test_never_calls_unique(self):
        """The decisive check -- make `torch.unique` explode. A textual guard
        can be fooled; this cannot."""
        for start in (0, 1, PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE + 3):
            alloc = _paged_allocator(lazy=True)
            row = alloc.alloc(3 * PAGE_SIZE)
            with self.subTest(start_pos=start):
                with mock.patch.object(
                    torch, "unique", side_effect=AssertionError("sync path taken")
                ):
                    alloc.free_segment(row[start : start + PAGE_SIZE], start_pos=start)

    def test_empty_segment_is_noop(self):
        alloc = _paged_allocator(lazy=True)
        before = alloc._free_phys_pages.numel()
        alloc.free_segment(torch.empty(0, dtype=torch.int64), start_pos=0)
        self.assertEqual(alloc._free_phys_pages.numel(), before)

    def test_page_size_one_takes_the_plain_path(self):
        """token == page: nothing to dedup, so `free_segment` must not invent
        a stride slice that would drop tokens."""
        alloc = _paged_allocator(lazy=True)
        alloc.page_size = 1
        v = alloc.alloc(PAGE_SIZE)
        n = v.numel()
        alloc.free_segment(v, start_pos=0)
        self.assertEqual(alloc._free_phys_pages.numel(), n)


class TestFreeGroupKeepsPositions(unittest.TestCase):
    """Mirrors `test_paged_free_segment.test_group_defers_until_group_end`.

    Bug regression: buffering RAW tokens and `torch.cat`-ing them at
    `free_group_end` destroys each segment's shape, so the merged tensor has no
    recoverable page structure and falls back to `torch.unique`. Measured as 71
    of 77 `_free_lazy` calls still syncing on gpt-oss and Qwen3.5 ps=256
    (eval_429), all attributed to `free_group_end` via the decode path. The fix
    buffers page REPRESENTATIVES, so the merge concatenates page ids.
    """

    def test_group_defers_until_group_end(self):
        alloc = _paged_allocator(lazy=True)
        row = alloc.alloc(2 * PAGE_SIZE)
        before = alloc._free_phys_pages.numel()
        alloc.free_group_begin()
        alloc.free_segment(row, start_pos=0)
        self.assertEqual(
            alloc._free_phys_pages.numel(), before, "must defer inside the group"
        )
        alloc.free_group_end()
        self.assertEqual(alloc._free_phys_pages.numel(), before + 2)

    def test_group_end_does_not_sync(self):
        """The property the whole fix exists for: a grouped segment free must
        complete with `torch.unique` disabled."""
        alloc = _paged_allocator(lazy=True)
        row = alloc.alloc(3 * PAGE_SIZE)
        alloc.free_group_begin()
        alloc.free_segment(row[:PAGE_SIZE], start_pos=0)
        alloc.free_segment(
            row[PAGE_SIZE + 3 : 2 * PAGE_SIZE + 3], start_pos=PAGE_SIZE + 3
        )
        with mock.patch.object(
            torch, "unique", side_effect=AssertionError("sync path taken")
        ):
            alloc.free_group_end()
        self.assertGreater(alloc._free_phys_pages.numel(), 0)

    def test_positionless_group_still_uses_the_unique_path(self):
        """Plain `free()` inside a group has no position to keep, so it must
        still go through the (syncing) dedup -- correctness over speed."""
        alloc = _paged_allocator(lazy=True)
        row = alloc.alloc(2 * PAGE_SIZE)
        alloc.free_group_begin()
        alloc.free(row)
        with self.assertRaises(AssertionError):
            with mock.patch.object(
                torch, "unique", side_effect=AssertionError("expected")
            ):
                alloc.free_group_end()


class TestEveryUnifiedAllocatorOverridesFreeSegment(unittest.TestCase):
    """Completeness guard. The base `free_segment` DISCARDS `start_pos` and
    calls plain `free`, so an allocator that inherits it sends every segment
    free into the syncing dedup -- silently, with no error and no wrong answer,
    just a stalled scheduler thread. That is exactly what happened: the SWA
    composite was overridden and the Mamba composite was not, and 77 of 77
    `_free_lazy` calls on Qwen3.5 ps=256 still synced (eval_428).
    """

    def test_all_overridden(self):
        for cls in (
            mea.MultiEndedAllocator,
            mea.UnifiedMambaTokenToKVPoolAllocator,
            mea.UnifiedSWATokenToKVPoolAllocator,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertIsNot(
                    cls.free_segment,
                    BaseTokenToKVPoolAllocator.free_segment,
                    msg=(
                        f"{cls.__name__} inherits the base `free_segment`, which "
                        f"discards `start_pos` -- every segment free will take the "
                        f"host-syncing dedup."
                    ),
                )

    def test_composites_buffer_reps_not_tokens_in_a_group(self):
        """The group buffer must exist on every allocator that can receive a
        segment free, or `free_segment` raises inside a group."""
        for cls in (
            mea.MultiEndedAllocator,
            mea.UnifiedMambaTokenToKVPoolAllocator,
            mea.UnifiedSWATokenToKVPoolAllocator,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertIn("free_page_reps_group", inspect.getsource(cls))


class TestFreeSwaWindowRatchetNoHostSync(unittest.TestCase):
    """The per-decode-step SWA window ratchet frees a CONTIGUOUS row slice
    with host-int, page-aligned bounds — the same shape `free_segment` was
    built for. `free_swa(..., start_pos=)` must therefore reach the swa side
    with caller-derived page ids: no `torch.unique` (data-dependent shape =
    host sync) and no stale-slot `.item()` on the per-step path.

    Poisoning the ops is the decisive form (a textual guard can be fooled).
    """

    PS = 4

    def _swa_composite(self, lazy=True):
        from test_multi_ended_allocator import _FakeKVCache, _make_mha_spec

        from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool

        full = _make_mha_spec("full", "up", layer_num=4)
        swa = _make_mha_spec("swa", "down", layer_num=2)
        total = 64 * full.entry_bytes() + 64 * swa.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device="cpu",
            enable_memory_saver=False,
            page_size=self.PS,
        )

        class _KV:
            def __init__(self, p):
                self.full_kv_pool = _FakeKVCache(p.max_slots("full"))
                self.swa_kv_pool = _FakeKVCache(p.max_slots("swa"))

            def attach_allocators(self, **kwargs):
                pass

        return mea.UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_KV(pool),
            device="cpu",
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=64,
            page_size=self.PS,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )

    def test_ratchet_shape_free_swa_never_syncs(self):
        """Aligned bounds (the ratchet guarantees them at ps>1): no unique,
        no item -- on the lazy production config."""
        alloc = self._swa_composite(lazy=True)
        v = alloc.alloc(8 * self.PS)
        self.assertIsNotNone(v)
        with (
            mock.patch.object(
                torch, "unique", side_effect=AssertionError("unique = host sync")
            ),
            mock.patch.object(
                torch.Tensor, "item", side_effect=AssertionError("item = host sync")
            ),
        ):
            alloc.free_swa(v[: 4 * self.PS], start_pos=0)
            alloc.free_swa(v[4 * self.PS :], start_pos=4 * self.PS)

    def test_unaligned_start_pos_still_no_sync(self):
        """`_page_reps_pieces` covers a misaligned start with a second piece;
        the sync-free property must not depend on alignment."""
        alloc = self._swa_composite(lazy=True)
        v = alloc.alloc(8 * self.PS)
        with mock.patch.object(
            torch, "unique", side_effect=AssertionError("unique = host sync")
        ):
            alloc.free_swa(v[1 : 5 * self.PS], start_pos=1)

    def test_start_pos_path_matches_the_fallback_end_state(self):
        """Derived property: the stride-rep path and the dedup fallback must
        leave IDENTICAL allocator state (v2p tombstones, capacity)."""
        for lazy in (True, False):
            with self.subTest(lazy=lazy):
                a1 = self._swa_composite(lazy=lazy)
                a2 = self._swa_composite(lazy=lazy)
                v1 = a1.alloc(6 * self.PS)
                v2 = a2.alloc(6 * self.PS)
                self.assertTrue(torch.equal(v1, v2))
                a1.free_swa(v1[: 4 * self.PS], start_pos=0)
                a2.free_swa(v2[: 4 * self.PS])  # fallback (radix shape)
                self.assertTrue(
                    torch.equal(
                        a1.swa_attn_allocator.virtual_to_physical,
                        a2.swa_attn_allocator.virtual_to_physical,
                    )
                )
                self.assertEqual(a1.available_size(), a2.available_size())
                self.assertEqual(
                    a1.swa_attn_allocator.schedulable_available_size(),
                    a2.swa_attn_allocator.schedulable_available_size(),
                )

    def test_double_ratchet_is_filtered_not_crashed(self):
        """Freeing an already-tombstoned range again must no-op through the
        liveness filter (radix eviction and the ratchet can overlap)."""
        alloc = self._swa_composite(lazy=True)
        v = alloc.alloc(4 * self.PS)
        alloc.free_swa(v, start_pos=0)
        alloc.free_swa(v, start_pos=0)  # all tombstoned -> filtered to empty


@unittest.skipUnless(
    torch.cuda.is_available(), "the fused tombstone is a Triton kernel"
)
class TestFusedTombstoneWritesBothTables(unittest.TestCase):
    """The source scan above accepts `free_unbind_inplace` as a no-sync
    mechanism; this is what makes that acceptance mean something. On CPU the
    launcher takes its pure-torch reference path, so nothing else in the suite
    ever runs the kernel that does the tombstoning.
    """

    def test_matches_the_reference_over_randomized_bindings(self):
        from sglang.kernels.ops.memory.virtual_slot import (
            bind_inplace,
            free_unbind_inplace,
        )

        g = torch.Generator(device="cuda").manual_seed(11)
        for trial in range(20):
            n_pages = int(torch.randint(4, 400, (1,), generator=g, device="cuda"))
            n_free = int(
                torch.randint(1, n_pages + 1, (1,), generator=g, device="cuda")
            )
            phys = torch.randperm(n_pages, device="cuda", generator=g).to(torch.int64)
            virt = torch.randperm(n_pages, device="cuda", generator=g).to(torch.int64)
            v2p = torch.full((n_pages,), -1, dtype=torch.int64, device="cuda")
            p2v = torch.full((n_pages,), -1, dtype=torch.int64, device="cuda")
            bind_inplace(virt, phys, v2p, p2v)
            self.assertTrue(torch.equal(v2p[virt], phys), f"bind trial {trial}")
            self.assertTrue(torch.equal(p2v[phys], virt), f"bind trial {trial}")

            freed_v = virt[:n_free]
            want_p = v2p[freed_v].clone()
            got_p = free_unbind_inplace(freed_v, v2p, p2v)

            self.assertTrue(torch.equal(got_p, want_p), f"freed pages, trial {trial}")
            self.assertTrue(
                torch.all(v2p[freed_v] == -1), f"v2p not tombstoned, trial {trial}"
            )
            self.assertTrue(
                torch.all(p2v[want_p] == -1), f"p2v not tombstoned, trial {trial}"
            )
            live_v = virt[n_free:]
            self.assertTrue(
                torch.equal(v2p[live_v], phys[n_free:]),
                f"a live binding was disturbed, trial {trial}",
            )

    def test_cuda_agrees_with_the_cpu_reference(self):
        from sglang.kernels.ops.memory.virtual_slot import free_unbind_inplace

        v = torch.tensor([3, 0, 5], dtype=torch.int64)
        cpu_v2p = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.int64)
        cpu_p2v = torch.tensor([5, 0, 1, 2, 3, 4], dtype=torch.int64)
        cu_v2p, cu_p2v = cpu_v2p.cuda(), cpu_p2v.cuda()
        cpu_out = free_unbind_inplace(v, cpu_v2p, cpu_p2v)
        cu_out = free_unbind_inplace(v.cuda(), cu_v2p, cu_p2v)
        self.assertTrue(torch.equal(cpu_out, cu_out.cpu()))
        self.assertTrue(torch.equal(cpu_v2p, cu_v2p.cpu()))
        self.assertTrue(torch.equal(cpu_p2v, cu_p2v.cpu()))

    def test_empty_free_is_a_noop(self):
        from sglang.kernels.ops.memory.virtual_slot import free_unbind_inplace

        v2p = torch.arange(4, dtype=torch.int64, device="cuda")
        before = v2p.clone()
        out = free_unbind_inplace(
            torch.empty(0, dtype=torch.int64, device="cuda"), v2p, v2p.clone()
        )
        self.assertEqual(int(out.numel()), 0)
        self.assertTrue(torch.equal(v2p, before))


if __name__ == "__main__":
    unittest.main()
