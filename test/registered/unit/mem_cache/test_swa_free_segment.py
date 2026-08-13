"""free_swa_segment vs an independent oracle: stride page extraction over segment
alignments, plus the free-group contract. See
SWATokenToKVPoolAllocator.free_swa_segment for why the data-dependent ops are avoided.

    python -m pytest test/registered/unit/mem_cache/test_swa_free_segment.py -v
"""

import inspect
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.multi_ended_allocator import UnifiedSWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SIZE = 256


def _pool():
    # The free path never reaches the pool, so isinstance plus swa_kv_pool is all
    # the allocator needs from it.
    pool = MagicMock(spec=BaseSWAKVPool)
    pool.swa_kv_pool = None
    return pool


def _alloc(page_size, need_sort=False):
    return SWATokenToKVPoolAllocator(
        size=SIZE,
        size_swa=SIZE,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_pool(),
        need_sort=need_sort,
    )


def _row(alloc, num_tokens):
    """alloc_extend without its triton kernel: one page-aligned full and swa
    allocation of the same length, so token t's swa slot is
    swa_page[t // ps] * ps + t % ps. The GPU suite covers the real path."""
    ps = alloc.page_size
    padded = -(num_tokens // -ps) * ps
    full = alloc.full_attn_allocator.alloc(padded)
    alloc.set_full_to_swa_mapping(full, alloc.swa_attn_allocator.alloc(padded))
    return full[:num_tokens]


def _freed(alloc):
    # need_sort (PD disagg) routes frees into release_pages, so span both.
    inner = alloc.swa_attn_allocator
    return set(torch.cat((inner.free_pages, inner.release_pages)).tolist())


def _oracle(row, start, end, ps, mapping):
    """Independent reference in plain Python -- never an allocator free path.
    Returns (swa page ids to release, mapping slots that must be zero)."""
    pages = sorted({row[i].item() // ps for i in range(start, end)})
    touched = [p * ps + off for p in pages for off in range(ps)]
    swa = sorted({mapping[t].item() // ps for t in touched if mapping[t].item() > 0})
    return swa, touched


class TestSWAFreeSegment(unittest.TestCase):
    def test_sweep_against_oracle(self):
        # Aligned start per the contract; aligned, partial and interior segments.
        # ps == 1 is the special-cased branch, ps == 4 the general one.
        for ps in (1, 4):
            for n in (ps, ps + 1, 3 * ps - 1):
                for start in range(0, n, ps):
                    for end in range(start + 1, n + 1):
                        a = _alloc(ps)
                        row = _row(a, n)
                        want_swa, want_zero = _oracle(
                            row, start, end, ps, a.full_to_swa_index_mapping.clone()
                        )
                        before = _freed(a)
                        a.free_swa_segment(
                            row[start:end], start_pos=start, swa_alive_from=start
                        )

                        label = f"{ps=} {n=} {start=} {end=}"
                        self.assertEqual(sorted(_freed(a) - before), want_swa, label)
                        got = a.full_to_swa_index_mapping[want_zero]
                        self.assertTrue(torch.equal(got, torch.zeros_like(got)), label)

    def test_group_defers_and_queues_owned_values(self):
        # Queueing the caller's view would make the flush read whatever a remap
        # wrote into that row in between.
        for ps in (1, 4):
            a = _alloc(ps)
            row = _row(a, 2 * ps)
            want_swa, want_zero = _oracle(
                row, 0, row.numel(), ps, a.full_to_swa_index_mapping.clone()
            )
            before = _freed(a)

            a.free_group_begin()
            a.free_swa_segment(row, start_pos=0, swa_alive_from=0)
            self.assertEqual(_freed(a), before, f"not deferred: {ps=}")
            row.fill_(1)  # stand in for the row being remapped mid-group
            a.free_group_end()

            self.assertEqual(sorted(_freed(a) - before), want_swa, f"{ps=}")
            got = a.full_to_swa_index_mapping[want_zero]
            self.assertTrue(torch.equal(got, torch.zeros_like(got)), f"{ps=}")

    def test_contract_asserts(self):
        a = _alloc(4)
        a.debug_mode = True
        row = _row(a, 8)
        with self.assertRaises(AssertionError):  # start_pos not page aligned
            a.free_swa_segment(row[1:], start_pos=1, swa_alive_from=1)
        a.free_swa_segment(row, start_pos=0, swa_alive_from=0)
        with self.assertRaises(AssertionError):  # range already freed
            a.free_swa_segment(row, start_pos=0, swa_alive_from=0)

    def test_dead_prefix_is_rejected_without_debug_mode(self):
        """The liveness precondition is host-side, so it holds without
        debug_mode -- reading a dead page would release page 0 and leak."""
        ps = 4
        a = _alloc(ps)
        a.debug_mode = False
        row = _row(a, 3 * ps)
        # Front page already evicted: the request is live only from ps onward.
        a.free_swa_segment(row[:ps], start_pos=0, swa_alive_from=0)
        with self.assertRaises(AssertionError):
            a.free_swa_segment(row, start_pos=0, swa_alive_from=ps)
        # Stating the frontier correctly frees only what is still mapped.
        a.free_swa_segment(row[ps:], start_pos=ps, swa_alive_from=ps)
        self.assertNotIn(0, _freed(a))


class TestOptOuts(unittest.TestCase):
    def _pure_swa(self):
        return PureSWATokenToKVPoolAllocator(
            size_swa=SIZE,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=_pool(),
            need_sort=False,
        )

    def test_pure_swa_free_segment_releases_each_slot_once(self):
        # full_attn_allocator IS swa_attn_allocator here, so the parent's
        # two-sided release would hand every slot back twice -- silently, and
        # a later alloc would then serve one slot to two requests.
        for alive in (0, None):
            a = self._pure_swa()
            before = a.available_size()
            row = a.alloc(4)
            a.free_segment(row, start_pos=0, swa_alive_from=alive)
            self.assertEqual(a.available_size(), before, f"{alive=}")

    def test_pure_swa_keeps_its_identity_mapping(self):
        # full == swa and the mapping is a constant identity table; zeroing it
        # would break every later translate.
        a = PureSWATokenToKVPoolAllocator(
            size_swa=SIZE,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=_pool(),
            need_sort=False,
        )
        before = a.full_to_swa_index_mapping.clone()
        a.free_swa_segment(a.alloc(4), start_pos=0, swa_alive_from=0)
        self.assertTrue(torch.equal(a.full_to_swa_index_mapping, before))

    def test_unified_swa_does_not_inherit_the_fast_path(self):
        # Shared mode has no mapping table (the swa v2p IS the mapping) and
        # tombstones with -1. Too heavy to build on CPU, so this is the only guard
        # against the override being dropped, which fails silently.
        self.assertIsNot(
            UnifiedSWATokenToKVPoolAllocator.free_swa_segment,
            SWATokenToKVPoolAllocator.free_swa_segment,
        )


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


class TestOverrideSignatures(unittest.TestCase):
    def test_overrides_accept_every_base_kwarg(self):
        """An override that drops a kwarg the base declares TypeErrors at its
        first caller, and only on the config that reaches that subclass -- so
        neither a CPU suite nor one model's e2e run catches it. Checks the whole
        family at once rather than one signature at a time."""
        checked = set()
        for name in ("free_segment", "free_swa_segment", "free_page_reps"):
            want = set(
                inspect.signature(getattr(BaseTokenToKVPoolAllocator, name)).parameters
            )
            for cls in _subclasses(BaseTokenToKVPoolAllocator):
                impl = cls.__dict__.get(name)
                if impl is None:
                    continue
                checked.add(f"{cls.__name__}.{name}")
                missing = want - set(inspect.signature(impl).parameters)
                self.assertFalse(
                    missing,
                    f"{cls.__name__}.{name} drops {sorted(missing)}; a caller "
                    f"passing them would TypeError",
                )

        # __subclasses__ only sees imported modules, so a vacuous pass is the
        # real risk here.
        for expected in (
            "PagedTokenToKVPoolAllocator.free_segment",
            "SWATokenToKVPoolAllocator.free_swa_segment",
            "UnifiedSWATokenToKVPoolAllocator.free_swa_segment",
        ):
            self.assertIn(expected, checked)

    def test_free_override_implies_free_segment_override(self):
        """SWATokenToKVPoolAllocator.free_segment drives full_attn_allocator and
        free_swa directly instead of going through self.free(). A subclass that
        redefines free() therefore has release semantics the parent's split does
        not reproduce -- PureSWA aliases the two inner allocators (so the split
        frees twice), UnifiedSWA clears both inverse histories (which the split
        skips). Both shipped as silent corruption before this check existed."""
        seen = set()
        for cls in _subclasses(SWATokenToKVPoolAllocator):
            if "free" not in cls.__dict__:
                continue
            seen.add(cls.__name__)
            # assertTrue, not assertIn: the latter dumps the whole __dict__.
            self.assertTrue(
                "free_segment" in cls.__dict__,
                f"{cls.__name__} overrides free() but not free_segment(); the "
                f"parent's free_segment would bypass its release semantics",
            )
        for expected in (
            "PureSWATokenToKVPoolAllocator",
            "UnifiedSWATokenToKVPoolAllocator",
        ):
            self.assertIn(expected, seen)


if __name__ == "__main__":
    unittest.main()
