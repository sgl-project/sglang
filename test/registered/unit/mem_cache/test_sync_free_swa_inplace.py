"""Tests for fixed-shape SWA page release."""

import inspect
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 128
FULL_PAGES = 32
SWA_PAGES = 16


def _make_allocator(
    page_size: int = PAGE_SIZE, *, need_sort: bool = False, device: str = "cpu"
) -> SWATokenToKVPoolAllocator:
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.full_kv_pool = None
    kvcache.swa_kv_pool = None
    kvcache.register_mapping.side_effect = lambda mapping: setattr(
        kvcache, "full_to_swa_index_mapping", mapping
    )
    kvcache.translate_loc_from_full_to_swa.side_effect = (
        lambda indices: kvcache.full_to_swa_index_mapping[indices]
    )
    return SWATokenToKVPoolAllocator(
        size=FULL_PAGES * page_size,
        size_swa=SWA_PAGES * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device=device,
        kvcache=kvcache,
        need_sort=need_sort,
    )


def _segments(
    seed: int,
    count: int,
    *,
    page_size: int = PAGE_SIZE,
    pages_per_segment: int = 2,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pages = torch.randperm(FULL_PAGES - 1, generator=generator) + 1
    pages = pages[: count * pages_per_segment].reshape(count, pages_per_segment)
    offsets = torch.arange(page_size)
    return [
        (segment_pages[:, None] * page_size + offsets).reshape(-1).to(torch.int64)
        for segment_pages in pages
    ]


def _install_mappings(
    allocator: SWATokenToKVPoolAllocator,
    segments: list[torch.Tensor],
    *,
    mapped_fraction: float = 1.0,
) -> None:
    for segment in segments:
        mapped_tokens = int(segment.numel() * mapped_fraction)
        mapped_tokens = mapped_tokens // allocator.page_size * allocator.page_size
        if mapped_tokens == 0:
            continue
        swa_indices = allocator.swa_attn_allocator.alloc(mapped_tokens)
        assert swa_indices is not None
        allocator.set_full_to_swa_mapping(segment[:mapped_tokens], swa_indices)


def _state(allocator: SWATokenToKVPoolAllocator):
    swa = allocator.swa_attn_allocator
    return (
        swa.free_pages.clone(),
        swa.get_all_free_pages().clone(),
        allocator.full_to_swa_index_mapping.clone(),
        allocator.swa_available_size(),
    )


def _alloc_extend_row(num_tokens: int):
    allocator = _make_allocator(device="cuda")
    prefix_lens_cpu = torch.zeros(1, dtype=torch.int64)
    seq_lens_cpu = torch.tensor([num_tokens], dtype=torch.int64)
    row = allocator.alloc_extend(
        prefix_lens=prefix_lens_cpu.cuda(),
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens_cpu.cuda(),
        seq_lens_cpu=seq_lens_cpu,
        last_loc=torch.tensor([-1], dtype=torch.int64, device="cuda"),
        extend_num_tokens=num_tokens,
    )
    assert row is not None
    return allocator, row


def _sync_error(fn):
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


class TestSyncFreeSWA(CustomTestCase):
    def assertStateEqual(self, lhs, rhs):
        for name, left, right in zip(
            ("free_pages", "all_free_pages", "mapping"), lhs[:3], rhs[:3]
        ):
            self.assertTrue(torch.equal(left, right), f"{name} differs")
        self.assertEqual(lhs[3], rhs[3])

    def _run_static_case(
        self,
        segments: list[torch.Tensor],
        *,
        fast_path: bool,
        page_size: int,
        grouped: bool,
        mixed_segment: torch.Tensor | None,
        mapped_fraction: float,
        need_sort: bool,
    ):
        allocator = _make_allocator(page_size, need_sort=need_sort)
        installed = segments + ([mixed_segment] if mixed_segment is not None else [])
        _install_mappings(allocator, installed, mapped_fraction=mapped_fraction)

        if grouped:
            allocator.free_group_begin()
        for segment in segments:
            if fast_path:
                allocator.free_swa(segment, start_pos=0)
            else:
                allocator.free_swa(segment)
        if mixed_segment is not None:
            allocator.free_swa(mixed_segment)
        if grouped:
            allocator.free_group_end()
        return _state(allocator)

    def test_static_fast_path_matches_legacy(self):
        cases = (
            ("paged", PAGE_SIZE, False, False, 1.0, False),
            ("paged_grouped", PAGE_SIZE, True, False, 1.0, False),
            ("mixed_group", PAGE_SIZE, True, True, 1.0, False),
            ("release_pages", PAGE_SIZE, False, False, 1.0, True),
            ("token", 1, False, False, 1.0, False),
            ("token_grouped", 1, True, False, 1.0, False),
        )
        for seed, case in enumerate(cases, start=1):
            name, page_size, grouped, mixed, fraction, need_sort = case
            segments = _segments(seed, 4 if mixed else 3, page_size=page_size)
            mixed_segment = segments.pop() if mixed else None
            kwargs = dict(
                page_size=page_size,
                grouped=grouped,
                mixed_segment=mixed_segment,
                mapped_fraction=fraction,
                need_sort=need_sort,
            )
            with self.subTest(name):
                legacy = self._run_static_case(segments, fast_path=False, **kwargs)
                fast = self._run_static_case(segments, fast_path=True, **kwargs)
                self.assertStateEqual(legacy, fast)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_real_alloc_extend_matches_legacy(self):
        num_tokens = 3 * PAGE_SIZE
        for start_pos in (0, PAGE_SIZE):
            fast, fast_row = _alloc_extend_row(num_tokens)
            legacy, legacy_row = _alloc_extend_row(num_tokens)

            fast.free_swa(fast_row[start_pos:], start_pos=start_pos)
            legacy.free_swa(legacy_row[start_pos:])
            torch.cuda.synchronize()

            with self.subTest(start_pos=start_pos):
                self.assertStateEqual(_state(legacy), _state(fast))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_segment_free_does_not_synchronize(self):
        warmup, warmup_row = _alloc_extend_row(2 * PAGE_SIZE)
        warmup.free_swa(warmup_row, start_pos=0)
        torch.cuda.synchronize()

        allocator, row = _alloc_extend_row(2 * PAGE_SIZE)
        torch.cuda.synchronize()
        peers = allocator.full_to_swa_index_mapping[row]
        if _sync_error(lambda: peers[peers > 0]) is None:
            self.skipTest("sync debug mode does not flag data-dependent shapes")

        self.assertIsNone(_sync_error(lambda: allocator.free_swa(row, start_pos=0)))

    def test_page_id_contract(self):
        allocator = _make_allocator()
        paged = allocator.swa_attn_allocator
        paged.debug_mode = False
        self.assertIsNotNone(paged.alloc(3 * PAGE_SIZE))
        paged.free_page_ids(torch.tensor([3, 1, 2]))
        self.assertTrue(torch.equal(paged.free_pages, torch.arange(1, SWA_PAGES + 1)))

        invalid_cases = (
            ("reserved", torch.tensor([0])),
            ("out_of_range", torch.tensor([SWA_PAGES + 1])),
            ("duplicate", torch.tensor([1, 1])),
        )
        for name, page_ids in invalid_cases:
            allocator = _make_allocator()
            paged = allocator.swa_attn_allocator
            paged.debug_mode = False
            before = (paged.free_pages.clone(), paged.get_all_free_pages().clone())
            with (
                self.subTest(name),
                self.assertRaisesRegex(RuntimeError, "valid and unique"),
            ):
                paged.free_page_ids(page_ids)
            self.assertTrue(torch.equal(paged.free_pages, before[0]))
            self.assertTrue(torch.equal(paged.get_all_free_pages(), before[1]))

        allocator = _make_allocator()
        paged = allocator.swa_attn_allocator
        paged.debug_mode = True
        allocated = paged.alloc(PAGE_SIZE)
        assert allocated is not None
        page_ids = allocated[::PAGE_SIZE] // PAGE_SIZE
        paged.free_page_ids(page_ids)
        with self.assertRaises(AssertionError):
            paged.free_page_ids(page_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_invalid_page_id_asserts(self):
        script = textwrap.dedent(f"""
            import torch

            from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

            allocator = PagedTokenToKVPoolAllocator(
                size={SWA_PAGES * PAGE_SIZE},
                page_size={PAGE_SIZE},
                dtype=torch.float16,
                device="cuda",
                kvcache=None,
                need_sort=False,
            )
            allocator.alloc({PAGE_SIZE})
            allocator.free_page_ids(torch.tensor([0], device="cuda"))
            torch.cuda.synchronize()
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertNotEqual(result.returncode, 0, output)
        self.assertTrue(
            "device-side assert" in output or "valid and unique" in output,
            f"returncode {result.returncode}; {output}",
        )

    def test_segment_contract(self):
        segment = _segments(11, 1)[0]

        legacy = _make_allocator()
        _install_mappings(legacy, [segment])
        legacy.free_swa(segment)

        allocator = _make_allocator()
        _install_mappings(allocator, [segment])
        allocator.free_swa(segment, start_pos=1)
        self.assertStateEqual(_state(legacy), _state(allocator))

        allocator = _make_allocator()
        before = _state(allocator)
        with self.assertRaisesRegex(RuntimeError, "fully mapped to one SWA page"):
            allocator.free_swa(segment, start_pos=0)
        self.assertStateEqual(before, _state(allocator))

    def test_segment_mapping_contract(self):
        segment = _segments(12, 1, pages_per_segment=1)[0]

        for name in ("partial", "cross_page"):
            allocator = _make_allocator()
            swa_indices = allocator.swa_attn_allocator.alloc(2 * PAGE_SIZE)
            assert swa_indices is not None
            mapping = swa_indices[:PAGE_SIZE].clone()
            mapping[17 if name == "partial" else -1] = (
                0 if name == "partial" else swa_indices[PAGE_SIZE]
            )
            allocator.set_full_to_swa_mapping(segment, mapping)
            with (
                self.subTest(name),
                self.assertRaisesRegex(RuntimeError, "fully mapped to one SWA page"),
            ):
                allocator.free_swa(segment, start_pos=0)

        two_pages = _segments(13, 1, pages_per_segment=2)[0]
        allocator = _make_allocator()
        swa_indices = allocator.swa_attn_allocator.alloc(PAGE_SIZE)
        assert swa_indices is not None
        allocator.set_full_to_swa_mapping(two_pages[:PAGE_SIZE], swa_indices)
        allocator.set_full_to_swa_mapping(two_pages[PAGE_SIZE:], swa_indices)
        with self.assertRaisesRegex(RuntimeError, "valid and unique"):
            allocator.free_swa(two_pages, start_pos=0)

    def test_grouped_out_of_window_eviction_avoids_unique(self):
        def setup():
            allocator = _make_allocator()
            row = torch.arange(6 * PAGE_SIZE, dtype=torch.int64) + PAGE_SIZE
            swa_indices = allocator.swa_attn_allocator.alloc(row.numel())
            assert swa_indices is not None
            allocator.set_full_to_swa_mapping(row, swa_indices)
            return allocator, row

        legacy, row = setup()
        legacy.free_swa(row[: 4 * PAGE_SIZE])

        fast, row = setup()
        req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_kv=True,
                cache_protected_len=0,
                swa_dead_lo=lambda page_size: 0,
                req_pool_idx=0,
                swa_evicted_seqlen=0,
            )
        )
        req_to_token_pool = SimpleNamespace(req_to_token=row.unsqueeze(0))
        available_before = fast.swa_available_size()
        fast.free_group_begin()
        with patch("torch.unique", side_effect=AssertionError("unexpected unique")):
            for pre_len in (3 * PAGE_SIZE, 5 * PAGE_SIZE):
                free_swa_out_of_window_slots(
                    req,
                    pre_len,
                    sliding_window_size=PAGE_SIZE,
                    page_size=PAGE_SIZE,
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=fast,
                )
            self.assertEqual(fast.swa_available_size(), available_before)
            fast.free_group_end()

        self.assertEqual(req.kv.swa_evicted_seqlen, 4 * PAGE_SIZE)
        self.assertStateEqual(_state(legacy), _state(fast))

    def test_tail_only_mapping_frees_from_the_tail_floor(self):
        """alloc_extend_swa_tail leaves the head unmapped; out-of-window frees
        start at the tail floor (decode preallocation sets swa_evicted_seqlen to
        it), so the fast path never sees the head and needs no global fallback."""

        def setup():
            allocator = _make_allocator()
            full_indices = torch.arange(2 * PAGE_SIZE) + PAGE_SIZE
            swa_indices = allocator.swa_attn_allocator.alloc(PAGE_SIZE)
            assert swa_indices is not None
            allocator.full_attn_allocator.alloc_extend = MagicMock(
                return_value=full_indices
            )
            allocator.swa_attn_allocator.alloc_extend = MagicMock(
                return_value=swa_indices
            )
            allocated = allocator.alloc_extend_swa_tail(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([2 * PAGE_SIZE]),
                seq_lens_cpu=torch.tensor([2 * PAGE_SIZE]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=2 * PAGE_SIZE,
                swa_tail_len=PAGE_SIZE,
            )
            self.assertFalse(hasattr(allocator, "_swa_mapping_may_be_partial"))
            self.assertTrue(torch.equal(allocated, full_indices))
            return allocator, full_indices

        legacy, full_indices = setup()
        legacy.free_swa(full_indices)
        fast, full_indices = setup()
        with patch("torch.unique", side_effect=AssertionError("unexpected unique")):
            fast.free_swa(full_indices[PAGE_SIZE:], start_pos=PAGE_SIZE)
        # The unmapped head is released by the cache's legacy free, a no-op for SWA.
        fast.free_swa(full_indices[:PAGE_SIZE])
        self.assertStateEqual(_state(legacy), _state(fast))

    def test_start_pos_is_accepted_by_every_swa_allocator(self):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        for cls in (
            SWATokenToKVPoolAllocator,
            PureSWATokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
        ):
            with self.subTest(cls.__name__):
                param = inspect.signature(cls.free_swa).parameters["start_pos"]
                self.assertIs(param.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(param.default)

    def test_pure_swa_allocator_ignores_start_pos(self):
        indices = torch.tensor([0, 3, 4])
        allocator = object.__new__(PureSWATokenToKVPoolAllocator)
        allocator.free_group = None
        allocator.swa_attn_allocator = MagicMock()
        allocator.free_swa(indices, start_pos=0)
        (freed,), _ = allocator.swa_attn_allocator.free.call_args
        self.assertTrue(torch.equal(freed, torch.tensor([3, 4])))


if __name__ == "__main__":
    unittest.main()
