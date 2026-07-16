from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from typing import Any
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import (  # noqa: E402
    alloc_for_decode_prealloc,
    alloc_for_decode_prealloc_hisparse,
)
from sglang.srt.mem_cache.allocation import (  # noqa: E402
    alloc_for_decode,
    alloc_for_extend,
    alloc_for_spec_decode,
)

_LEGACY_MODULE = "sglang.srt.hardware_backend.npu.allocation_legacy"


class _ReachedPageAlignedPath(Exception):
    pass


def _make_allocator(*, uses_legacy: bool) -> Any:
    def alloc(need_size: int) -> None:
        raise _ReachedPageAlignedPath()

    return types.SimpleNamespace(
        uses_legacy_real_length_alloc=uses_legacy,
        page_size=4,
        alloc=alloc,
        alloc_logical_only=alloc,
    )


def _make_batch(*, uses_legacy: bool) -> Any:
    def maybe_evict_swa() -> None:
        raise _ReachedPageAlignedPath()

    return types.SimpleNamespace(
        token_to_kv_pool_allocator=_make_allocator(uses_legacy=uses_legacy),
        maybe_evict_swa=maybe_evict_swa,
    )


def _make_tree_cache(*, uses_legacy: bool) -> Any:
    return types.SimpleNamespace(
        token_to_kv_pool_allocator=_make_allocator(uses_legacy=uses_legacy)
    )


class TestLegacyAllocDispatch(CustomTestCase):
    def test_extend_routes_to_legacy_when_the_allocator_declares_it(self):
        """An allocator that cannot take whole-page alloc must never reach the aligned path."""
        batch = _make_batch(uses_legacy=True)
        sentinel = object()

        with mock.patch(
            f"{_LEGACY_MODULE}.alloc_for_extend_legacy", return_value=sentinel
        ) as legacy:
            result = alloc_for_extend(batch)

        legacy.assert_called_once_with(batch)
        self.assertIs(result, sentinel)

    def test_decode_routes_to_legacy_when_the_allocator_declares_it(self):
        """Decode must dispatch on the same declaration as extend, or a request splits paths."""
        batch = _make_batch(uses_legacy=True)
        sentinel = object()

        with mock.patch(
            f"{_LEGACY_MODULE}.alloc_for_decode_legacy", return_value=sentinel
        ) as legacy:
            result = alloc_for_decode(batch, token_per_req=1)

        legacy.assert_called_once_with(batch, 1)
        self.assertIs(result, sentinel)

    def test_spec_decode_routes_to_legacy_when_the_allocator_declares_it(self):
        """Speculative decode allocates from the same pool and must honour the same opt-out."""
        tree_cache = _make_tree_cache(uses_legacy=True)
        empty = torch.zeros(0, dtype=torch.int64)

        with mock.patch(f"{_LEGACY_MODULE}.alloc_for_spec_decode_legacy") as legacy:
            alloc_for_spec_decode(
                tree_cache,
                types.SimpleNamespace(req_to_token=None),
                reqs=[],
                req_pool_indices=empty,
                cur_kv_lens=empty,
                cur_kv_lens_cpu=empty,
                nxt_kv_lens=empty,
                nxt_kv_lens_cpu=empty,
                batch=None,
            )

        legacy.assert_called_once()

    def test_prealloc_routes_to_legacy_when_the_allocator_declares_it(self):
        """PD preallocation admits requests before the scheduler sees them; it must opt out too."""
        allocator = _make_allocator(uses_legacy=True)
        sentinel = object()

        with mock.patch(
            f"{_LEGACY_MODULE}.alloc_for_decode_prealloc_legacy", return_value=sentinel
        ) as legacy:
            result = alloc_for_decode_prealloc(
                allocator,
                types.SimpleNamespace(),
                req=types.SimpleNamespace(kv=None),
                fill_len=10,
                total_prefix_len=0,
                prefix_len=0,
                prefix_indices=None,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        legacy.assert_called_once()
        self.assertIs(result, sentinel)

    def test_hisparse_prealloc_routes_to_legacy_when_the_allocator_declares_it(self):
        """The hisparse prealloc entry is a separate function and so needs its own dispatch."""
        allocator = _make_allocator(uses_legacy=True)
        sentinel = object()

        with mock.patch(
            f"{_LEGACY_MODULE}.alloc_for_decode_prealloc_hisparse_legacy",
            return_value=sentinel,
        ) as legacy:
            result = alloc_for_decode_prealloc_hisparse(
                allocator,
                types.SimpleNamespace(),
                req=types.SimpleNamespace(kv=None),
                fill_len=10,
                total_prefix_len=0,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        legacy.assert_called_once()
        self.assertIs(result, sentinel)

    def test_extend_takes_the_page_aligned_path_without_the_declaration(self):
        """The other half: a dispatch degraded to always-legacy would pass every case above."""
        batch = _make_batch(uses_legacy=False)

        with mock.patch(f"{_LEGACY_MODULE}.alloc_for_extend_legacy") as legacy:
            with self.assertRaises(_ReachedPageAlignedPath):
                alloc_for_extend(batch)

        legacy.assert_not_called()

    def test_prealloc_takes_the_page_aligned_path_without_the_declaration(self):
        """Same negative branch for the PD entry, whose aligned body calls alloc directly."""
        allocator = _make_allocator(uses_legacy=False)

        with mock.patch(f"{_LEGACY_MODULE}.alloc_for_decode_prealloc_legacy") as legacy:
            with self.assertRaises(_ReachedPageAlignedPath):
                alloc_for_decode_prealloc(
                    allocator,
                    types.SimpleNamespace(),
                    req=types.SimpleNamespace(kv=None),
                    fill_len=10,
                    total_prefix_len=0,
                    prefix_len=0,
                    prefix_indices=None,
                    uses_swa_tail=False,
                    swa_tail_len=0,
                )

        legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
