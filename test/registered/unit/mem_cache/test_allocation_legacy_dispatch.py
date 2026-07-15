from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from typing import Any
from unittest import mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import (  # noqa: E402
    alloc_for_decode,
    alloc_for_extend,
)

_LEGACY_MODULE = "sglang.srt.hardware_backend.npu.allocation_legacy"


class _ReachedPageAlignedPath(Exception):
    pass


def _make_batch(*, uses_legacy: bool) -> Any:
    """A batch whose only working method is the first one the aligned path calls."""

    def maybe_evict_swa() -> None:
        raise _ReachedPageAlignedPath()

    return types.SimpleNamespace(
        token_to_kv_pool_allocator=types.SimpleNamespace(
            uses_legacy_real_length_alloc=uses_legacy,
            page_size=4,
        ),
        maybe_evict_swa=maybe_evict_swa,
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

    def test_an_allocator_with_no_platform_identity_takes_the_page_aligned_path(self):
        """Dispatch must read the declaration alone: this allocator names no platform at all."""
        batch = _make_batch(uses_legacy=False)

        with mock.patch(f"{_LEGACY_MODULE}.alloc_for_extend_legacy") as legacy:
            with self.assertRaises(_ReachedPageAlignedPath):
                alloc_for_extend(batch)

        legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
