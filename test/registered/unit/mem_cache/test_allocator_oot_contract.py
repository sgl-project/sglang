from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest
from typing import Any
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import alloc_for_extend  # noqa: E402
from sglang.srt.mem_cache.allocator import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)

_LEGACY_MODULE = "sglang.srt.hardware_backend.npu.allocation_legacy"


class _FakeOutOfTreePagedAllocator(PagedTokenToKVPoolAllocator):
    """A tree-external platform's allocator, as get_paged_allocator_cls hands it back.

    Deliberately overrides nothing: this stands for the out-of-tree subclass that
    inherits whatever the base offers. Repo greps cannot see such a class, so the
    contract it lands on has to hold by construction.
    """


class _LegacyOutOfTreePagedAllocator(PagedTokenToKVPoolAllocator):
    """An out-of-tree allocator that overrides the real-length entry but never declares it."""

    def alloc_extend(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("dispatch must not reach alloc_extend on its own")


def _make_allocator(cls: type, *, page_size: int = 16) -> Any:
    return cls(
        size=page_size * 8,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=types.SimpleNamespace(),
        need_sort=False,
    )


class TestOutOfTreePagedAllocatorContract(CustomTestCase):
    def test_out_of_tree_paged_subclass_alloc_returns_whole_pages(self):
        """An out-of-tree subclass that does nothing must still satisfy the aligned contract."""
        allocator = _make_allocator(_FakeOutOfTreePagedAllocator)

        out = allocator.alloc(32)

        self.assertIsNotNone(out)
        self.assertEqual(int(out.numel()), 32)
        pages = sorted(set((out // 16).tolist()))
        self.assertEqual(len(pages), 2)
        for page in pages:
            slots = sorted(int(s) for s in out if s // 16 == page)
            self.assertEqual(slots, [page * 16 + i for i in range(16)])

    def test_out_of_tree_paged_subclass_inherits_the_page_aligned_declaration(self):
        """Inheriting legacy by accident would route a whole platform down the wrong path."""
        self.assertIs(
            _FakeOutOfTreePagedAllocator.uses_legacy_real_length_alloc, False
        )


class TestLegacyDispatchReadsTheDeclarationOnly(CustomTestCase):
    """The declaration is the only dispatch criterion -- not the shape of the class.

    An allocator carrying a real-length alloc_extend is not thereby a legacy
    allocator: in-tree HiSparse and DSV4 both keep one for the legacy module to
    call. If dispatch ever sniffed for the method instead of reading the flag,
    every such allocator would be misrouted, and this is the case that reds.
    """

    def test_overriding_alloc_extend_does_not_by_itself_select_the_legacy_path(self):
        """Sniffing for alloc_extend instead of the flag would silently misroute callers."""
        allocator = _make_allocator(_LegacyOutOfTreePagedAllocator)
        self.assertTrue(hasattr(allocator, "alloc_extend"))
        self.assertIs(allocator.uses_legacy_real_length_alloc, False)

        batch = types.SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            maybe_evict_swa=mock.MagicMock(side_effect=RuntimeError("aligned path")),
        )

        with mock.patch(
            f"{_LEGACY_MODULE}.alloc_for_extend_legacy"
        ) as legacy, self.assertRaises(RuntimeError):
            alloc_for_extend(batch)

        legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
