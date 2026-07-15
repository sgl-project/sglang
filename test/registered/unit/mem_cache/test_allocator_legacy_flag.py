from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocator import (  # noqa: E402
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (  # noqa: E402
    SWATokenToKVPoolAllocator,
)


class TestUsesLegacyRealLengthAlloc(CustomTestCase):
    def test_base_allocator_declares_the_page_aligned_contract(self):
        """The default must be the aligned contract, so only opt-outs carry the burden."""
        self.assertIs(BaseTokenToKVPoolAllocator.uses_legacy_real_length_alloc, False)

    def test_in_tree_allocators_inherit_the_page_aligned_contract(self):
        """An in-tree allocator silently flipping to legacy would disable the guardrail."""
        for allocator_cls in (
            TokenToKVPoolAllocator,
            PagedTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
        ):
            with self.subTest(allocator=allocator_cls.__name__):
                self.assertIs(allocator_cls.uses_legacy_real_length_alloc, False)


if __name__ == "__main__":
    unittest.main()
