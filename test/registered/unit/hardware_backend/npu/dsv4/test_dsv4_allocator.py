import inspect
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (  # noqa: E402
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.base import (  # noqa: E402
    BaseTokenToKVPoolAllocator,
)


def _uninitialized_allocator() -> DSV4NPUTokenToKVPoolAllocator:
    return object.__new__(DSV4NPUTokenToKVPoolAllocator)


class TestDSV4AllocIsFailLoud(CustomTestCase):
    def test_alloc_raises_instead_of_silently_allocating_partial_pools(self):
        """Returning a tensor here means the c4/c128 and state pools were never allocated."""
        allocator = _uninitialized_allocator()

        with self.assertRaises(NotImplementedError) as ctx:
            allocator.alloc(64)

        self.assertIn("DSV4OutCacheLoc", str(ctx.exception))

    def test_alloc_does_not_return_a_tensor(self):
        """A future DSV4-aware alloc must return the bundle, not the SWA tensor."""
        allocator = _uninitialized_allocator()

        result = None
        try:
            result = allocator.alloc(64)
        except NotImplementedError:
            pass

        self.assertNotIsInstance(result, torch.Tensor)


class TestDSV4LegacySwaTailEntry(CustomTestCase):
    def test_the_legacy_call_site_matches_the_legacy_entry(self):
        """A signature drift on either side would only surface as a TypeError on NPU."""
        entry = inspect.signature(
            DSV4NPUTokenToKVPoolAllocator.alloc_extend_swa_tail_legacy
        )
        entry.bind(
            _uninitialized_allocator(),
            prefix_lens=object(),
            prefix_lens_cpu=object(),
            seq_lens=object(),
            seq_lens_cpu=object(),
            last_loc=object(),
            extend_num_tokens=0,
            swa_tail_len=0,
        )

    def test_the_legacy_entry_is_not_the_page_aligned_one(self):
        """Aliasing the two names would feed batch tensors to the seq_len/swa_tail_len entry."""
        aligned = inspect.signature(DSV4NPUTokenToKVPoolAllocator.alloc_extend_swa_tail)
        self.assertEqual(list(aligned.parameters), ["self", "seq_len", "swa_tail_len"])
        self.assertNotEqual(
            list(aligned.parameters),
            list(
                inspect.signature(
                    DSV4NPUTokenToKVPoolAllocator.alloc_extend_swa_tail_legacy
                ).parameters
            ),
        )


class TestDSV4LegacyDeclaration(CustomTestCase):
    def test_uses_legacy_real_length_alloc_is_true(self):
        """DSV4-NPU is the one allocator driven with real lengths; nothing else declares it."""
        self.assertIs(DSV4NPUTokenToKVPoolAllocator.uses_legacy_real_length_alloc, True)

    def test_the_base_default_stays_false(self):
        """Flipping the default would route every allocator down the legacy path."""
        self.assertIs(BaseTokenToKVPoolAllocator.uses_legacy_real_length_alloc, False)


if __name__ == "__main__":
    unittest.main()
