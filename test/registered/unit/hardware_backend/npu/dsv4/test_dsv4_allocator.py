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
    """The entries under test refuse before touching any state, so skip __init__."""
    return object.__new__(DSV4NPUTokenToKVPoolAllocator)


class TestDSV4AllocIsFailLoud(CustomTestCase):
    """DSV4 inherits SWA's alloc, which only knows the full and SWA pools.

    A DSV4 allocation must also cover the c4/c128 KV and compress-state pools and
    return a DSV4OutCacheLoc bundle. Inheriting alloc silently would allocate two
    of the five pools and return a bare tensor, and the caller would read
    batch.out_cache_loc_dsv4 as None far away from here. The SWA page_size == 1
    assert used to block this by accident; now that page_size > 1 is allowed, the
    refusal has to be deliberate.
    """

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
    """The legacy module preallocates the SWA tail with the old real-length call.

    DSV4 inherits the page-aligned alloc_extend_swa_tail from SWA, which the
    legacy module cannot use: it passes batch tensors, not two ints. The legacy
    entry keeps the old signature so that call keeps resolving.
    """

    def test_the_legacy_call_site_matches_the_legacy_entry(self):
        """A signature drift on either side would only surface as a TypeError on NPU."""
        entry = inspect.signature(
            DSV4NPUTokenToKVPoolAllocator.alloc_extend_swa_tail_legacy
        )
        # Exactly the keywords alloc_for_decode_prealloc_legacy passes.
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
