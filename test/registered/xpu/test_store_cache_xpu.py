"""
Verifies the fused ``store_cache_xpu`` KV-cache write path on Intel XPU.

This branch wires the fused SYCL ``store_cache_xpu`` kernel (from
sgl-kernel-xpu) into sglang's KV-cache writer
``sglang.srt.mem_cache.memory_pool._set_kv_buffer_impl``. On XPU that
dispatch replaces 2x ``index_put`` with a single kernel launch.

The tests exercise sglang's own dispatch (not the kernel in isolation), so
they fail if the wiring regresses to the ``index_put`` fallback:

  - ``test_parity_*``     : fused write matches an ``index_put`` reference.
  - ``test_dispatches_*`` : the fused kernel is actually the path taken.
  - ``test_single_token`` : the common decode (1 token) case.

Run from test/registered::

  python3 -m unittest xpu.test_store_cache_xpu

Requires Intel XPU and an sgl-kernel-xpu build that provides
``store_cache_xpu`` (built from sgl-kernel-xpu main).
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.utils import is_xpu
from sglang.test.test_utils import CustomTestCase


def _reference_store(k, v, k_cache, v_cache, indices):
    """Naive index_put write — the path the fused kernel replaces."""
    k_cache[indices] = k
    v_cache[indices] = v


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestStoreCacheXPU(CustomTestCase):
    """store_cache_xpu, exercised through sglang's _set_kv_buffer_impl."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.mem_cache.memory_pool import _get_store_cache_xpu

        if _get_store_cache_xpu() is None:
            raise unittest.SkipTest(
                "sgl_kernel.store_cache_xpu not available; build sgl-kernel-xpu "
                "from main (pip install -ve .) to enable the fused KV-cache path"
            )

    def _store(self, k, v, k_cache, v_cache, indices):
        """Invoke sglang's KV-cache writer (the integration point)."""
        from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl

        row_dim = k.shape[-1]
        _set_kv_buffer_impl(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
            k.dtype,
            torch.xpu,
            alt_stream=None,
            same_kv_dim=True,
        )
        torch.xpu.synchronize()

    def _assert_parity(self, num_tokens, row_dim, dtype):
        torch.manual_seed(42)
        cache_size = 2048

        k = torch.randn(num_tokens, row_dim, dtype=dtype, device="xpu")
        v = torch.randn(num_tokens, row_dim, dtype=dtype, device="xpu")
        indices = torch.randperm(cache_size, device="xpu")[:num_tokens].to(torch.int64)

        k_ref = torch.zeros(cache_size, row_dim, dtype=dtype, device="xpu")
        v_ref = torch.zeros_like(k_ref)
        k_test = torch.zeros_like(k_ref)
        v_test = torch.zeros_like(k_ref)

        _reference_store(k, v, k_ref, v_ref, indices)
        self._store(k, v, k_test, v_test, indices)

        torch.testing.assert_close(k_test, k_ref)
        torch.testing.assert_close(v_test, v_ref)

    def test_parity_shapes(self):
        """Fused write matches index_put across token counts and row dims."""
        for num_tokens in (1, 4, 32, 128):
            for row_dim in (128, 256, 512, 1024):
                with self.subTest(num_tokens=num_tokens, row_dim=row_dim):
                    self._assert_parity(num_tokens, row_dim, torch.bfloat16)

    def test_parity_dtypes(self):
        """Both KV-cache dtypes write correctly."""
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self._assert_parity(32, 256, dtype)

    def test_single_token(self):
        """Single-token decode (the most common runtime case)."""
        torch.manual_seed(0)
        row_dim, cache_size = 512, 4096

        k = torch.randn(1, row_dim, dtype=torch.bfloat16, device="xpu")
        v = torch.randn(1, row_dim, dtype=torch.bfloat16, device="xpu")
        indices = torch.tensor([42], dtype=torch.int64, device="xpu")

        k_cache = torch.zeros(cache_size, row_dim, dtype=torch.bfloat16, device="xpu")
        v_cache = torch.zeros_like(k_cache)

        self._store(k, v, k_cache, v_cache, indices)

        torch.testing.assert_close(k_cache[42], k[0])
        torch.testing.assert_close(v_cache[42], v[0])

    def test_dispatches_to_fused_kernel(self):
        """sglang must take the fused path on XPU, not the index_put fallback.

        Wrap the kernel and assert it is invoked exactly once. Guards against
        the dispatch silently regressing (e.g. if can_use_store_cache starts
        gating XPU again, which can't JIT-compile the CUDA kernel).
        """
        import sgl_kernel

        from sglang.srt.mem_cache import memory_pool

        calls = {"n": 0}
        original = sgl_kernel.store_cache_xpu

        def counting_store(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        sgl_kernel.store_cache_xpu = counting_store
        memory_pool._get_store_cache_xpu.cache_clear()
        try:
            torch.manual_seed(7)
            row_dim, cache_size, num_tokens = 256, 1024, 8
            k = torch.randn(num_tokens, row_dim, dtype=torch.bfloat16, device="xpu")
            v = torch.randn(num_tokens, row_dim, dtype=torch.bfloat16, device="xpu")
            indices = torch.randperm(cache_size, device="xpu")[:num_tokens].to(
                torch.int64
            )
            k_cache = torch.zeros(
                cache_size, row_dim, dtype=torch.bfloat16, device="xpu"
            )
            v_cache = torch.zeros_like(k_cache)

            self._store(k, v, k_cache, v_cache, indices)
        finally:
            sgl_kernel.store_cache_xpu = original
            memory_pool._get_store_cache_xpu.cache_clear()

        self.assertEqual(
            calls["n"],
            1,
            "expected _set_kv_buffer_impl to call the fused store_cache_xpu "
            "exactly once on XPU; it likely fell back to index_put",
        )


from sglang.test.ci.ci_register import register_xpu_ci

# Pure unit test (no server); fast and runs on the 1-GPU XPU runner.
register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")

if __name__ == "__main__":
    unittest.main()
