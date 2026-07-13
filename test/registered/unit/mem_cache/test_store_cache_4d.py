"""Parity tests for the `store_cache_4d` Triton kernel.

The kernel writes K/V into the 4-D page-major envelope view. These tests prove
it produces byte-identical output to the legacy advanced-indexing path on
representative fixtures:

  - ``page_size = 1`` (envelope-degenerate, the critical compatibility case)
  - ``page_size > 1`` (layer-major within page)
  - both int32 and int64 ``loc`` dtypes
  - bf16 and fp8_e5m2 view dtypes
  - asymmetric ``head_dim != v_head_dim``
  - empty ``loc`` (no-op)

Skipped on CPU — Triton requires a GPU.

    python -m pytest test/registered/unit/mem_cache/test_store_cache_4d.py -v
"""

import importlib.util
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()
# The set_kv_buffer integration test needs UnifiedMHATokenToKVPool, which only
# exists once the shared-KV-pool feature lands; skip it where absent.
_HAS_SHARED_POOL = (
    importlib.util.find_spec("sglang.srt.mem_cache.unified_memory_pool") is not None
)

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


def _legacy_advanced_indexing_write(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
) -> None:
    """Reference implementation: the legacy bypass-super() advanced-indexing
    path that the Triton kernel replaces. Used as the byte-identity oracle
    for the parity tests below.
    """
    if page_size == 1:
        k_view[loc, 0] = cache_k
        v_view[loc, 0] = cache_v
    else:
        page_id = loc // page_size
        tok_in_p = loc % page_size
        k_view[page_id, tok_in_p] = cache_k
        v_view[page_id, tok_in_p] = cache_v


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require CUDA")
class TestStoreCache4D(unittest.TestCase):
    """Byte-identity parity vs the legacy advanced-indexing write path."""

    def _make_view_and_cache(
        self,
        num_pages: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        v_head_dim: int,
        N: int,
        dtype: torch.dtype = torch.bfloat16,
        loc_dtype: torch.dtype = torch.int64,
        seed: int = 0xC0FFEE,
    ):
        torch.manual_seed(seed)
        # The unified memory pool's views are 4-D `(num_pages, page_size, head_num,
        # head_dim)` with the trailing two dims contiguous. We allocate two
        # independent contiguous buffers (one for the kernel-under-test,
        # one as the legacy-path target) so we can compare them.
        k_view = torch.zeros(
            (num_pages, page_size, head_num, head_dim),
            dtype=dtype,
            device="cuda",
        )
        v_view = torch.zeros(
            (num_pages, page_size, head_num, v_head_dim),
            dtype=dtype,
            device="cuda",
        )
        cache_k = torch.randn(
            (N, head_num, head_dim), dtype=torch.float32, device="cuda"
        ).to(dtype)
        cache_v = torch.randn(
            (N, head_num, v_head_dim), dtype=torch.float32, device="cuda"
        ).to(dtype)
        # Valid loc values in [0, num_pages * page_size); generate without
        # duplicates so the comparison is unambiguous (advanced-indexing
        # with duplicates is order-undefined for both paths).
        total_slots = num_pages * page_size
        assert N <= total_slots
        loc = torch.randperm(total_slots, device="cuda")[:N].to(loc_dtype)
        return k_view, v_view, cache_k, cache_v, loc

    def _check_parity(
        self,
        num_pages: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        v_head_dim: int,
        N: int,
        dtype: torch.dtype = torch.bfloat16,
        loc_dtype: torch.dtype = torch.int64,
    ):
        from sglang.kernels.ops.kvcache.cache_move import store_cache_4d

        # Two independent target buffers — one for the kernel, one for the
        # legacy reference path.
        k_kernel, v_kernel, cache_k, cache_v, loc = self._make_view_and_cache(
            num_pages,
            page_size,
            head_num,
            head_dim,
            v_head_dim,
            N,
            dtype=dtype,
            loc_dtype=loc_dtype,
        )
        k_legacy = k_kernel.clone()
        v_legacy = v_kernel.clone()

        # Kernel-under-test
        store_cache_4d(k_kernel, v_kernel, cache_k, cache_v, loc, page_size)
        # Legacy reference
        _legacy_advanced_indexing_write(
            k_legacy, v_legacy, cache_k, cache_v, loc, page_size
        )

        # Byte-identical comparison — the kernel must reproduce the
        # advanced-indexing path bit-for-bit, NOT just numerically close.
        # For fp8 dtypes, torch.equal works on the integer bit pattern.
        self.assertTrue(
            torch.equal(k_kernel, k_legacy),
            f"K view mismatch: ps={page_size}, dtype={dtype}, "
            f"loc_dtype={loc_dtype}, N={N}",
        )
        self.assertTrue(
            torch.equal(v_kernel, v_legacy),
            f"V view mismatch: ps={page_size}, dtype={dtype}, "
            f"loc_dtype={loc_dtype}, N={N}",
        )

    # ---- Test 1: ps=1 envelope-degenerate (the critical compat case) ----

    def test_store_cache_4d_ps1_byte_identical(self):
        """At page_size=1 the kernel constexpr-folds to the slot-major
        envelope view. Output must be byte-identical to advanced indexing.
        This protects against byte-layout regression."""
        self._check_parity(
            num_pages=64,
            page_size=1,
            head_num=4,
            head_dim=128,
            v_head_dim=128,
            N=16,
        )

    # ---- Test 2: ps>1 layer-major within page ----

    def test_store_cache_4d_ps_gt1_byte_identical(self):
        """At page_size > 1 the kernel splits loc into (page_id, tok_in_p)
        and writes via the 4-D stride. Output must match the equivalent
        advanced-indexing write."""
        self._check_parity(
            num_pages=8,
            page_size=64,
            head_num=4,
            head_dim=128,
            v_head_dim=128,
            N=128,
        )

    # ---- Test 3: int32 loc dtype ----

    def test_store_cache_4d_int32_loc(self):
        """The SWA-side path passes int32 loc (matches the SWA Triton
        kernel contract). PyTorch advanced indexing tolerates either
        int32 or int64; the kernel must too."""
        self._check_parity(
            num_pages=32,
            page_size=1,
            head_num=4,
            head_dim=64,
            v_head_dim=64,
            N=10,
            loc_dtype=torch.int32,
        )

    # ---- Test 4: int64 loc dtype (already exercised, explicit) ----

    # ---- Test 5: bf16 dtype (the production case) ----

    # ---- Test 6: fp8_e5m2 dtype ----

    def test_store_cache_4d_dtype_fp8_e5m2(self):
        """fp8_e5m2 is used for KV-cache quantization. Caller is responsible
        for the cast; the kernel sees same-dtype source and destination."""
        self._check_parity(
            num_pages=16,
            page_size=64,
            head_num=4,
            head_dim=128,
            v_head_dim=128,
            N=64,
            dtype=torch.float8_e5m2,
        )

    # ---- Test 7: empty loc (no-op) ----

    def test_store_cache_4d_empty_loc(self):
        """N=0 must be a no-op: no kernel launch, no exception, no buffer
        mutation."""
        from sglang.kernels.ops.kvcache.cache_move import store_cache_4d

        k_view = torch.zeros((8, 4, 4, 64), dtype=torch.bfloat16, device="cuda")
        v_view = torch.zeros((8, 4, 4, 64), dtype=torch.bfloat16, device="cuda")
        k_before = k_view.clone()
        v_before = v_view.clone()
        cache_k = torch.empty((0, 4, 64), dtype=torch.bfloat16, device="cuda")
        cache_v = torch.empty((0, 4, 64), dtype=torch.bfloat16, device="cuda")
        loc = torch.empty((0,), dtype=torch.int64, device="cuda")

        store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size=4)

        # Buffers must be unchanged.
        self.assertTrue(torch.equal(k_view, k_before))
        self.assertTrue(torch.equal(v_view, v_before))

    # ---- Test 8: head_dim != v_head_dim (asymmetric, e.g. MLA-style) ----

    def test_store_cache_4d_v_head_dim_differs(self):
        """When v_head_dim != head_dim, the kernel's K and V branches use
        different per-token strides. Exercises the stride_k_tok ≠
        stride_v_tok branch."""
        self._check_parity(
            num_pages=8,
            page_size=16,
            head_num=2,
            head_dim=128,
            v_head_dim=64,
            N=16,
        )


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require CUDA")
class TestStoreCache4DAssertions(unittest.TestCase):
    """The wrapper's contract assertions must fire on bad inputs."""

    def test_rejects_non_contiguous_view_trailing_dim(self):
        """Wrapper requires `stride[-1] == 1` and `stride[-2] == head_dim`
        (the trailing two dims must be contiguous). A permutation that
        breaks this should trigger AssertionError."""
        from sglang.kernels.ops.kvcache.cache_move import store_cache_4d

        # Build a 4-D view, then permute the last two dims → trailing
        # contiguity violated.
        k_view = torch.zeros(
            (4, 4, 4, 64), dtype=torch.bfloat16, device="cuda"
        ).permute(
            0, 1, 3, 2
        )  # now shape (4, 4, 64, 4); strides broken
        v_view = torch.zeros((4, 4, 4, 64), dtype=torch.bfloat16, device="cuda")
        cache_k = torch.zeros((2, 4, 64), dtype=torch.bfloat16, device="cuda")
        cache_v = torch.zeros((2, 4, 64), dtype=torch.bfloat16, device="cuda")
        loc = torch.arange(2, dtype=torch.int64, device="cuda")
        with self.assertRaises(AssertionError):
            store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size=4)

    def test_rejects_dtype_mismatch(self):
        """All four tensors must share a dtype; the caller is responsible
        for any cast before the call."""
        from sglang.kernels.ops.kvcache.cache_move import store_cache_4d

        k_view = torch.zeros((4, 4, 4, 64), dtype=torch.bfloat16, device="cuda")
        v_view = torch.zeros((4, 4, 4, 64), dtype=torch.bfloat16, device="cuda")
        cache_k = torch.zeros((2, 4, 64), dtype=torch.float16, device="cuda")
        cache_v = torch.zeros((2, 4, 64), dtype=torch.bfloat16, device="cuda")
        loc = torch.arange(2, dtype=torch.int64, device="cuda")
        with self.assertRaises(AssertionError):
            store_cache_4d(k_view, v_view, cache_k, cache_v, loc, page_size=4)


@unittest.skipUnless(
    _HAS_CUDA and _HAS_SHARED_POOL,
    "Triton kernels require CUDA; UnifiedMHATokenToKVPool required",
)
class TestStoreCache4DThroughSetKVBuffer(unittest.TestCase):
    """Integration parity test — exercises the kernel through the FULL
    ``UnifiedMHATokenToKVPool.set_kv_buffer`` path (the direct PHYSICAL write +
    the dtype cast; the pool no longer translates). Confirms it produces
    bit-identical output to a PyTorch advanced-indexing reference write.
    """

    def _build_pool(self, page_size: int):
        """Build a small UnifiedMHATokenToKVPool. The pool writes PHYSICAL locs
        directly (no allocator / v2p translate), so `set_kv_buffer` receives the
        already-physical write location."""
        import torch as _t

        from sglang.srt.mem_cache.unified_memory_pool import (
            MHASubPoolSpec,
            UnifiedKVPool,
            UnifiedMHATokenToKVPool,
        )

        spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=4,
            head_dim=64,
            store_dtype=_t.bfloat16,
            grow_direction="up",
        )
        total = spec.entry_bytes() * 64
        # Use a peer to satisfy the two-sub-pool contract.
        peer = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=4,
            head_dim=64,
            store_dtype=_t.bfloat16,
            grow_direction="down",
        )
        pool = UnifiedKVPool(
            total_bytes=total + peer.entry_bytes() * 16,
            sub_pool_specs=[spec, peer],
            device="cuda",
            enable_memory_saver=False,
            page_size=page_size,
        )
        kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=pool,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=0,
            end_layer=2,
            enable_alt_stream=False,
        )

        return kv_pool

    def _run_set_kv_buffer_and_compare(self, page_size: int):
        import torch as _t

        kv_pool = self._build_pool(page_size)

        # A fake `layer` object with the minimum interface
        # `set_kv_buffer` reads: `.layer_id`.
        class _FakeLayer:
            layer_id = 0

        layer = _FakeLayer()
        head_num, head_dim = 4, 64
        N = 16
        # Generate valid loc in range [0, num_pages * page_size).
        num_pages = kv_pool.k_buffer[0].shape[0]
        total = num_pages * page_size
        assert N <= total
        loc = _t.randperm(total, device="cuda")[:N].to(_t.int64)
        cache_k = _t.randn((N, head_num, head_dim), dtype=_t.bfloat16, device="cuda")
        cache_v = _t.randn((N, head_num, head_dim), dtype=_t.bfloat16, device="cuda")

        # Production path: the Triton `store_cache_4d` kernel via set_kv_buffer.
        kv_pool.set_kv_buffer(layer, loc, cache_k.clone(), cache_v.clone())
        k_kernel = kv_pool.k_buffer[0].clone()
        v_kernel = kv_pool.v_buffer[0].clone()

        # Reference: PyTorch advanced-indexing into a fresh view at the same
        # (physical) loc, with no dtype cast (store_dtype == dtype) — the exact
        # write the kernel performs.
        kv_pool.k_buffer[0].zero_()
        kv_pool.v_buffer[0].zero_()
        k_view = kv_pool.k_buffer[0]
        v_view = kv_pool.v_buffer[0]
        if page_size == 1:
            k_view[loc, 0] = cache_k
            v_view[loc, 0] = cache_v
        else:
            page_id = loc // page_size
            tok_in_p = loc % page_size
            k_view[page_id, tok_in_p] = cache_k
            v_view[page_id, tok_in_p] = cache_v
        k_ref = kv_pool.k_buffer[0].clone()
        v_ref = kv_pool.v_buffer[0].clone()

        self.assertTrue(
            _t.equal(k_kernel, k_ref),
            f"K view mismatch through set_kv_buffer at ps={page_size}",
        )
        self.assertTrue(
            _t.equal(v_kernel, v_ref),
            f"V view mismatch through set_kv_buffer at ps={page_size}",
        )

    def test_integration_ps1(self):
        self._run_set_kv_buffer_and_compare(page_size=1)

    def test_integration_ps64(self):
        self._run_set_kv_buffer_and_compare(page_size=64)


if __name__ == "__main__":
    unittest.main()
