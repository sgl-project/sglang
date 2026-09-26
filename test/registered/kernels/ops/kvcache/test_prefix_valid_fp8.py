"""Fused prefix-valid FP8 KV-cache commit correctness tests."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    _resolve_fused_scale,
    _set_kv_buffer_prefix_valid_impl_fp8,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: ROCm uses E4M3FNUZ, so this catches backend-specific FP8
# pointer inference and conversion behavior that CUDA's E4M3FN run cannot.
register_amd_ci(est_time=40, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"


def _valid_rows(loc_2d: torch.Tensor, commit_lens: torch.Tensor):
    row = torch.arange(loc_2d.shape[1], device=loc_2d.device)
    mask = row[None, :] < commit_lens[:, None]
    src = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    dst = loc_2d.reshape(-1).index_select(0, src).long()
    return src, dst


def _eager_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    result = x.clone()
    result.div_(scale)
    return result.to(fp8_dtype)


def _make_pool(
    row_dim: int,
    total_slots: int,
    dtype: torch.dtype = fp8_dtype,
) -> MHATokenToKVPool:
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.dtype = dtype
    pool.store_dtype = torch.uint8 if dtype == fp8_dtype else dtype
    pool.start_layer = 0
    pool.row_dim = row_dim
    pool.v_row_dim = row_dim
    pool.head_dim = row_dim
    pool.v_head_dim = row_dim
    shape = (total_slots, 1, row_dim)
    pool.k_buffer = [torch.full(shape, 0x5A, dtype=pool.store_dtype, device=DEVICE)]
    pool.v_buffer = [torch.full(shape, 0xA5, dtype=pool.store_dtype, device=DEVICE)]
    return pool


class TestPrefixValidFp8Kernel(CustomTestCase):
    def test_matches_eager_bytes_and_preserves_uncommitted_slots(self):
        """A fused register-only rewrite must preserve the eager FP8 bytes.

        The dimensions cover every tile-selection branch and masked tail. Mixed
        commit lengths also catch kernels that copy padded rows or use one batch's
        commit length for another.
        """
        batch_size, block_size, total_slots = 3, 5, 20
        loc_2d = torch.arange(
            1,
            1 + batch_size * block_size,
            dtype=torch.int64,
            device=DEVICE,
        ).view(batch_size, block_size)
        commit_lens = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)
        src_rows, dst_rows = _valid_rows(loc_2d, commit_lens)
        k_scale, v_scale = 0.375, 1.75

        for source_dtype in (torch.bfloat16, torch.float16, torch.float32):
            for row_dim in (63, 2048, 4097):
                with self.subTest(source_dtype=source_dtype, row_dim=row_dim):
                    torch.manual_seed(1234 + row_dim)
                    shape = (batch_size * block_size, 1, row_dim)
                    if row_dim == 63:
                        storage_shape = (batch_size * block_size, 2, row_dim)
                        source = torch.randn(
                            storage_shape, dtype=source_dtype, device=DEVICE
                        )
                        k, v = source[:, :1], source[:, 1:]
                        self.assertFalse(k.is_contiguous())
                        self.assertFalse(v.is_contiguous())
                    else:
                        k = torch.randn(shape, dtype=source_dtype, device=DEVICE)
                        v = torch.randn(shape, dtype=source_dtype, device=DEVICE)

                    cache_shape = (total_slots, 1, row_dim)
                    k_storage = torch.full(
                        cache_shape, 0x5A, dtype=torch.uint8, device=DEVICE
                    )
                    v_storage = torch.full(
                        cache_shape, 0xA5, dtype=torch.uint8, device=DEVICE
                    )
                    expected_k_storage = k_storage.clone()
                    expected_v_storage = v_storage.clone()

                    expected_k_storage[dst_rows] = _eager_quantize(k, k_scale).view(
                        torch.uint8
                    )[src_rows]
                    expected_v_storage[dst_rows] = _eager_quantize(v, v_scale).view(
                        torch.uint8
                    )[src_rows]

                    _set_kv_buffer_prefix_valid_impl_fp8(
                        k,
                        v,
                        k_storage.view(fp8_dtype),
                        v_storage.view(fp8_dtype),
                        k_scale,
                        v_scale,
                        loc_2d,
                        commit_lens,
                        row_dim,
                    )

                    self.assertTrue(torch.equal(k_storage, expected_k_storage))
                    self.assertTrue(torch.equal(v_storage, expected_v_storage))

    def test_matches_eager_at_fp8_boundaries(self):
        """Representable limits, NaNs, and signed zero follow eager FP8 casting."""
        fp8_max = torch.finfo(fp8_dtype).max
        values = [
            -fp8_max,
            -fp8_max + 1.0,
            -0.75 * fp8_max,
            -1.0625,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.0625,
            0.75 * fp8_max,
            fp8_max - 1.0,
            fp8_max,
            float("nan"),
        ]
        row_dim = len(values)
        loc_2d = torch.tensor([[3]], dtype=torch.int64, device=DEVICE)
        commit_lens = torch.tensor([1], dtype=torch.int32, device=DEVICE)

        for source_dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(source_dtype=source_dtype):
                k = torch.tensor(values, dtype=source_dtype, device=DEVICE).view(
                    1, 1, row_dim
                )
                v = k.flip(-1).contiguous()
                k_storage = torch.full(
                    (8, 1, row_dim), 0x5A, dtype=torch.uint8, device=DEVICE
                )
                v_storage = torch.full(
                    (8, 1, row_dim), 0xA5, dtype=torch.uint8, device=DEVICE
                )
                expected_k = k_storage.clone()
                expected_v = v_storage.clone()
                expected_k[3] = _eager_quantize(k, 1.0).view(torch.uint8)[0]
                expected_v[3] = _eager_quantize(v, 1.0).view(torch.uint8)[0]

                _set_kv_buffer_prefix_valid_impl_fp8(
                    k,
                    v,
                    k_storage.view(fp8_dtype),
                    v_storage.view(fp8_dtype),
                    1.0,
                    1.0,
                    loc_2d,
                    commit_lens,
                    row_dim,
                )

                self.assertTrue(torch.equal(k_storage, expected_k))
                self.assertTrue(torch.equal(v_storage, expected_v))

    def test_pool_dispatch_uses_gpu_parameter_float_shadows(self):
        """GPU scale Parameters must fuse without synchronizing through item()."""
        row_dim, total_slots = 65, 12
        pool = _make_pool(row_dim, total_slots)
        k_scale, v_scale = 0.625, 1.375
        layer_k_scale = torch.nn.Parameter(
            torch.tensor(k_scale, dtype=torch.float32, device=DEVICE),
            requires_grad=False,
        )
        layer_v_scale = torch.nn.Parameter(
            torch.tensor(v_scale, dtype=torch.float32, device=DEVICE),
            requires_grad=False,
        )
        layer = SimpleNamespace(
            layer_id=0,
            k_scale=layer_k_scale,
            v_scale=layer_v_scale,
            k_scale_float=k_scale,
            v_scale_float=v_scale,
        )
        loc_2d = torch.tensor([[2, 7, 9]], dtype=torch.int32, device=DEVICE)
        commit_lens = torch.tensor([2], dtype=torch.int64, device=DEVICE)
        k = torch.randn((3, 1, row_dim), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        k_before, v_before = k.clone(), v.clone()
        expected_k = pool.k_buffer[0].clone()
        expected_v = pool.v_buffer[0].clone()
        expected_k[[2, 7]] = _eager_quantize(k, k_scale).view(torch.uint8)[:2]
        expected_v[[2, 7]] = _eager_quantize(v, v_scale).view(torch.uint8)[:2]

        pool.set_kv_buffer_prefix_valid(
            layer,
            loc_2d,
            commit_lens,
            k,
            v,
            layer_k_scale,
            layer_v_scale,
        )

        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(v, v_before))

    def test_gpu_tensor_without_float_shadow_preserves_eager_fallback(self):
        """An unresolved device scale must keep the tensor-aware eager path."""
        row_dim, total_slots = 65, 12
        pool = _make_pool(row_dim, total_slots)
        k_scale = torch.tensor(0.625, dtype=torch.float32, device=DEVICE)
        v_scale = torch.tensor(1.375, dtype=torch.float32, device=DEVICE)
        layer = SimpleNamespace(
            layer_id=0,
            k_scale=k_scale,
            v_scale=v_scale,
            k_scale_float=None,
            v_scale_float=None,
        )
        loc_2d = torch.tensor([[2, 7, 9]], dtype=torch.int64, device=DEVICE)
        commit_lens = torch.tensor([2], dtype=torch.int32, device=DEVICE)
        k = torch.randn((3, 1, row_dim), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        expected_scaled_k = k.clone()
        expected_scaled_v = v.clone()
        expected_scaled_k.div_(k_scale)
        expected_scaled_v.div_(v_scale)
        expected_k = pool.k_buffer[0].clone()
        expected_v = pool.v_buffer[0].clone()
        expected_k[[2, 7]] = expected_scaled_k.to(fp8_dtype).view(torch.uint8)[:2]
        expected_v[[2, 7]] = expected_scaled_v.to(fp8_dtype).view(torch.uint8)[:2]

        pool.set_kv_buffer_prefix_valid(
            layer,
            loc_2d,
            commit_lens,
            k,
            v,
            k_scale,
            v_scale,
        )

        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
        self.assertTrue(torch.equal(k, expected_scaled_k))
        self.assertTrue(torch.equal(v, expected_scaled_v))

    def test_missing_scale_preserves_eager_fallback(self):
        """A missing scale must not partially enter the scalar-only fused path."""
        row_dim, total_slots = 65, 12
        pool = _make_pool(row_dim, total_slots)
        v_scale = 1.375
        layer = SimpleNamespace(
            layer_id=0,
            k_scale=None,
            v_scale=v_scale,
            k_scale_float=None,
            v_scale_float=v_scale,
        )
        loc_2d = torch.tensor([[2, 7, 9]], dtype=torch.int64, device=DEVICE)
        commit_lens = torch.tensor([2], dtype=torch.int32, device=DEVICE)
        k = torch.randn((3, 1, row_dim), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        k_before = k.clone()
        expected_scaled_v = v.clone()
        expected_scaled_v.div_(v_scale)
        expected_k = pool.k_buffer[0].clone()
        expected_v = pool.v_buffer[0].clone()
        expected_k[[2, 7]] = k.to(fp8_dtype).view(torch.uint8)[:2]
        expected_v[[2, 7]] = expected_scaled_v.to(fp8_dtype).view(torch.uint8)[:2]

        pool.set_kv_buffer_prefix_valid(
            layer,
            loc_2d,
            commit_lens,
            k,
            v,
            None,
            v_scale,
        )

        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(v, expected_scaled_v))

    def test_non_fp8_cache_preserves_original_copy_path(self):
        """A non-FP8 cache must bypass quantization and retain prefix masking."""
        row_dim, total_slots = 65, 12
        pool = _make_pool(row_dim, total_slots, dtype=torch.bfloat16)
        layer = SimpleNamespace(layer_id=0)
        loc_2d = torch.tensor([[2, 7, 9]], dtype=torch.int64, device=DEVICE)
        commit_lens = torch.tensor([2], dtype=torch.int32, device=DEVICE)
        k = torch.randn((3, 1, row_dim), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        expected_k = pool.k_buffer[0].clone()
        expected_v = pool.v_buffer[0].clone()
        expected_k[[2, 7]] = k[:2]
        expected_v[[2, 7]] = v[:2]

        pool.set_kv_buffer_prefix_valid(layer, loc_2d, commit_lens, k, v)

        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))


class TestResolveFusedScale(CustomTestCase):
    def test_supported_and_fallback_scale_forms(self):
        """Only synchronization-free scalar forms may enter the fused kernel."""
        layer_scale = torch.tensor(0.5, dtype=torch.float32, device=DEVICE)
        unrelated_scale = torch.tensor(0.75, dtype=torch.float32, device=DEVICE)

        cases = (
            (0.25, None, None, 0.25),
            (2, None, None, 2.0),
            (torch.tensor(0.5), None, None, 0.5),
            (layer_scale, layer_scale, 0.5, 0.5),
            (unrelated_scale, layer_scale, 0.5, None),
            (None, None, 0.5, None),
        )
        for scale, owner, shadow, expected in cases:
            with self.subTest(
                scale_type=type(scale).__name__,
                has_owner=owner is not None,
                shadow=shadow,
            ):
                self.assertEqual(_resolve_fused_scale(scale, owner, shadow), expected)


if __name__ == "__main__":
    unittest.main()
