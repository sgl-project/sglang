"""Fused prefix-valid FP8 KV-cache commit correctness tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
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


def _eager_quantize(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
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
    def test_dflash_constructor_initializes_fp8_prefix_scales(self):
        """A real FP8 draft must initialize scales and reach the fused writer."""
        from transformers import LlamaConfig

        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8KVCacheMethod
        from sglang.srt.models.dflash import DFlashAttention
        from sglang.srt.runtime_context import get_context, get_parallel

        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=128,
        )
        with (
            get_context().override_server_args(tp_size=1),
            get_parallel().override(tp_rank=0),
        ):
            with torch.device(DEVICE):
                plain = DFlashAttention(config, layer_id=0).attn
            self.assertIsNone(plain.quant_method)
            self.assertIsNone(plain.k_scale)
            self.assertIsNone(plain.v_scale)

            for loaded_scales in (None, (0.625, 1.375)):
                with self.subTest(loaded_scales=loaded_scales):
                    with torch.device(DEVICE):
                        layer = DFlashAttention(
                            config, layer_id=0, quant_config=Fp8Config()
                        ).attn
                    self.assertIsInstance(layer.quant_method, Fp8KVCacheMethod)
                    for scale in (layer.k_scale, layer.v_scale):
                        self.assertIsInstance(scale, torch.nn.Parameter)
                        self.assertTrue(scale.is_cuda)
                        self.assertEqual(scale.dtype, torch.float32)
                        self.assertEqual(scale.ndim, 0)
                        self.assertEqual(scale.item(), -1.0)
                    if loaded_scales is not None:
                        with torch.no_grad():
                            layer.k_scale.fill_(loaded_scales[0])
                            layer.v_scale.fill_(loaded_scales[1])
                    layer.quant_method.process_weights_after_loading(layer)
                    factor = 2 if loaded_scales and is_fp8_fnuz() else 1
                    expected_scales = tuple(
                        value * factor for value in (loaded_scales or (1.0, 1.0))
                    )
                    self.assertEqual(
                        (layer.k_scale_float, layer.v_scale_float), expected_scales
                    )
                    self.assertEqual(
                        (layer.k_scale.item(), layer.v_scale.item()), expected_scales
                    )

                    pool = _make_pool(64, 8)
                    loc = torch.tensor([[2, 5, -1]], device=DEVICE)
                    lengths = torch.tensor([2], dtype=torch.int32, device=DEVICE)
                    k = torch.linspace(-2, 2, 192, device=DEVICE).to(torch.bfloat16)
                    k = k.view(3, 1, 64)
                    v = k.flip(-1)
                    before_k, before_v = k.clone(), v.clone()
                    expected_k = pool.k_buffer[0].clone()
                    expected_v = pool.v_buffer[0].clone()
                    expected_k[[2, 5]] = _eager_quantize(k, layer.k_scale).view(
                        torch.uint8
                    )[:2]
                    expected_v[[2, 5]] = _eager_quantize(v, layer.v_scale).view(
                        torch.uint8
                    )[:2]
                    # The fused implementation and GPU stores run unmocked.
                    with patch(
                        "sglang.srt.mem_cache.memory_pool._set_kv_buffer_prefix_valid_impl",
                        side_effect=AssertionError("FP8 draft took the eager fallback"),
                    ):
                        pool.set_kv_buffer_prefix_valid(
                            layer, loc, lengths, k, v, layer.k_scale, layer.v_scale
                        )
                    self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
                    self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
                    self.assertTrue(torch.equal(k, before_k))
                    self.assertTrue(torch.equal(v, before_v))

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
        expected_k[[2, 7]] = _eager_quantize(k, layer_k_scale).view(torch.uint8)[:2]
        expected_v[[2, 7]] = _eager_quantize(v, layer_v_scale).view(torch.uint8)[:2]

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

    def test_scale_kind_preserves_eager_division(self):
        """GPU scalar divisors round differently from host scalars before division.

        Compare against the original scale operands, including the BF16 0.1
        counterexample. Mixed scale kinds must retain independent K/V semantics.
        """
        row_dim = 65
        loc = torch.tensor([[2, -1, -1], [5, 1, -1]], device=DEVICE)
        lengths = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
        src, dst = _valid_rows(loc, lengths)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for k_kind, v_kind in (
                ("gpu", "gpu"),
                ("gpu", "python"),
                ("python", "gpu"),
                ("cpu", "gpu"),
                ("gpu", "cpu"),
            ):
                with self.subTest(dtype=dtype, k_kind=k_kind, v_kind=v_kind):
                    pool = _make_pool(row_dim, 8)
                    k_scale = torch.nn.Parameter(
                        torch.tensor(0.1, dtype=torch.float32, device=DEVICE),
                        requires_grad=False,
                    )
                    v_scale = torch.nn.Parameter(
                        torch.tensor(0.3, dtype=torch.float32, device=DEVICE),
                        requires_grad=False,
                    )
                    layer = SimpleNamespace(
                        layer_id=0,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        k_scale_float=k_scale.item(),
                        v_scale_float=v_scale.item(),
                    )
                    k_arg = {
                        "gpu": k_scale,
                        "python": layer.k_scale_float,
                        "cpu": k_scale.cpu(),
                    }[k_kind]
                    v_arg = {
                        "gpu": v_scale,
                        "python": layer.v_scale_float,
                        "cpu": v_scale.cpu(),
                    }[v_kind]
                    k = (
                        torch.linspace(-2, 2, 6 * row_dim, device=DEVICE)
                        .to(dtype)
                        .view(6, 1, row_dim)
                    )
                    k[:, :, 0] = 0.0966796875
                    v = k.flip(-1)
                    before_k, before_v = k.clone(), v.clone()
                    expected_k = pool.k_buffer[0].clone()
                    expected_v = pool.v_buffer[0].clone()
                    expected_k[dst] = _eager_quantize(k, k_arg).view(torch.uint8)[src]
                    expected_v[dst] = _eager_quantize(v, v_arg).view(torch.uint8)[src]
                    pool.set_kv_buffer_prefix_valid(
                        layer, loc, lengths, k, v, k_arg, v_arg
                    )
                    self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
                    self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
                    self.assertTrue(torch.equal(k, before_k))
                    self.assertTrue(torch.equal(v, before_v))

    def test_host_scale_rounding_through_pool(self):
        """Host-scale reciprocal rounding must not move FP32 values onto FP8 ties."""
        loc = torch.tensor([[3, -1]], device=DEVICE)
        lengths = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        for scale in (10000.0, torch.tensor(10000.0)):
            with self.subTest(scale_type=type(scale).__name__):
                pool = _make_pool(65, 8)
                layer = SimpleNamespace(layer_id=0)
                k = torch.full(
                    (2, 1, 65), 29.296873092651367, dtype=torch.float32, device=DEVICE
                )
                v = -k
                before_k, before_v = k.clone(), v.clone()
                expected_k = pool.k_buffer[0].clone()
                expected_v = pool.v_buffer[0].clone()
                expected_k[3] = _eager_quantize(k, scale).view(torch.uint8)[0]
                expected_v[3] = _eager_quantize(v, scale).view(torch.uint8)[0]
                pool.set_kv_buffer_prefix_valid(layer, loc, lengths, k, v, scale, scale)
                self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
                self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))
                self.assertTrue(torch.equal(k, before_k))
                self.assertTrue(torch.equal(v, before_v))

    def test_scale_division_at_rounding_boundaries(self):
        """Host and GPU scales must retain their own eager FP8 rounding semantics."""
        loc = torch.tensor([[0]], dtype=torch.int64, device=DEVICE)
        lengths = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for value in (0.1, 0.3, 0.7, 1.3, 0.625, 1.375, 1e-4, 1e4, 2.0**-140):
                with self.subTest(dtype=dtype, scale=value):
                    scale = torch.nn.Parameter(
                        torch.tensor(value, dtype=torch.float32, device=DEVICE),
                        requires_grad=False,
                    )
                    if dtype == torch.float32:
                        fp8 = (
                            torch.arange(127, dtype=torch.uint8, device=DEVICE)
                            .view(fp8_dtype)
                            .float()
                        )
                        mid = (fp8[:-1] + fp8[1:]) * 0.5 * scale
                        values = torch.stack(
                            (
                                torch.nextafter(
                                    mid, torch.full_like(mid, -float("inf"))
                                ),
                                mid,
                                torch.nextafter(
                                    mid, torch.full_like(mid, float("inf"))
                                ),
                            )
                        ).flatten()
                        values = torch.cat((values, -values))
                    else:
                        # Every BF16/FP16 bit pattern, including subnormals and NaNs.
                        values = (
                            torch.arange(65536, dtype=torch.int32, device=DEVICE)
                            .to(torch.int16)
                            .view(dtype)
                        )
                    k = values.view(1, 1, -1)
                    v = k.flip(-1)
                    k_cache = torch.empty_like(k, dtype=fp8_dtype)
                    v_cache = torch.empty_like(v, dtype=fp8_dtype)
                    for kind, original_scale in (
                        ("gpu", scale),
                        ("python", scale.item()),
                        ("cpu", scale.cpu()),
                    ):
                        with self.subTest(scale_kind=kind):
                            _set_kv_buffer_prefix_valid_impl_fp8(
                                k,
                                v,
                                k_cache,
                                v_cache,
                                scale.item(),
                                scale.item(),
                                loc,
                                lengths,
                                k.shape[-1],
                                k_scale_is_tensor=kind == "gpu",
                                v_scale_is_tensor=kind == "gpu",
                            )
                            self.assertTrue(
                                torch.equal(
                                    k_cache.view(torch.uint8),
                                    _eager_quantize(k, original_scale).view(
                                        torch.uint8
                                    ),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    v_cache.view(torch.uint8),
                                    _eager_quantize(v, original_scale).view(
                                        torch.uint8
                                    ),
                                )
                            )

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
        vector_scale = layer_scale.view(1)

        cases = (
            (0.25, None, None, 0.25),
            (2, None, None, 2.0),
            (torch.tensor(0.5), None, None, 0.5),
            (layer_scale, layer_scale, 0.5, 0.5),
            (unrelated_scale, layer_scale, 0.5, None),
            (None, None, 0.5, None),
            (vector_scale, vector_scale, 0.5, None),
            (torch.tensor([0.5]), None, None, None),
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
