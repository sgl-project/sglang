"""Numerical and slot regressions for the dense NHD FP8 KV store."""

import itertools
import unittest

import torch

from sglang.kernels.ops.kvcache.cache_move import store_cache_fp8
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, saturate_to_fp8_range
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.prefix_valid import (
    assert_bytes_equal,
    assert_prefix_commit,
    make_kv_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _scales(kind):
    if kind == "none":
        return None, None
    if kind == "host":
        return 0.375, 1.75
    return tuple(
        torch.tensor(value, dtype=torch.float32, device=DEVICE)
        for value in (0.375, 1.75)
    )


def _store_matches_eager(k, v, k_cache, v_cache, loc, k_scale, v_scale):
    # An ordinary store is a prefix commit of one block whose length covers loc.
    lengths = torch.tensor([loc.numel()], dtype=torch.int32, device=DEVICE)
    with assert_prefix_commit(
        k, v, k_cache, v_cache, loc.view(1, -1), lengths, k_scale, v_scale
    ):
        store_cache_fp8(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            indices=loc,
            k_scale=k_scale,
            v_scale=v_scale,
        )


class TestStoreCacheFp8(CustomTestCase):
    def test_matches_eager_bytes_and_preserves_other_slots(self):
        """Cover block tails, padded token strides, index dtypes and scale kinds."""
        num_tokens, total_slots = 9, 32
        cases = itertools.product(
            SOURCE_DTYPES,
            ((63, True), (2048, False), (4097, False)),
            (torch.int32, torch.int64),
            ("none", "host", "device"),
        )
        for dtype, (row_dim, padded), index_dtype, kind in cases:
            with self.subTest(
                dtype=dtype, row_dim=row_dim, index_dtype=index_dtype, scale=kind
            ):
                torch.manual_seed(1234 + row_dim)
                if padded:
                    source = torch.randn(
                        (num_tokens, 2, row_dim), dtype=dtype, device=DEVICE
                    )
                    k, v = source[:, :1], source[:, 1:]
                    self.assertFalse(k.is_contiguous())
                else:
                    k = torch.randn(
                        (num_tokens, 1, row_dim), dtype=dtype, device=DEVICE
                    )
                    v = torch.randn_like(k)
                slots = torch.randperm(total_slots - 1, device=DEVICE)[:num_tokens]
                loc = (slots + 1).to(index_dtype)
                k_cache, v_cache = make_kv_cache(total_slots, row_dim)
                _store_matches_eager(k, v, k_cache, v_cache, loc, *_scales(kind))

    def test_asymmetric_widths(self):
        """K and V rows of different widths must use their own strides and masks."""
        for k_width, v_width in ((192, 128), (128, 192), (65, 513)):
            with self.subTest(k_width=k_width, v_width=v_width):
                torch.manual_seed(k_width * 1000 + v_width)
                k = torch.randn((6, 1, k_width), dtype=torch.bfloat16, device=DEVICE)
                v = torch.randn((6, 1, v_width), dtype=torch.bfloat16, device=DEVICE)
                k_cache = make_kv_cache(16, k_width)[0]
                v_cache = make_kv_cache(16, v_width)[1]
                loc = torch.tensor([3, 1, 15, 7, 2, 9], device=DEVICE)
                _store_matches_eager(k, v, k_cache, v_cache, loc, *_scales("device"))

    def test_matches_eager_at_fp8_boundaries(self):
        """Representable limits, overflow, NaNs and signed zero follow eager casting."""
        fp8_max = torch.finfo(fp8_dtype).max
        values = [
            -2 * fp8_max,
            -fp8_max,
            -fp8_max + 1.0,
            -1.0625,
            -0.0,
            0.0,
            1.0625,
            fp8_max - 1.0,
            fp8_max,
            2 * fp8_max,
            float("nan"),
        ]
        loc = torch.tensor([3], device=DEVICE)
        for dtype in SOURCE_DTYPES:
            for scale in (1.0, 0.5):
                with self.subTest(dtype=dtype, scale=scale):
                    k = torch.tensor(values, dtype=dtype, device=DEVICE).view(1, 1, -1)
                    v = k.flip(-1).contiguous()
                    k_cache, v_cache = make_kv_cache(8, len(values))
                    _store_matches_eager(k, v, k_cache, v_cache, loc, scale, scale)

    def test_scale_division_at_rounding_boundaries(self):
        """Host and device scales keep their distinct eager rounding semantics."""
        loc = torch.tensor([1], device=DEVICE)
        for dtype in (torch.bfloat16, torch.float16):
            # Every bit pattern, including subnormals and NaNs.
            k = (
                torch.arange(65536, dtype=torch.int32, device=DEVICE)
                .to(torch.int16)
                .view(dtype)
                .view(1, 1, -1)
            )
            v = k.flip(-1).contiguous()
            for value in (0.1, 0.3, 1.3, 1e-4, 1e4):
                device_scale = torch.tensor(value, dtype=torch.float32, device=DEVICE)
                for kind, scale in (("host", value), ("device", device_scale)):
                    with self.subTest(dtype=dtype, scale=value, scale_kind=kind):
                        k_cache, v_cache = make_kv_cache(2, k.shape[-1])
                        _store_matches_eager(k, v, k_cache, v_cache, loc, scale, scale)

    def test_reserved_slot_zero_is_skipped(self):
        """CUDA-graph padding rows target slot zero and must not be written."""
        k = torch.randn((4, 1, 128), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        # Padding rows may hold garbage; NaN makes an accidental write visible.
        k[[0, 2]] = float("nan")
        v[[0, 2]] = float("nan")
        k_cache, v_cache = make_kv_cache(8, 128)
        expected = [cache.view(torch.uint8).clone() for cache in (k_cache, v_cache)]
        loc = torch.tensor([0, 5, 0, 2], device=DEVICE)
        for value, reference in zip((k, v), expected):
            quantized = saturate_to_fp8_range(value / 0.5, fp8_dtype).to(fp8_dtype)
            reference[loc[[1, 3]]] = quantized.view(torch.uint8)[[1, 3]]

        store_cache_fp8(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            indices=loc,
            k_scale=0.5,
            v_scale=0.5,
        )

        assert_bytes_equal(k_cache, expected[0], "K cache")
        assert_bytes_equal(v_cache, expected[1], "V cache")

    def test_empty_indices_are_a_no_op(self):
        k = torch.empty((0, 1, 64), dtype=torch.bfloat16, device=DEVICE)
        k_cache, v_cache = make_kv_cache(4, 64)
        before = [cache.clone() for cache in (k_cache, v_cache)]

        store_cache_fp8(
            k=k,
            v=k,
            k_cache=k_cache,
            v_cache=v_cache,
            indices=torch.empty(0, dtype=torch.int64, device=DEVICE),
            k_scale=None,
            v_scale=None,
        )

        assert_bytes_equal(k_cache, before[0], "K cache")
        assert_bytes_equal(v_cache, before[1], "V cache")


if __name__ == "__main__":
    unittest.main()
