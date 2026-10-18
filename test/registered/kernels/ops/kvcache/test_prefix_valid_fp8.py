"""Numerical and masking regressions for fused prefix-valid FP8 commits."""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl_fp8
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.prefix_valid import assert_prefix_commit, make_kv_cache
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: ROCm's E4M3FNUZ pointer inference and conversion differ from CUDA.
register_amd_ci(est_time=25, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class TestPrefixValidFp8Kernel(CustomTestCase):
    def test_matches_eager_bytes_and_preserves_uncommitted_slots(self):
        """Cover tile branches, noncontiguous inputs, tails and mixed commit lengths."""
        batch_size, block_size = 3, 5
        loc = torch.arange(1, 16, device=DEVICE).view(batch_size, block_size)
        lengths = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)
        for dtype in SOURCE_DTYPES:
            for row_dim, noncontiguous in ((63, True), (2048, False), (4097, False)):
                with self.subTest(dtype=dtype, row_dim=row_dim):
                    torch.manual_seed(1234 + row_dim)
                    if noncontiguous:
                        source = torch.randn(
                            (15, 2, row_dim), dtype=dtype, device=DEVICE
                        )
                        k, v = source[:, :1], source[:, 1:]
                        self.assertFalse(k.is_contiguous())
                        self.assertFalse(v.is_contiguous())
                    else:
                        k = torch.randn((15, 1, row_dim), dtype=dtype, device=DEVICE)
                        v = torch.randn_like(k)
                    k_cache, v_cache = make_kv_cache(20, row_dim)
                    with assert_prefix_commit(
                        k, v, k_cache, v_cache, loc, lengths, 0.375, 1.75
                    ):
                        _set_kv_buffer_prefix_valid_impl_fp8(
                            k, v, k_cache, v_cache, 0.375, 1.75, loc, lengths, row_dim
                        )

    def test_matches_eager_at_fp8_boundaries(self):
        """Representable limits, NaNs and signed zero follow eager FP8 casting."""
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
        loc = torch.tensor([[3]], device=DEVICE)
        lengths = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        for dtype in SOURCE_DTYPES:
            with self.subTest(dtype=dtype):
                k = torch.tensor(values, dtype=dtype, device=DEVICE).view(1, 1, -1)
                v = k.flip(-1).contiguous()
                k_cache, v_cache = make_kv_cache(8, len(values))
                with assert_prefix_commit(
                    k, v, k_cache, v_cache, loc, lengths, 1.0, 1.0
                ):
                    _set_kv_buffer_prefix_valid_impl_fp8(
                        k, v, k_cache, v_cache, 1.0, 1.0, loc, lengths, len(values)
                    )

    def test_scale_division_at_rounding_boundaries(self):
        """Host and GPU scales must retain their distinct eager rounding semantics."""
        loc = torch.tensor([[0]], device=DEVICE)
        lengths = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        scales = (0.1, 0.3, 0.7, 1.3, 0.625, 1.375, 1e-4, 1e4, 2.0**-140)
        for dtype in SOURCE_DTYPES:
            for value in scales:
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
                            torch.nextafter(mid, torch.full_like(mid, -float("inf"))),
                            mid,
                            torch.nextafter(mid, torch.full_like(mid, float("inf"))),
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
                k_cache, v_cache = make_kv_cache(1, k.shape[-1])
                scalar = scale.item()
                for kind, original_scale in (
                    ("gpu", scale),
                    ("python", scalar),
                    ("cpu", scale.cpu()),
                ):
                    with self.subTest(dtype=dtype, scale=value, scale_kind=kind):
                        with assert_prefix_commit(
                            k,
                            v,
                            k_cache,
                            v_cache,
                            loc,
                            lengths,
                            original_scale,
                            original_scale,
                        ):
                            _set_kv_buffer_prefix_valid_impl_fp8(
                                k,
                                v,
                                k_cache,
                                v_cache,
                                scalar,
                                scalar,
                                loc,
                                lengths,
                                k.shape[-1],
                                k_scale_is_tensor=kind == "gpu",
                                v_scale_is_tensor=kind == "gpu",
                            )


if __name__ == "__main__":
    unittest.main()
