"""Numerical and masking regressions for fused FP8 KV stores and prefix-valid commits."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.memory_pool import (
    _set_kv_buffer_prefix_valid_impl_fp8,
    _try_set_kv_buffer_fp8,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.prefix_valid import (
    assert_bytes_equal,
    assert_prefix_commit,
    make_kv_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: ROCm's E4M3FNUZ pointer inference and conversion differ from CUDA.
register_amd_ci(est_time=25, stage="base-a-jit-kernel-unit", runner_config="amd")

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


# Slot 0 is the reserved CUDA-graph padding slot, so its row must not be written.
STORE_SLOTS = (3, 0, 7, 1, 12)


def _store(k, v, k_cache, v_cache, loc, k_scale, v_scale):
    return _try_set_kv_buffer_fp8(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        indices=loc,
        k_scale=k_scale,
        v_scale=v_scale,
        row_dim=k.shape[-1],
        size_limit=k_cache.shape[0],
    )


def _scale(value):
    return torch.tensor(value, dtype=torch.float32, device=DEVICE)


@contextmanager
def _assert_store(k, v, k_cache, v_cache, loc, k_scale, v_scale):
    # Use the ordinary writer's div_ -> cast oracle, without the prefix-valid
    # oracle's explicit clamp (the ROCm eager cast has different overflow behavior).
    before = (k.clone(), v.clone())
    expected = []
    valid = loc != 0
    for source, cache, scale in zip(before, (k_cache, v_cache), (k_scale, v_scale)):
        value = source.clone()
        if scale is not None:
            value.div_(scale)
        target = cache.contiguous().view(torch.uint8).clone()
        target[loc[valid]] = value.to(cache.dtype).view(torch.uint8)[valid]
        expected.append(target)
    yield
    for label, actual, reference in zip(
        ("K cache", "V cache", "K input", "V input"),
        (k_cache, v_cache, k, v),
        (*expected, *before),
    ):
        assert_bytes_equal(actual, reference, label)


class TestSetKvBufferFp8Kernel(CustomTestCase):
    def setUp(self):
        super().setUp()
        if not memory_pool._triton_fp8_supported():
            self.skipTest("Ordinary fused E4M3 stores require CUDA SM89+")

    def test_matches_eager_bytes_for_every_scale_form(self):
        """Absent, host and device scales, a tile tail and strided rows all
        write the bytes of eager division; slot 0 and the inputs stay as they were."""
        loc = torch.tensor(STORE_SLOTS, device=DEVICE)
        scale_forms = {
            "none": (None, None),
            # Not exact in BF16 or FP16, so the divisor rounding is exercised.
            "host": (0.1, 1.3),
            "device": (_scale(0.1), _scale(1.3)),
            "mixed": (None, _scale(1.3)),
        }
        for dtype in SOURCE_DTYPES:
            for row_dim, strided in ((63, True), (256, False), (1030, False)):
                for form, (k_scale, v_scale) in scale_forms.items():
                    with self.subTest(dtype=dtype, row_dim=row_dim, scales=form):
                        torch.manual_seed(row_dim)
                        shape = (len(STORE_SLOTS), 2, row_dim)
                        source = torch.randn(shape, dtype=dtype, device=DEVICE)
                        if strided:
                            k, v = source[:, :1], source[:, 1:]
                        else:
                            k, v = source[:, :1].clone(), source[:, 1:].clone()
                        k_cache, v_cache = make_kv_cache(16, row_dim)
                        with _assert_store(
                            k,
                            v,
                            k_cache,
                            v_cache,
                            loc,
                            k_scale,
                            v_scale,
                        ):
                            self.assertTrue(
                                _store(k, v, k_cache, v_cache, loc, k_scale, v_scale)
                            )

    def test_saturates_out_of_range_values(self):
        """Magnitudes past the FP8 limit store the limit, NaN stays NaN and
        zero keeps its sign."""
        fp8_max = torch.finfo(fp8_dtype).max
        values = [
            float("-inf"),
            -1000.0,
            -fp8_max,
            -1.0625,
            -0.0,
            0.0,
            1.0625,
            fp8_max,
            1000.0,
            float("inf"),
            float("nan"),
        ]
        loc = torch.tensor([3], device=DEVICE)
        for dtype in SOURCE_DTYPES:
            for scale in (None, 0.5):
                with self.subTest(dtype=dtype, scale=scale):
                    k = torch.tensor(values, dtype=dtype, device=DEVICE).view(1, 1, -1)
                    v = k.flip(-1).contiguous()
                    k_cache, v_cache = make_kv_cache(8, len(values))
                    with _assert_store(k, v, k_cache, v_cache, loc, scale, scale):
                        self.assertTrue(
                            _store(k, v, k_cache, v_cache, loc, scale, scale)
                        )

    def test_all_bf16_fp16_values_follow_eager_cast(self):
        """Do not replace source-rounded division with FP32 reciprocal scaling."""
        loc = torch.tensor([1], device=DEVICE)
        for dtype in (torch.bfloat16, torch.float16):
            values = (
                torch.arange(65536, dtype=torch.int32, device=DEVICE)
                .to(torch.int16)
                .view(dtype)
                .view(1, 1, -1)
            )
            k, v = values, values.flip(-1)
            k_cache, v_cache = make_kv_cache(2, values.numel())
            for scale in (None, 0.1, _scale(0.1)):
                with self.subTest(dtype=dtype, scale=scale):
                    with _assert_store(k, v, k_cache, v_cache, loc, scale, scale):
                        self.assertTrue(
                            _store(k, v, k_cache, v_cache, loc, scale, scale)
                        )

    def test_graph_replay_reads_current_scales_and_locations(self):
        """Captured writes must read live device scales, locations and source rows."""
        k = torch.randn((4, 1, 128), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        k_cache, v_cache = make_kv_cache(16, 128)
        loc = torch.tensor([1, 4, 0, 0], device=DEVICE)
        k_scale, v_scale = _scale(0.1), _scale(1.3)
        _store(k, v, k_cache, v_cache, loc, k_scale, v_scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _store(k, v, k_cache, v_cache, loc, k_scale, v_scale)
        for destinations, ks, vs in (([7, 0, 2, 9], 0.3, 0.7), ([0] * 4, 1.3, 0.1)):
            loc.copy_(torch.tensor(destinations, device=DEVICE))
            k_scale.fill_(ks)
            v_scale.fill_(vs)
            k.normal_()
            v.normal_()
            with _assert_store(k, v, k_cache, v_cache, loc, k_scale, v_scale):
                graph.replay()

    def test_empty_write_preserves_cache(self):
        k = torch.empty((0, 1, 128), dtype=torch.bfloat16, device=DEVICE)
        k_cache, v_cache = make_kv_cache(2, 128)
        loc = torch.empty(0, dtype=torch.int64, device=DEVICE)
        with _assert_store(k, k, k_cache, v_cache, loc, None, None):
            self.assertTrue(_store(k, k, k_cache, v_cache, loc, None, None))

    def test_declines_scales_and_devices_it_does_not_take(self):
        """A per-channel, non-fp32 or off-device scale, a CPU pool and a GPU
        below SM89 report False and write nothing, so the eager path still runs."""
        row_dim = 64
        cases = {
            "per_channel": (DEVICE, torch.full((row_dim,), 0.5, device=DEVICE), True),
            "fp64": (
                DEVICE,
                torch.tensor(0.5, dtype=torch.float64, device=DEVICE),
                True,
            ),
            "off_device": (DEVICE, torch.tensor(0.5), True),
            "cpu_pool": ("cpu", None, True),
            "pre_sm89": (DEVICE, None, False),
        }
        for name, (device, k_scale, supported) in cases.items():
            with (
                self.subTest(case=name),
                patch.object(memory_pool, "_triton_fp8_supported", lambda: supported),
            ):
                k = torch.randn((2, 1, row_dim), dtype=torch.bfloat16, device=device)
                v = torch.randn_like(k)
                k_cache, v_cache = make_kv_cache(8, row_dim, device=device)
                before = (k_cache.clone(), v_cache.clone())
                loc = torch.tensor([2, 5], device=device)
                self.assertFalse(_store(k, v, k_cache, v_cache, loc, k_scale, None))
                assert_bytes_equal(k_cache, before[0], "K cache")
                assert_bytes_equal(v_cache, before[1], "V cache")


if __name__ == "__main__":
    unittest.main()
