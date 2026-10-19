# SPDX-License-Identifier: Apache-2.0
"""Channel-first RMSNorm parity and large-index compilation regressions."""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="The channel RMSNorm Triton kernel requires NVIDIA CUDA",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 32, 3, 7), (2, 32, 1, 3, 7), (1, 1152, 1, 3, 7)])
@pytest.mark.parametrize("zeros", [False, True])
def test_channel_rmsnorm_matches_torch(dtype, shape, zeros):
    from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
        channel_rmsnorm_preserve_reduction,
    )

    torch.manual_seed(1042)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if zeros:
        x.zero_()
    weight = torch.randn(
        (shape[1],) + (1,) * (len(shape) - 2), device="cuda", dtype=dtype
    )
    scale = shape[1] ** 0.5

    actual = channel_rmsnorm_preserve_reduction(x, weight, scale)
    expected = F.normalize(x.float(), dim=1).to(dtype) * scale * weight + 0.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,channels,spatial",
    [
        (1, 1, 2**31 - 1),
        (1, 1, 2**31),
        (1, 2, 2**30),
        (2, 2, 2**29),
        (1, 1152, 1072 * 1920),
        (1, 288, 2144 * 3840),
        # Synthetic factorization of the other observed out-of-range scalar.
        (1, 288, 2160 * 3840),
    ],
)
def test_channel_rmsnorm_large_index_constants(dtype, batch, channels, spatial):
    from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
        _channel_rmsnorm_finish_kernel,
    )

    # Execute only the first tile: these scalar-specialized kernels need not
    # allocate multi-billion-element tensors to reproduce the compilation error.
    # Every load/store in this tile is within the small buffers below. This
    # checks large constants, not memory accesses at offsets above INT32_MAX.
    block = 512
    torch.manual_seed(1042)
    x = torch.randn(block, device="cuda", dtype=dtype)
    norm = torch.linspace(0.75, 1.75, block, device="cuda", dtype=torch.float32)
    weight = torch.linspace(0.5, 1.5, channels, device="cuda", dtype=dtype)
    actual = torch.empty_like(x)
    scale = channels**0.5

    _channel_rmsnorm_finish_kernel[(1,)](
        x,
        norm,
        weight,
        actual,
        batch * channels * spatial,
        channels,
        spatial,
        scale,
        block,
        enable_fp_fusion=False,
    )

    expected = (x.float() / norm).to(dtype) * scale * weight[0] + 0.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_LARGE_CHANNEL_RMSNORM") != "1",
    reason="Opt in to the full large-index allocation with SGLANG_TEST_LARGE_CHANNEL_RMSNORM=1",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 288, 1, 2144, 3840), (2, 32, 1, 33554435)])
def test_channel_rmsnorm_real_high_offsets(dtype, shape, record_property):
    from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
        channel_rmsnorm_preserve_reduction,
    )

    # This explicit hardware regression processes the entire allocation, so a
    # cast after int32 multiplication cannot hide behind a first-tile test.
    free, _ = torch.cuda.mem_get_info()
    if free < 40 * 1024**3:
        pytest.skip("The full-tensor reference needs at least 40 GiB free CUDA memory")
    torch.manual_seed(1042)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.linspace(-1.5, 1.5, shape[1], device="cuda", dtype=dtype).reshape(
        (shape[1],) + (1,) * (len(shape) - 2)
    )
    scale = shape[1] ** 0.5
    assert x.numel() > 2**31 - 1

    actual = channel_rmsnorm_preserve_reduction(x, weight, scale)
    expected = F.normalize(x.float(), dim=1).to(dtype) * scale * weight + 0.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    spatial = x.numel() // (shape[0] * shape[1])
    batch_span = shape[1] * spatial
    offsets = sorted(
        {
            i
            for i in [
                0,
                spatial - 1,
                spatial,
                batch_span - 1,
                batch_span,
                2**31 - 2,
                2**31 - 1,
                2**31,
                2**31 + 1,
                x.numel() - 512,
                x.numel() - 1,
            ]
            if 0 <= i < x.numel()
        }
    )
    indices = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    record_property("numel", x.numel())
    record_property("checked_offsets", offsets)
    record_property("input_samples", x.flatten()[indices].float().tolist())
    record_property("actual_samples", actual.flatten()[indices].float().tolist())
    record_property("expected_samples", expected.flatten()[indices].float().tolist())
    record_property("whole_tensor_byte_equal", True)
