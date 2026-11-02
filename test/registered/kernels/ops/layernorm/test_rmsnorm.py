"""The rmsnorm family JIT kernel: plain, fused residual-add, fp8 per-tensor quant, HF rounding."""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.layernorm import norm as jit_norm
from sglang.kernels.ops.layernorm.norm import RMSNormMode, precompile_rmsnorm
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=1500, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=600, suite="jit-kernel-unit-test-amd")

EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.bfloat16, torch.float16]
BS_LIST = [1, 2, 3, 7, 16, 33, 128, 256, 257, 1000, 4096, 8193]
HIDDEN_SIZE_LIST = [
    32,
    64,
    96,
    128,
    256,
    288,
    384,
    512,
    1024,
    1536,
    2048,
    2304,
    2560,
    2880,
    3072,
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
    12288,
    16384,
]
CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
    [
        (1, 512),
        (7, 2880),
        (33, 4096),
        (257, 7168),
        (1000, 3584),
        (4096, 1536),
        (3, 16384),
        (8193, 1024),
        (5, 128),
        (1000, 64),
        (33, 288),
    ],
)


@pytest.fixture(scope="module", autouse=True)
def _prebuild_modules() -> None:
    """Build every JIT signature the cases reach at once; one at a time they take minutes."""
    from functools import partial

    from sglang.kernels.jit.utils import parallel_compile

    signatures = set()
    for _, hidden_size in CASES:
        for dtype in DTYPES:
            for mode in (RMSNormMode.PLAIN, RMSNormMode.FUSED_ADD):
                signatures.add((hidden_size, dtype, mode, False))
                signatures.add((hidden_size, dtype, mode, True))
            if not is_hip() and dtype == torch.bfloat16:
                signatures.add((hidden_size, dtype, RMSNormMode.QUANT, False))
                signatures.add((hidden_size, dtype, RMSNormMode.FUSED_ADD_QUANT, False))
    parallel_compile(
        partial(precompile_rmsnorm, *sig) for sig in sorted(signatures, key=str)
    )


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    scale: float | None,
    hf: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The spec: a fused residual is rounded once to the activation dtype and the rounded value
    is normalized; HF rounds x * r before the weight multiply; fp8 saturates at +-448."""
    s = (x.float() + residual.float()).to(x.dtype) if residual is not None else x
    sf = s.float()
    r = torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + EPS)
    if hf:
        y = (sf * r).to(x.dtype).float() * weight.float()
    else:
        y = sf * r * weight.float()
    if scale is not None:
        y = (y / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    else:
        y = y.to(x.dtype)
    return y, (s if residual is not None else None)


def _assert_close_16(out: torch.Tensor, ref: torch.Tensor, max_ulp: int = 1) -> None:
    # accumulation order and the fused product-then-scale epilogue move at most one 16-bit ulp;
    # the HF form rounds twice, so a 1-ulp flip of x * r can become 2 ulp after the weight
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    diff = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    diff = torch.where((out == 0) & (ref == 0), torch.zeros_like(diff), diff)  # +-0
    assert diff.max().item() <= max_ulp, f"max ulp distance {diff.max().item()}"


def _inputs(batch_size: int, hidden_size: int, dtype: torch.dtype, residual: bool):
    g = torch.Generator(device=DEVICE).manual_seed(batch_size * 131 + hidden_size)
    x = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype, generator=g)
    w = 1 + 0.5 * torch.randn(hidden_size, device=DEVICE, dtype=dtype, generator=g)
    res = (
        torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype, generator=g)
        if residual
        else None
    )
    return x, w, res


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [False, True])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, in_place: bool
) -> None:
    x, w, _ = _inputs(batch_size, hidden_size, dtype, residual=False)
    ref, _ = _reference(x, w, None, None, hf=False)
    if in_place:
        out = x.clone()
        jit_norm.rmsnorm(out, w, out, EPS)
    else:
        out = torch.empty_like(x)
        jit_norm.rmsnorm(x, w, out, EPS)
    _assert_close_16(out, ref)


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> None:
    x, w, res = _inputs(batch_size, hidden_size, dtype, residual=True)
    ref, ref_res = _reference(x, w, res, None, hf=False)
    jit_norm.fused_add_rmsnorm(x, res, w, EPS)
    assert torch.equal(res, ref_res), "the rounded residual has no order freedom"
    _assert_close_16(x, ref)


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_hf(batch_size: int, hidden_size: int, dtype: torch.dtype) -> None:
    x, w, res = _inputs(batch_size, hidden_size, dtype, residual=True)
    ref, _ = _reference(x, w, None, None, hf=True)
    out = torch.empty_like(x)
    jit_norm.rmsnorm(x, w, out, EPS, cast_x_before_out_mul=True)
    _assert_close_16(out, ref, max_ulp=2)
    ref_fused, ref_res = _reference(x, w, res, None, hf=True)
    jit_norm.fused_add_rmsnorm(x, res, w, EPS, cast_x_before_out_mul=True)
    assert torch.equal(res, ref_res)
    _assert_close_16(x, ref_fused, max_ulp=2)


@pytest.mark.skipif(is_hip(), reason="fp8 e4m3 quant modes are CUDA-only")
@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("fused", [False, True])
# [1] and 0-D (``input_scale.max()``) are both produced by static FP8 loaders.
@pytest.mark.parametrize("scale_shape", [(1,), ()])
def test_rmsnorm_quant(
    batch_size: int, hidden_size: int, fused: bool, scale_shape: tuple
) -> None:
    dtype = torch.bfloat16
    x, w, res = _inputs(batch_size, hidden_size, dtype, residual=fused)
    scale = torch.full(scale_shape, 0.05, device=DEVICE, dtype=torch.float32)
    ref, ref_res = _reference(x, w, res, scale.item(), hf=False)
    out = torch.empty(batch_size, hidden_size, device=DEVICE, dtype=torch.float8_e4m3fn)
    if fused:
        jit_norm.fused_add_rmsnorm_quant(x, res, w, scale, out, EPS)
        assert torch.equal(res, ref_res)
    else:
        jit_norm.rmsnorm_quant(x, w, scale, out, EPS)
    # rounding-boundary flips are one fp8 ulp at the element's magnitude
    mismatch = (out.view(torch.uint8) != ref.view(torch.uint8)).float().mean().item()
    assert mismatch < 2e-3, f"fp8 bit mismatch fraction {mismatch}"
    torch.testing.assert_close(
        out.float(), ref.float(), atol=0.0, rtol=0.13
    )  # <= 1 ulp of e4m3 (2^-3)


def test_rmsnorm_strided_rows() -> None:
    x_full = torch.randn(64, 4096 + 64, device=DEVICE, dtype=torch.bfloat16)
    x = x_full[:, :4096]
    w = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)
    ref, _ = _reference(x.contiguous(), w, None, None, hf=False)
    out = torch.empty_like(x_full)[:, :4096]
    jit_norm.rmsnorm(x, w, out, EPS)
    _assert_close_16(out, ref)
    assert torch.equal(x_full[:, 4096:], x_full[:, 4096:])  # untouched padding


def test_rmsnorm_empty() -> None:
    x = torch.empty(0, 4096, device=DEVICE, dtype=torch.bfloat16)
    w = torch.ones(4096, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    jit_norm.rmsnorm(x, w, out, EPS)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (16, False),
        (32, True),
        (64, True),
        (96, True),
        (256, True),
        (500, False),
        (512, True),
        (576, True),
        (2880, True),
        (3584, True),
        (8192, True),
        (8256, True),
        (12288, True),
        (16384, True),
        (16400, False),
    ],
)
def test_rmsnorm_hidden_size_support(hidden_size: int, expected: bool) -> None:
    assert jit_norm.is_supported_rmsnorm_hidden_size(hidden_size) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
