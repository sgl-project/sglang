import pytest
import torch

from sglang.jit_kernel.cast import downcast_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DTYPES = [torch.bfloat16, torch.float16]

# FP8 E4M3 representable range (matches kFP8E4M3Max in type.cuh)
_FP8_E4M3_MAX = 448.0


def _run(input_sl, head, dim, out_sl, dtype):
    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")
    downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)
    return k_out, v_out


def _ref_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference: replicate kernel precision — scale_inv in dtype T, then to fp8.

    Mirrors the kernel logic:
      scale_inv = cast<T>(1.0f) / cast<T>(scale[0])
      out[j]    = cast<fp8_e4m3_t>(clamp(x[j] * scale_inv))
    """
    dtype = x.dtype
    scale_inv = x.new_ones(1) / scale[0].to(dtype)
    x_scaled = (x * scale_inv).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return x_scaled.to(torch.float8_e4m3fn).view(torch.uint8)


def _ref_downcast(
    x: torch.Tensor,
    scale: torch.Tensor,
    loc: torch.Tensor,
    out_sl: int,
    mult: int = 1,
    offset: int = 0,
) -> torch.Tensor:
    """Scatter _ref_fp8 output to the correct output slots via loc/mult/offset."""
    head, dim = x.shape[1], x.shape[2]
    out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device=x.device)
    fp8 = _ref_fp8(x, scale)
    for i, dst in enumerate(loc.tolist()):
        out[dst * mult + offset] = fp8[i]
    return out


# ---------------------------------------------------------------------------
# Existing sanity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_sl,head,dim,out_sl", [(4, 8, 128, 16)])
def test_downcast_fp8(input_sl, head, dim, out_sl, dtype):
    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)

    # Verify written slots are non-zero (fp8 of random non-zero values)
    assert k_out[:input_sl].any(), "k_out should have non-zero fp8 values"
    assert v_out[:input_sl].any(), "v_out should have non-zero fp8 values"
    # Verify unwritten slots remain zero
    assert not k_out[input_sl:].any(), "k_out slots beyond input_sl should be zero"
    assert not v_out[input_sl:].any(), "v_out slots beyond input_sl should be zero"


# ---------------------------------------------------------------------------
# Numerical correctness: kernel output must match PyTorch fp8 reference.
# This verifies that cast<T>(float) and cast<fp8_e4m3_t>(T) produce the
# same bit patterns as the removed ConvertFromFloat / ConvertToFP8 structs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_sl,head,dim,out_sl", [(4, 8, 128, 16), (1, 4, 64, 8)])
def test_downcast_fp8_matches_reference(input_sl, head, dim, out_sl, dtype):
    torch.manual_seed(42)
    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)

    k_ref = _ref_downcast(k, k_scale, loc, out_sl)
    v_ref = _ref_downcast(v, v_scale, loc, out_sl)

    torch.testing.assert_close(k_out, k_ref, msg="k: kernel vs reference mismatch")
    torch.testing.assert_close(v_out, v_ref, msg="v: kernel vs reference mismatch")


# ---------------------------------------------------------------------------
# Scale: a non-unit scale divides the values before fp8 conversion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_val", [0.5, 2.0, 0.1])
def test_downcast_fp8_scale(scale_val, dtype):
    torch.manual_seed(0)
    input_sl, head, dim, out_sl = 4, 4, 64, 8

    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    k_scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc)

    k_ref = _ref_downcast(k, k_scale, loc, out_sl)
    v_ref = _ref_downcast(v, v_scale, loc, out_sl)

    torch.testing.assert_close(
        k_out, k_ref, msg=f"scale={scale_val}: kernel vs reference mismatch"
    )
    torch.testing.assert_close(
        v_out, v_ref, msg=f"scale={scale_val}: kernel vs reference mismatch"
    )


# ---------------------------------------------------------------------------
# Clamping: values exceeding ±448 must be saturated to fp8 max/min.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_downcast_fp8_clamp(dtype):
    input_sl, head, dim, out_sl = 2, 1, 8, 4

    # All values well outside fp8 range so clamping is unavoidable.
    k = torch.full((input_sl, head, dim), 1000.0, dtype=dtype, device="cuda")
    v = torch.full((input_sl, head, dim), -1000.0, dtype=dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, scale, scale, loc)

    # Reference fp8 max/min byte values (E4M3: 0x7e = 448.0, 0xfe = -448.0)
    fp8_pos_max = (
        torch.tensor([_FP8_E4M3_MAX], dtype=dtype, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .item()
    )
    fp8_neg_max = (
        torch.tensor([-_FP8_E4M3_MAX], dtype=dtype, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .item()
    )

    assert (
        k_out[:input_sl] == fp8_pos_max
    ).all(), "large positive values should clamp to fp8 max"
    assert (
        v_out[:input_sl] == fp8_neg_max
    ).all(), "large negative values should clamp to fp8 min"


# ---------------------------------------------------------------------------
# Scatter: loc controls which output rows receive the converted values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_downcast_fp8_loc(dtype):
    torch.manual_seed(7)
    input_sl, head, dim, out_sl = 3, 2, 32, 10

    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Write to non-contiguous output positions: 0, 5, 9
    loc = torch.tensor([0, 5, 9], dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, scale, scale, loc)

    k_ref = _ref_downcast(k, scale, loc, out_sl)
    v_ref = _ref_downcast(v, scale, loc, out_sl)

    torch.testing.assert_close(
        k_out, k_ref, msg="loc scatter: kernel vs reference mismatch"
    )
    torch.testing.assert_close(
        v_out, v_ref, msg="loc scatter: kernel vs reference mismatch"
    )

    # Slots not in loc must remain zero
    written = {0, 5, 9}
    for i in range(out_sl):
        if i not in written:
            assert not k_out[i].any(), f"k_out[{i}] should be zero (not a loc target)"
            assert not v_out[i].any(), f"v_out[{i}] should be zero (not a loc target)"


# ---------------------------------------------------------------------------
# mult/offset: output index = loc[i] * mult + offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mult,offset", [(2, 0), (1, 3), (2, 1)])
def test_downcast_fp8_mult_offset(mult, offset, dtype):
    torch.manual_seed(3)
    input_sl, head, dim = 2, 2, 32
    out_sl = input_sl * mult + offset + 4  # ensure output is large enough

    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, scale, scale, loc, mult=mult, offset=offset)

    k_ref = _ref_downcast(k, scale, loc, out_sl, mult=mult, offset=offset)
    v_ref = _ref_downcast(v, scale, loc, out_sl, mult=mult, offset=offset)

    torch.testing.assert_close(
        k_out, k_ref, msg=f"mult={mult},offset={offset}: kernel vs reference mismatch"
    )
    torch.testing.assert_close(
        v_out, v_ref, msg=f"mult={mult},offset={offset}: kernel vs reference mismatch"
    )


# ---------------------------------------------------------------------------
# static_cast conversion: verify static_cast<fp8_e4m3_t> matches PyTorch fp8
# for a comprehensive sweep including values near and at the fp8 boundary.
# This specifically validates that the static_cast fallback (used after
# removing explicit __nv_cvt_*raw_to_fp8 from dtype_trait) produces the
# same bit patterns as the reference path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_downcast_fp8_static_cast_boundary(dtype):
    """Test conversion accuracy near ±448 fp8 boundary using static_cast path."""
    torch.manual_seed(0)
    # Values specifically chosen to stress the static_cast conversion path:
    #   - exactly at ±448 (representable fp8 max)
    #   - just inside the range
    #   - just outside (must saturate)
    #   - zero, small, and mid-range values
    boundary_vals = [
        0.0,
        1.0,
        -1.0,
        100.0,
        -100.0,
        447.0,
        -447.0,
        448.0,
        -448.0,
        449.0,
        -449.0,
        1000.0,
        -1000.0,
    ]
    input_sl = len(boundary_vals)
    head, dim, out_sl = 1, 8, input_sl

    base = torch.tensor(boundary_vals, dtype=dtype, device="cuda")
    k = base.unsqueeze(1).unsqueeze(2).expand(input_sl, head, dim).contiguous()
    v = (-base).unsqueeze(1).unsqueeze(2).expand(input_sl, head, dim).contiguous()
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")

    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    downcast_fp8(k, v, k_out, v_out, scale, scale, loc)

    k_ref = _ref_downcast(k, scale, loc, out_sl)
    v_ref = _ref_downcast(v, scale, loc, out_sl)

    torch.testing.assert_close(
        k_out, k_ref, msg="boundary values: k static_cast vs reference mismatch"
    )
    torch.testing.assert_close(
        v_out, v_ref, msg="boundary values: v static_cast vs reference mismatch"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
