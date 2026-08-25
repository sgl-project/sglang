import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560


def _reference_hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager reference, mirrors GatedResidual._combine_compute in
    hyperconnection.py.

    With compute_dtype=float64 this serves as the near-exact reference for
    precision assertions.
    """
    R = residual.to(compute_dtype).unflatten(-1, (hc, hs))
    gates = 2 * torch.sigmoid(
        F.linear(normed_residual.to(compute_dtype), inject_weight.to(compute_dtype))
        / hc
    )
    injection = block_output.to(compute_dtype).unsqueeze(-2) * gates.unsqueeze(-1)
    return (R + injection).flatten(-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype, hc: int = HC_COUNT, hs: int = HIDDEN_SIZE):
    torch.manual_seed(0)
    block_output = torch.randn(num_tokens, hs, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    normed_residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    inject_weight = torch.randn(hc, hc * hs, dtype=dtype, device="cuda") * 0.02
    return block_output, residual, normed_residual, inject_weight


# Tolerances are at the output-dtype quantization floor, measured against the
# fp64 reference on B300 (sm103), worst case over M in {1,7,128,8192}: bf16 max
# rel err 7.8e-3 (exactly 1 ulp at the lower edge of a binade, 9 / 83.8M
# elements at M=8192), fp16 max rel err below 1e-3 (1 ulp = 9.8e-4). The kernel
# accumulates in fp32 like the eager fp32 path; the residual error is fp32
# dot-product reordering flipping the final bf16/fp16 rounding at ties.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 8192])
def test_hc_combine_correctness(dtype, num_tokens):
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        num_tokens, dtype
    )

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hc_combine_out_param(dtype):
    block_output, residual, normed_residual, inject_weight = _make_inputs(64, dtype)
    out = torch.empty_like(residual)

    result = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        out=out,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected, **_TOLERANCES[dtype])


def test_hc_combine_3d_input():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(32, dtype)
    block_output = block_output.reshape(4, 8, HIDDEN_SIZE)
    residual = residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)
    normed_residual = normed_residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert out.shape == residual.shape
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


def test_hc_combine_unsupported_dtype():
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        4, torch.float32
    )
    with pytest.raises(RuntimeError, match="dtype"):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )


def test_hc_combine_bad_hidden_size():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        4, dtype, hc=4, hs=1000  # 4 * 1000 = 4000, not a multiple of 2048
    )
    with pytest.raises(RuntimeError, match="2048"):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            4,
            1000,
        )


def test_hc_combine_dtype_mismatch():
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        4, torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="dtype"):
        hc_combine(
            block_output.float(),
            residual,
            normed_residual,
            inject_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )


def test_hc_combine_shape_mismatch():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(4, dtype)
    bad_weight = torch.randn(HC_COUNT, HC_COUNT * HIDDEN_SIZE + 2048, dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            bad_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
