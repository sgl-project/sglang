# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_bmm_fp8.py
#
# Tests the flashinfer JIT bmm_fp8 path and optionally compares it against
# the sgl_kernel AOT cuBLASLt baseline.

import pytest
import torch
import torch.nn.functional as F

import flashinfer

try:
    from sgl_kernel import bmm_fp8 as sgl_bmm_fp8

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_OUT_DTYPES = [torch.bfloat16, torch.float16]
_SHAPES = [
    (16, 48, 64, 80),  # (batch, M, K, N) — original test shape
    (2, 48, 64, 80),
    (4, 16, 32, 16),
    (1, 128, 256, 128),
]


def to_float8(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("batch,M,K,N", _SHAPES)
@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
@pytest.mark.parametrize("b_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("a_dtype", _FP8_DTYPES)
def test_jit_bmm_fp8(a_dtype, b_dtype, out_dtype, batch, M, K, N):
    """flashinfer JIT bmm_fp8 should match a float32 torch.bmm reference."""
    if a_dtype == torch.float8_e5m2 and b_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")

    A = torch.randn([batch, M, K], device="cuda", dtype=torch.bfloat16)
    # B shape (batch, K, N): construct as (batch, N, K) then transpose to row-major
    B = torch.randn([batch, N, K], device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )

    A_fp8, A_inv_s = to_float8(A, dtype=a_dtype)
    B_fp8, B_inv_s = to_float8(B, dtype=b_dtype)

    res = flashinfer.bmm_fp8(A_fp8, B_fp8, A_inv_s, B_inv_s, out_dtype)

    assert res.shape == (batch, M, N), f"shape mismatch: got {res.shape}"
    reference = torch.bmm(A, B)
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99, f"cosine similarity {cos_sim:.4f} < 0.99"
    print(
        f"flashinfer JIT vs reference | "
        f"shape=({batch},{M},{K},{N}) a={a_dtype} b={b_dtype} out={out_dtype} | "
        f"cos_sim={cos_sim:.6f} PASSED"
    )


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not installed")
@pytest.mark.parametrize("batch,M,K,N", _SHAPES)
@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
@pytest.mark.parametrize("b_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("a_dtype", _FP8_DTYPES)
def test_jit_bmm_fp8_agrees_with_sgl_kernel(
    a_dtype, b_dtype, out_dtype, batch, M, K, N
):
    """flashinfer JIT and sgl_kernel AOT cuBLASLt results should agree."""
    if a_dtype == torch.float8_e5m2 and b_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")

    A = torch.randn([batch, M, K], device="cuda", dtype=torch.bfloat16)
    B = torch.randn([batch, N, K], device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )

    A_fp8, A_inv_s = to_float8(A, dtype=a_dtype)
    B_fp8, B_inv_s = to_float8(B, dtype=b_dtype)

    res_jit = flashinfer.bmm_fp8(A_fp8, B_fp8, A_inv_s, B_inv_s, out_dtype)
    res_aot = sgl_bmm_fp8(A_fp8, B_fp8, A_inv_s, B_inv_s, out_dtype)

    cos_sim = F.cosine_similarity(
        res_aot.reshape(-1).float(), res_jit.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.999, (
        f"JIT vs AOT cosine similarity {cos_sim:.4f} < 0.999 "
        f"(a={a_dtype}, b={b_dtype}, out={out_dtype}, shape=({batch},{M},{K},{N}))"
    )
    print(
        f"flashinfer JIT vs sgl_kernel AOT | "
        f"shape=({batch},{M},{K},{N}) a={a_dtype} b={b_dtype} out={out_dtype} | "
        f"cos_sim={cos_sim:.6f} PASSED"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
