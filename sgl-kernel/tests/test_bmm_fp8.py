# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_bmm_fp8.py

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bmm_fp8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_bmm_fp8(input_dtype, mat2_dtype, res_dtype):
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")

    input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    res = torch.empty([16, 48, 80], device="cuda", dtype=res_dtype)
    bmm_fp8(input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype, res)

    reference = torch.bmm(input, mat2)
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
