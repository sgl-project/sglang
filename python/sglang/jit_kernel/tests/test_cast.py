import pytest
import torch

from sglang.jit_kernel.cast import downcast_fp8

DTYPES = [torch.bfloat16, torch.float16]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
