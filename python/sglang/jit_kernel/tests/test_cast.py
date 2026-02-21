import pytest
import torch
from sgl_kernel.elementwise import downcast_fp8 as aot_downcast_fp8

from sglang.jit_kernel.cast import downcast_fp8 as jit_downcast_fp8

DTYPES = [torch.bfloat16, torch.float16]


def _make_inputs(input_sl, head, dim, out_sl, dtype):
    k = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    v = torch.randn(input_sl, head, dim, dtype=dtype, device="cuda")
    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    loc = torch.arange(input_sl, dtype=torch.int64, device="cuda")
    return k, v, k_out, v_out, k_scale, v_scale, loc


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_sl,head,dim,out_sl", [(4, 8, 128, 16)])
def test_downcast_fp8(input_sl, head, dim, out_sl, dtype):
    k, v, _, _, k_scale, v_scale, loc = _make_inputs(input_sl, head, dim, out_sl, dtype)

    jit_k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    jit_v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    jit_downcast_fp8(k, v, jit_k_out, jit_v_out, k_scale, v_scale, loc)

    aot_k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    aot_v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device="cuda")
    aot_downcast_fp8(k, v, aot_k_out, aot_v_out, k_scale, v_scale, loc)

    assert torch.all(jit_k_out == aot_k_out), "k_out mismatch between JIT and AOT"
    assert torch.all(jit_v_out == aot_v_out), "v_out mismatch between JIT and AOT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
