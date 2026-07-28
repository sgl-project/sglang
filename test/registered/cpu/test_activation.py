import sys

import pytest
import torch
from utils import GeluAndMul, SiluAndMul, precision

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

torch.manual_seed(1234)

M = [128, 129, 257]
N = [22016, 22018]
DTYPES = [torch.float16, torch.bfloat16]


def _assert_close(ref_out, out):
    atol = rtol = precision[ref_out.dtype]
    torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("m", M)
def test_activation(m, n, dtype):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    x = torch.randn([m, n], dtype=dtype)
    _assert_close(SiluAndMul(x), torch.ops.sgl_kernel.silu_and_mul_cpu(x))

    x = torch.randn([m, n], dtype=dtype)
    _assert_close(
        GeluAndMul(x, approximate="none"), torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
    )

    x = torch.randn([m, n], dtype=dtype)
    _assert_close(
        GeluAndMul(x, approximate="tanh"),
        torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x),
    )


@pytest.mark.parametrize("gate_3d", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("m", [1, 17, 128])
def test_fused_sigmoid_mul(m, num_heads, head_dim, dtype, gate_3d):
    x = torch.randn([m, num_heads * head_dim], dtype=dtype)
    if gate_3d:
        gate_storage = torch.randn([m, num_heads, head_dim * 2], dtype=dtype)
        gate = gate_storage[..., :head_dim]
        assert not gate.is_contiguous()
    else:
        gate = torch.randn_like(x)

    gate_ref = gate.reshape(m, -1) if gate_3d else gate
    _assert_close(
        x * torch.sigmoid(gate_ref),
        torch.ops.sgl_kernel.fused_sigmoid_mul_cpu(x, gate, False),
    )

    x_inplace = x.clone()
    ref_inplace = x_inplace * torch.sigmoid(gate_ref)
    out_inplace = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu(x_inplace, gate, True)
    assert out_inplace.data_ptr() == x_inplace.data_ptr()
    _assert_close(ref_inplace, x_inplace)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
