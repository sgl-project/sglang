import sys

import pytest
import torch

from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"

SHAPES = get_ci_test_range(
    [(128, 64, 128), (64, 32, 64)],
    [(128, 64, 128)],
)
NUM_HEADS = get_ci_test_range([8, 16, 32, 64], [16, 32])
BATCH_SIZES = get_ci_test_range(
    [1, 4, 17, 64, 257, 1024, 4096, 16384],
    [1, 64, 1024, 16384],
)


def _ref(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype):
    s, h, qk_nope = k_nope.shape
    qk_rope = k_pe.shape[-1]
    v_head = v.shape[-1]
    if k_pe.dim() == 3:
        k_pe = k_pe.squeeze(1)

    k_bf16 = torch.empty(
        (s, h, qk_nope + qk_rope), dtype=k_nope.dtype, device=k_nope.device
    )
    k_bf16[..., :qk_nope] = k_nope
    k_bf16[..., qk_nope:] = k_pe.unsqueeze(1).expand(-1, h, -1)

    k_fp8 = (k_bf16.float() * k_scale_inv).to(fp8_dtype)
    v_fp8 = (v.float() * v_scale_inv).to(fp8_dtype)
    return k_fp8, v_fp8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_correctness(dtype, shape, num_heads, batch_size):
    qk_nope, qk_rope, v_head = shape

    torch.manual_seed(0)
    k_nope = torch.randn((batch_size, num_heads, qk_nope), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((batch_size, 1, qk_rope), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, num_heads, v_head), dtype=dtype, device=DEVICE)

    k_scale_inv = 0.7
    v_scale_inv = 1.3

    k_fp8, v_fp8 = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, k_scale_inv=k_scale_inv, v_scale_inv=v_scale_inv
    )

    k_ref, v_ref = _ref(k_nope, k_pe, v, k_scale_inv, v_scale_inv, torch.float8_e4m3fn)

    torch.testing.assert_close(k_fp8.float(), k_ref.float(), rtol=1e-2, atol=0.5)
    torch.testing.assert_close(v_fp8.float(), v_ref.float(), rtol=1e-2, atol=0.5)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_strided_inputs(dtype):
    s, h = 16, 32
    qk_nope, qk_rope, v_head = 128, 64, 128

    full = torch.randn(
        (s, h, qk_nope * 2), dtype=dtype, device=DEVICE, requires_grad=False
    )
    k_nope = full[..., qk_nope:]
    assert k_nope.stride(-1) == 1

    k_pe = torch.randn((s, 1, qk_rope), dtype=dtype, device=DEVICE)
    v = torch.randn((s, h, v_head), dtype=dtype, device=DEVICE)

    k_fp8, v_fp8 = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    k_ref, v_ref = _ref(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    torch.testing.assert_close(k_fp8.float(), k_ref.float(), rtol=1e-2, atol=0.5)
    torch.testing.assert_close(v_fp8.float(), v_ref.float(), rtol=1e-2, atol=0.5)


def test_kpe_2d_accepted():
    s, h = 8, 16
    qk_nope, qk_rope, v_head = 128, 64, 128
    dtype = torch.bfloat16

    k_nope = torch.randn((s, h, qk_nope), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((s, qk_rope), dtype=dtype, device=DEVICE)
    v = torch.randn((s, h, v_head), dtype=dtype, device=DEVICE)

    k_fp8, v_fp8 = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    k_ref, v_ref = _ref(k_nope, k_pe.unsqueeze(1), v, 1.0, 1.0, torch.float8_e4m3fn)
    torch.testing.assert_close(k_fp8.float(), k_ref.float(), rtol=1e-2, atol=0.5)
    torch.testing.assert_close(v_fp8.float(), v_ref.float(), rtol=1e-2, atol=0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
