"""Fused in-place QK norm on a strided fused-QKV slice.

`test_qknorm.py` only ever feeds the kernel standalone contiguous `q`/`k`, but
`apply_qk_norm` hands it 3-D views of a packed QKV projection whose rows are
strided over the K/V columns. The kernel takes `q_stride`/`k_stride` as
parameters and indexes `token * stride + head * head_dim`, so that layout is
supported by construction -- these tests pin it, since nothing in-tree covered
it before.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.norm import (
    can_use_fused_inplace_qknorm,
    fused_inplace_qknorm,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=25, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
EPS = 1e-6

# (num_tokens, num_q_heads, num_kv_heads, head_dim)
LAYOUTS = [
    (4096, 32, 8, 128),  # Qwen3-8B: row stride 6144
    (1024, 32, 4, 128),  # Qwen3-30B-A3B: row stride 5120
    (17, 16, 8, 128),  # ragged, GQA 2:1
    (512, 16, 16, 64),  # MHA, head_dim 64
    (128, 8, 8, 256),  # head_dim 256 (CTA kernel path)
]
DTYPES = [torch.bfloat16, torch.float16]


def _fused_qkv(num_tokens, num_q_heads, num_kv_heads, head_dim, dtype, v_fill=None):
    """Build a packed QKV buffer and return 3-D strided views of q and k."""
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn(
        num_tokens, q_size + 2 * kv_size, device=DEVICE, dtype=torch.float32
    ).to(dtype)
    if v_fill is not None:
        qkv[:, q_size + kv_size :] = v_fill
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q3 = q.view(num_tokens, num_q_heads, head_dim)
    k3 = k.view(num_tokens, num_kv_heads, head_dim)
    # The layout under test: rows strided over the K/V columns.
    assert q3.stride(0) == q_size + 2 * kv_size
    assert not q3.is_contiguous()
    return qkv, q3, k3, v


def _weights(head_dim, dtype):
    return (
        torch.randn(head_dim, device=DEVICE, dtype=dtype),
        torch.randn(head_dim, device=DEVICE, dtype=dtype),
    )


def _reference(x, weight, eps=EPS):
    """RMSNorm per head, in fp32, matching RMSNorm.forward_native."""
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _skip_unless_supported(head_dim, dtype):
    if not can_use_fused_inplace_qknorm(head_dim, dtype):
        pytest.skip(f"JIT QK-norm unavailable for head_dim={head_dim} {dtype}")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_matches_contiguous_bitwise(layout, dtype):
    """A strided slice and a dense copy of the same data must produce
    bit-identical output -- the property a single row stride cannot provide."""
    torch.manual_seed(0)
    head_dim = layout[-1]
    _skip_unless_supported(head_dim, dtype)

    _, q_strided, k_strided, _ = _fused_qkv(*layout, dtype)
    q_dense, k_dense = q_strided.contiguous(), k_strided.contiguous()
    q_weight, k_weight = _weights(head_dim, dtype)

    fused_inplace_qknorm(
        q_strided, k_strided, q_weight, k_weight, EPS, head_dim=head_dim
    )
    fused_inplace_qknorm(q_dense, k_dense, q_weight, k_weight, EPS, head_dim=head_dim)

    assert torch.equal(q_strided, q_dense)
    assert torch.equal(k_strided, k_dense)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_matches_reference(layout, dtype):
    torch.manual_seed(0)
    head_dim = layout[-1]
    _skip_unless_supported(head_dim, dtype)

    _, q, k, _ = _fused_qkv(*layout, dtype)
    q_weight, k_weight = _weights(head_dim, dtype)
    q_expected = _reference(q.clone(), q_weight)
    k_expected = _reference(k.clone(), k_weight)

    fused_inplace_qknorm(q, k, q_weight, k_weight, EPS, head_dim=head_dim)

    torch.testing.assert_close(q, q_expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_operates_in_place_without_copying(layout):
    """The point of the strided path: no reallocation, and the V columns of the
    shared buffer are left alone."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    head_dim = layout[-1]
    _skip_unless_supported(head_dim, dtype)

    sentinel = 3.0
    qkv, q, k, v = _fused_qkv(*layout, dtype, v_fill=sentinel)
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()
    q_weight, k_weight = _weights(head_dim, dtype)

    fused_inplace_qknorm(q, k, q_weight, k_weight, EPS, head_dim=head_dim)

    assert q.data_ptr() == q_ptr
    assert k.data_ptr() == k_ptr
    # An out-of-column write would land in V.
    assert torch.equal(v, torch.full_like(v, sentinel))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_does_not_read_outside_its_columns(layout):
    """Poison V with NaN: a row-major (single-stride) misread of a q row runs
    past the K columns into V and would surface as NaN in the output."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    head_dim = layout[-1]
    _skip_unless_supported(head_dim, dtype)

    qkv, q, k, _ = _fused_qkv(*layout, dtype, v_fill=float("nan"))
    q_weight, k_weight = _weights(head_dim, dtype)

    fused_inplace_qknorm(q, k, q_weight, k_weight, EPS, head_dim=head_dim)

    assert torch.isfinite(q).all()
    assert torch.isfinite(k).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
