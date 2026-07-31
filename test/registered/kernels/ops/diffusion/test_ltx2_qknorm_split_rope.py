import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.ltx2_qknorm_split_rope import (
    can_use_ltx2_qknorm_split_rope_cuda,
    ltx2_qknorm_split_rope_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

BF16_FUSED_ATOL = 1.6e-1


def _require_cuda_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("LTX2 QKNorm split-RoPE CUDA path is validated on B200")


@pytest.fixture(autouse=True)
def cuda_setup():
    _require_cuda_b200()
    torch.cuda.manual_seed(20260630)


def _make_cos_sin(
    batch: int, seq_len: int, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2
    cos = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    sin = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    return cos, sin


def _apply_split_rotary_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_dtype = x.dtype
    batch = x.shape[0]
    _, num_heads, seq_len, _ = cos.shape
    x = x.reshape(batch, seq_len, num_heads, -1).swapaxes(1, 2)
    last = x.shape[-1]
    half = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, half)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second_x)
    out[..., 1:, :].addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    return out.swapaxes(1, 2).reshape(batch, seq_len, -1).to(dtype=x_dtype)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rms_norm isn't autocast fp32-preserving, so feed fp32 inputs directly
    # to keep the normalized value unrounded until the final RoPE output.
    q_norm = F.rms_norm(q.float(), (q.shape[-1],), q_weight.float(), eps)
    k_norm = F.rms_norm(k.float(), (k.shape[-1],), k_weight.float(), eps)
    q_ref = _apply_split_rotary_ref(q_norm, q_cos, q_sin)
    k_ref = _apply_split_rotary_ref(k_norm, k_cos, k_sin)
    return q_ref.to(dtype=torch.bfloat16), k_ref.to(dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim",
    [
        (1, 3, 3, 32, 128),
        (1, 5, 2, 32, 64),
        (2, 4, 3, 32, 64),
    ],
)
def test_ltx2_qknorm_split_rope_matches_torch_exactly(
    batch: int, q_seq: int, k_seq: int, num_heads: int, head_dim: int
) -> None:
    hidden = num_heads * head_dim
    eps = 1e-6
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    q_ref, k_ref = _reference(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, eps)
    q_out, k_out = ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        eps=eps,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


def test_ltx2_qknorm_split_rope_rejects_unsupported_inputs() -> None:
    q = torch.randn((1, 3, 4096), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_cos, q_sin = _make_cos_sin(1, 3, 32, 128)
    q_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q.float(),
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos.transpose(-1, -2),
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )


def test_ltx2_qknorm_split_rope_custom_op_torch_compile_fullgraph() -> None:
    batch, q_seq, k_seq, num_heads, head_dim = 1, 3, 2, 32, 64
    hidden = num_heads * head_dim
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def fn(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight):
        return ltx2_qknorm_split_rope_cuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            eps=1e-6,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    compiled = torch.compile(fn, fullgraph=True)
    q_out, k_out = compiled(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight)
    q_ref, k_ref = _reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, 1e-6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
