"""ERNIE fused norm/scale/shift fast paths must stay bit-exact vs eager."""

import sys
from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.ernie_image as ernie_image
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_gated_norm_scale_shift,
    _ernie_norm_scale_shift,
    _ernie_qknorm_rope,
    _ernie_qknorm_rope_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(1, 4216, 4096), (2, 1140, 4096), (1, 128, 2048)])
def test_fused_norm_scale_shift_is_bit_exact(shape):
    # (1, 4216, 4096) is the real ERNIE-Image shape (1024^2 image + text
    # tokens, hidden 4096); 2048 covers the threads_per_row=32 regime.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = RMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden))
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    update = torch.randn_like(x)
    scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = _ernie_norm_scale_shift(norm, x, scale, shift)
        ref = norm(x) * (1 + scale) + shift
        assert torch.equal(out, ref)

        out2, res = _ernie_gated_norm_scale_shift(
            norm, residual, update, gate, scale, shift
        )
        res_ref = residual + gate * update
        ref2 = norm(res_ref) * (1 + scale) + shift
        assert torch.equal(res, res_ref)
        assert torch.equal(out2, ref2)

    # the fast paths must actually be in use (not silently disabled)
    assert ernie_image._ERNIE_NORM.verified
    assert ernie_image._ERNIE_GATED_NORM.verified
    assert not ernie_image._ERNIE_NORM.disabled
    assert not ernie_image._ERNIE_GATED_NORM.disabled


def test_fused_qknorm_rope_is_bit_exact():
    torch.manual_seed(1)
    ernie_image._ERNIE_QKNORM_ROPE.disabled = False
    ernie_image._ERNIE_QKNORM_ROPE.verified = False
    batch, seq, heads, head_dim = 1, 257, 32, 128
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    positions = torch.arange(seq, device="cuda", dtype=torch.long)

    q_ref, k_ref = _ernie_qknorm_rope_reference(
        q.clone(), k.clone(), q_norm, k_norm, head_dim, cos, sin
    )
    q_out, k_out = _ernie_qknorm_rope(
        q,
        k,
        q_norm,
        k_norm,
        head_dim,
        cos,
        sin,
        cache,
        positions,
    )

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert ernie_image._ERNIE_QKNORM_ROPE.verified
    assert not ernie_image._ERNIE_QKNORM_ROPE.disabled


def test_qknorm_rope_first_attempt_exception_uses_pristine_inputs():
    torch.manual_seed(2)
    ernie_image._ERNIE_QKNORM_ROPE.disabled = False
    ernie_image._ERNIE_QKNORM_ROPE.verified = False
    batch, seq, heads, head_dim = 1, 17, 4, 128
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    positions = torch.arange(seq, device="cuda", dtype=torch.long)
    q_ref, k_ref = _ernie_qknorm_rope_reference(
        q.clone(), k.clone(), q_norm, k_norm, head_dim, cos, sin
    )

    def mutate_then_raise(**kwargs):
        kwargs["q"].zero_()
        kwargs["k"].zero_()
        raise RuntimeError("synthetic kernel failure")

    with patch.object(ernie_image, "apply_qk_norm_rope", mutate_then_raise):
        q_out, k_out = _ernie_qknorm_rope(
            q,
            k,
            q_norm,
            k_norm,
            head_dim,
            cos,
            sin,
            cache,
            positions,
        )

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert ernie_image._ERNIE_QKNORM_ROPE.disabled


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
