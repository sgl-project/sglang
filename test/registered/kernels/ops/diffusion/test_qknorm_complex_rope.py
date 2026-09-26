# SPDX-License-Identifier: Apache-2.0
"""CUDA Q/K RMSNorm + complex RoPE (Qwen-Image 2.1) must be bit-exact with the eager chain.

Regressions caught: a changed reduction order or rounding point in the fused
kernel (mismatch against ``RMSNorm(cast_x_before_out_mul=True)`` + complex64
multiply), wrong FMA orientation for this GPU, wrong prefix/token row placement
in the packed K/V buffers, and predicates admitting layouts the kernel cannot
address.
"""

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_qknorm_complex_rope_cuda,
    can_use_qknorm_complex_rope_pack,
    qknorm_complex_rope_cuda,
    qknorm_complex_rope_pack_,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

EPS = 1e-6
HEAD_DIM = 128


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def make_norm():
    norm = RMSNorm(HEAD_DIM, EPS, cast_x_before_out_mul=True, force_native=True).to(
        device="cuda", dtype=torch.bfloat16
    )
    norm.weight.normal_()
    return norm


def make_rope(seq, scale=20.0):
    angles = torch.randn(seq, HEAD_DIM // 2, device="cuda") * scale
    return torch.polar(torch.ones_like(angles), angles)


def eager_norm_rope(x, norm, rope):
    normed = norm(x)
    z = torch.view_as_complex(normed.float().reshape(*normed.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def assert_bits_equal(actual, expected):
    assert actual.dtype is expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "shape", [(1, 4096, 32, 128), (1, 18, 32, 128), (2, 17, 4, 128), (1, 1, 1, 128)]
)
@pytest.mark.parametrize("amplitude", [1e-3, 1.0, 40.0])
def test_norm_rope_matches_eager(shape, amplitude):
    torch.manual_seed(0)
    norm = make_norm()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude
    rope = make_rope(shape[1])
    assert can_use_qknorm_complex_rope_cuda(x, norm.weight, rope)
    out = qknorm_complex_rope_cuda(x, norm.weight, rope, EPS)
    assert_bits_equal(out, eager_norm_rope(x, norm, rope))


def test_norm_rope_is_sensitive_to_fma_orientation():
    # The two complex-multiply FMA orientations differ on a large random sample;
    # the equality test above therefore also pins the orientation probe.
    torch.manual_seed(1)
    x = torch.randn(1, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)
    rope = make_rope(4096)
    normed = x  # rotation only: pretend the norm is the identity for this check
    z = torch.view_as_complex(normed.float().reshape(*normed.shape[:-1], -1, 2))
    re, im = z.real, z.imag
    c, s = rope.real[None, :, None], rope.imag[None, :, None]
    imag_a = torch.addcmul(im * c, re, s).to(torch.bfloat16)
    imag_b = torch.addcmul(re * s, im, c).to(torch.bfloat16)
    assert not torch.equal(imag_a, imag_b)


def _pack_inputs(batch, seq, prefix, heads):
    torch.manual_seed(2)
    q = torch.randn(batch, seq, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    kp = torch.randn(
        batch, prefix, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    vp = torch.randn_like(kp)
    rope = make_rope(seq)
    return q, k, v, kp, vp, rope


@pytest.mark.parametrize("batch,seq,prefix,heads", [(1, 4096, 18, 32), (2, 17, 5, 4)])
@pytest.mark.parametrize("in_place", [False, True])
def test_pack_matches_eager(batch, seq, prefix, heads, in_place):
    q, k, v, kp, vp, rope = _pack_inputs(batch, seq, prefix, heads)
    norm_q, norm_k = make_norm(), make_norm()
    expected_q = eager_norm_rope(q, norm_q, rope)
    expected_k = torch.cat([kp, eager_norm_rope(k, norm_k, rope)], dim=1)
    expected_v = torch.cat([vp, v], dim=1)

    k_out = torch.empty(
        batch, prefix + seq, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v_out = torch.empty_like(k_out)
    if in_place:
        # the projection wrote raw K and V into the token rows already
        k_out[:, prefix:].copy_(k)
        v_out[:, prefix:].copy_(v)
        k_src, v_src = None, None
    else:
        k_src, v_src = k, v
    q_work = q.clone()
    assert can_use_qknorm_complex_rope_pack(
        q_work, k_out, v_out, norm_q.weight, norm_k.weight, rope, kp, vp, k_src, v_src
    )
    qknorm_complex_rope_pack_(
        q_work,
        k_out,
        v_out,
        norm_q.weight,
        norm_k.weight,
        rope,
        kp,
        vp,
        k_src,
        v_src,
        EPS,
    )
    assert_bits_equal(q_work, expected_q)
    assert_bits_equal(k_out, expected_k)
    assert_bits_equal(v_out, expected_v)


def test_pack_accepts_padded_token_stride():
    # Column views of a wider [B, P+S, 3C] buffer: heads contiguous, token stride 3C.
    batch, seq, prefix, heads = 1, 33, 4, 8
    q, k, v, kp, vp, rope = _pack_inputs(batch, seq, prefix, heads)
    norm_q, norm_k = make_norm(), make_norm()
    wide = torch.empty(
        batch, prefix + seq, 3 * heads * HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    hd = heads * HEAD_DIM
    q_view = wide[:, prefix:, :hd].view(batch, seq, heads, HEAD_DIM)
    k_out = wide[:, :, hd : 2 * hd].view(batch, prefix + seq, heads, HEAD_DIM)
    v_out = wide[:, :, 2 * hd :].view(batch, prefix + seq, heads, HEAD_DIM)
    q_view.copy_(q)
    k_out[:, prefix:].copy_(k)
    v_out[:, prefix:].copy_(v)
    assert can_use_qknorm_complex_rope_pack(
        q_view, k_out, v_out, norm_q.weight, norm_k.weight, rope, kp, vp, None, None
    )
    qknorm_complex_rope_pack_(
        q_view,
        k_out,
        v_out,
        norm_q.weight,
        norm_k.weight,
        rope,
        kp,
        vp,
        None,
        None,
        EPS,
    )
    assert_bits_equal(q_view, eager_norm_rope(q, norm_q, rope))
    assert_bits_equal(k_out, torch.cat([kp, eager_norm_rope(k, norm_k, rope)], dim=1))
    assert_bits_equal(v_out, torch.cat([vp, v], dim=1))


def test_predicates_reject_unsupported_layouts():
    norm = make_norm()
    x = torch.randn(1, 17, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rope = make_rope(17)
    assert can_use_qknorm_complex_rope_cuda(x, norm.weight, rope)
    assert not can_use_qknorm_complex_rope_cuda(x.cpu(), norm.weight.cpu(), rope.cpu())
    assert not can_use_qknorm_complex_rope_cuda(x.half(), norm.weight.half(), rope)
    assert not can_use_qknorm_complex_rope_cuda(
        x[..., :64], norm.weight[:64], rope[:, :32]
    )
    assert not can_use_qknorm_complex_rope_cuda(x.transpose(1, 2), norm.weight, rope)
    assert not can_use_qknorm_complex_rope_cuda(x, norm.weight, rope[:-1])
    assert not can_use_qknorm_complex_rope_cuda(
        x, norm.weight, rope.to(torch.complex128)
    )
    assert not can_use_qknorm_complex_rope_cuda(x, norm.weight.float(), rope)
    k_out = torch.empty(1, 17 + 3, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kp = torch.randn(1, 3, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    assert can_use_qknorm_complex_rope_pack(
        x, k_out, k_out.clone(), norm.weight, norm.weight, rope, kp, kp, None, None
    )
    # wrong prefix + seq length, batch mismatch, non-contiguous prefix, short K source
    assert not can_use_qknorm_complex_rope_pack(
        x,
        k_out[:, :-1],
        k_out.clone(),
        norm.weight,
        norm.weight,
        rope,
        kp,
        kp,
        None,
        None,
    )
    assert not can_use_qknorm_complex_rope_pack(
        x,
        k_out,
        k_out.clone(),
        norm.weight,
        norm.weight,
        rope,
        kp.expand(2, -1, -1, -1),
        kp,
        None,
        None,
    )
    assert not can_use_qknorm_complex_rope_pack(
        x,
        k_out,
        k_out.clone(),
        norm.weight,
        norm.weight,
        rope,
        kp.transpose(1, 2),
        kp,
        None,
        None,
    )
    assert not can_use_qknorm_complex_rope_pack(
        x, k_out, k_out.clone(), norm.weight, norm.weight, rope, kp, kp, x[:, :-1], None
    )


def test_pack_graph_capture_and_replay():
    batch, seq, prefix, heads = 1, 65, 3, 8
    q, k, v, kp, vp, rope = _pack_inputs(batch, seq, prefix, heads)
    norm_q, norm_k = make_norm(), make_norm()
    q_work = q.clone()
    k_out = torch.empty(
        batch, prefix + seq, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v_out = torch.empty_like(k_out)
    k_out[:, prefix:].copy_(k)
    v_out[:, prefix:].copy_(v)
    # warm the JIT module outside the capture
    qknorm_complex_rope_pack_(
        q_work.clone(),
        k_out.clone(),
        v_out.clone(),
        norm_q.weight,
        norm_k.weight,
        rope,
        kp,
        vp,
        None,
        None,
        EPS,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qknorm_complex_rope_pack_(
            q_work,
            k_out,
            v_out,
            norm_q.weight,
            norm_k.weight,
            rope,
            kp,
            vp,
            None,
            None,
            EPS,
        )
    # new inputs through the captured buffers
    q2, k2, v2, kp2, vp2, _ = _pack_inputs(batch, seq, prefix, heads)
    q_work.copy_(q2)
    k_out[:, prefix:].copy_(k2)
    v_out[:, prefix:].copy_(v2)
    kp.copy_(kp2)
    vp.copy_(vp2)
    graph.replay()
    torch.cuda.synchronize()
    assert_bits_equal(q_work, eager_norm_rope(q2, norm_q, rope))
    assert_bits_equal(k_out, torch.cat([kp2, eager_norm_rope(k2, norm_k, rope)], dim=1))
    assert_bits_equal(v_out, torch.cat([vp2, v2], dim=1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
