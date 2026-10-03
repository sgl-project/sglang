# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_qknorm_complex_rope_cuda,
    fused_complex_rope,
    qknorm_complex_rope_cuda,
    qknorm_complex_rope_pack_,
)
from sglang.multimodal_gen.runtime.models.dits import qwen_image21
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, rope):
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def inputs(shape, dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    # a contiguous slice retains the nonzero cache offset used by SP ranks
    angles = torch.randn(shape[1] + 5, shape[-1] // 2, device="cuda") * 20
    rope = torch.polar(torch.ones_like(angles), angles)[5:]
    return x, rope


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 32), (2, 17, 3, 64), (1, 257, 16, 128), (1, 4096, 32, 128)]
)
def test_complex_rope_matches_complex_multiplication(dtype, shape):
    x, rope = inputs(shape, dtype)
    actual = fused_complex_rope(x, rope)
    torch.testing.assert_close(actual, reference(x, rope), atol=0, rtol=0)


def test_complex_rope_layout_guards():
    x, rope = inputs((2, 17, 3, 64), torch.bfloat16)
    with pytest.raises(RuntimeError):
        fused_complex_rope(x.cpu(), rope.cpu())
    with pytest.raises(RuntimeError):
        fused_complex_rope(x.double(), rope)
    with pytest.raises(RuntimeError):
        fused_complex_rope(x, rope.to(torch.complex128))
    with pytest.raises(RuntimeError):
        fused_complex_rope(x[:, ::2], rope[::2])
    with pytest.raises(RuntimeError):
        fused_complex_rope(x, rope[:-1])
    with pytest.raises(RuntimeError):
        fused_complex_rope(x[:, :0], rope[:0])


def test_complex_rope_compile_and_graph_replay():
    x, rope = inputs((1, 257, 8, 128), torch.bfloat16)
    compiled = torch.compile(fused_complex_rope, fullgraph=True)
    torch.testing.assert_close(compiled(x, rope), reference(x, rope), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_complex_rope(x, rope)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(out, reference(x, rope), atol=0, rtol=0)


def test_qwen21_rope_first_sight_verification(monkeypatch):
    x, rope = inputs((1, 257, 8, 128), torch.bfloat16)
    gate = BitExactFusionGate("test complex RoPE")
    monkeypatch.setattr(qwen_image21, "_ROPE_FUSION", gate)
    torch.testing.assert_close(
        qwen_image21.apply_rope(x, rope), reference(x, rope), atol=0, rtol=0
    )
    assert gate.verified and not gate.disabled

    gate = BitExactFusionGate("test mismatched RoPE")
    monkeypatch.setattr(qwen_image21, "_ROPE_FUSION", gate)
    monkeypatch.setattr(
        qwen_image21, "fused_complex_rope", lambda x, rope: torch.zeros_like(x)
    )
    torch.testing.assert_close(
        qwen_image21.apply_rope(x, rope), reference(x, rope), atol=0, rtol=0
    )
    assert gate.disabled and not gate.verified


# -------------------------------------------------------------------------
# Qwen-Image 2.1 -- CUDA Q/K RMSNorm + complex RoPE, and the packed K/V variant
# -------------------------------------------------------------------------


EPS = 1e-6

HEAD_DIM = 128


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


@torch.no_grad()
@pytest.mark.parametrize(
    "shape", [(1, 4096, 32, 128), (1, 18, 32, 128), (2, 17, 4, 128), (1, 1, 1, 128)]
)
@pytest.mark.parametrize("amplitude", [1e-3, 1.0, 40.0])
def test_norm_rope_matches_eager(shape, amplitude):
    torch.manual_seed(0)
    norm = make_norm()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude
    rope = make_rope(shape[1])
    assert can_use_qknorm_complex_rope_cuda(x.dtype, x.shape[-1])
    out = qknorm_complex_rope_cuda(x, norm.weight, rope, EPS)
    assert_bits_equal(out, eager_norm_rope(x, norm, rope))


@torch.no_grad()
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


@torch.no_grad()
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
    assert can_use_qknorm_complex_rope_cuda(q_work.dtype, q_work.shape[-1])
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


@torch.no_grad()
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
    assert can_use_qknorm_complex_rope_cuda(q_view.dtype, q_view.shape[-1])
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


@torch.no_grad()
def test_native_launchers_reject_invalid_inputs():
    norm = make_norm()
    x = torch.randn(1, 17, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rope = make_rope(17)
    assert can_use_qknorm_complex_rope_cuda(x.dtype, x.shape[-1])
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x.cpu(), norm.weight.cpu(), rope.cpu(), EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x.half(), norm.weight.half(), rope, EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x[..., :64], norm.weight[:64], rope[:, :32], EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x.transpose(1, 2), norm.weight, rope, EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x, norm.weight, rope[:-1], EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x, norm.weight, rope.to(torch.complex128), EPS)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x, norm.weight.float(), rope, EPS)
    k_out = torch.empty(1, 17 + 3, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kp = torch.randn(1, 3, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    assert can_use_qknorm_complex_rope_cuda(x.dtype, x.shape[-1])
    # wrong prefix + seq length, batch mismatch, non-contiguous prefix, short K source
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_pack_(
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
            EPS,
        )
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_pack_(
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
            EPS,
        )
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_pack_(
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
            EPS,
        )
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_pack_(
            x,
            k_out,
            k_out.clone(),
            norm.weight,
            norm.weight,
            rope,
            kp,
            kp,
            x[:, :-1],
            None,
            EPS,
        )


@torch.no_grad()
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


@torch.no_grad()
@pytest.mark.parametrize(
    "cuda_disabled,triton_disabled", [(False, False), (True, False), (True, True)]
)
def test_model_qknorm_rope_accepts_packed_projection_views(
    monkeypatch, cuda_disabled, triton_disabled
):
    packed = torch.randn(2, 17, 3, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    x = packed.unbind(2)[0]
    assert not x.is_contiguous()
    norm, rope = make_norm(), make_rope(17)
    for name, disabled in (
        ("_QK_ROPE_CUDA_FUSION", cuda_disabled),
        ("_QK_ROPE_FUSION", triton_disabled),
        ("_ROPE_FUSION", False),
    ):
        gate = BitExactFusionGate(name)
        gate.disabled = disabled
        monkeypatch.setattr(qwen_image21, name, gate)
    expected = reference(norm(x), rope)
    actual = qwen_image21.apply_qk_norm_rope(x, norm, rope)
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


@torch.no_grad()
def test_model_pack_verifies_strided_projection_views(monkeypatch):
    from types import SimpleNamespace

    batch, seq, prefix, heads = 1, 17, 3, 4
    packed = torch.randn(
        batch, prefix + seq, 3, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q = packed[:, prefix:, 0]
    k_out, v_out = packed[:, :, 1], packed[:, :, 2]
    k, v = k_out[:, prefix:], v_out[:, prefix:]
    kp, vp = torch.randn_like(q[:, :prefix]), torch.randn_like(q[:, :prefix])
    norm_q, norm_k, rope = make_norm(), make_norm(), make_rope(seq)
    attention = SimpleNamespace(head_dim=HEAD_DIM, norm_q=norm_q, norm_k=norm_k)
    expected = (
        reference(norm_q(q), rope),
        torch.cat((kp, reference(norm_k(k), rope)), 1),
        torch.cat((vp, v), 1),
    )
    gate = BitExactFusionGate("test packed projection verification")
    monkeypatch.setattr(qwen_image21, "_KV_PACK_CUDA_FUSION", gate)
    actual = qwen_image21.QwenImage21Attention._pack_kv_cuda(
        attention, q, k, v, rope, kp, vp, k_out, v_out
    )
    assert gate.verified
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


@torch.no_grad()
def test_native_launcher_preserves_rope_rank_and_row_alignment():
    norm, rope = make_norm(), make_rope(17)
    x = torch.randn(1, 17, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(x, norm.weight, rope.view(17, 1, 64), EPS)
    storage = torch.empty(1, 17, 4 * HEAD_DIM + 1, device=x.device, dtype=x.dtype)
    unaligned_rows = storage[..., :-1].view_as(x)
    with pytest.raises(RuntimeError):
        qknorm_complex_rope_cuda(unaligned_rows, norm.weight, rope, EPS)


@torch.no_grad()
def test_compiled_native_norm_rope_preserves_output_layout():
    # Dense transposes differ from packed column views: empty_like would retain
    # their strides, but the out-of-place kernel writes contiguous output.
    x = torch.randn(17, 2, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(
        0, 1
    )
    norm, rope = make_norm(), make_rope(17)
    compiled = torch.compile(qknorm_complex_rope_cuda, fullgraph=True)
    actual = compiled(x, norm.weight, rope, EPS)
    assert torch.equal(actual, reference(norm(x), rope))
    assert actual.is_contiguous()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
