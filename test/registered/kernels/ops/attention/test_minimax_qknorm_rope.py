"""Correctness for the fused MiniMax-M3 Gemma-RMSNorm + partial NeoX RoPE kernel.

Verifies the in-place fused kernel reproduces GemmaRMSNorm((1+w)) + partial NeoX
RoPE to the bf16 round-off floor, leaves V untouched, and matches sglang's RoPE
convention (cos|sin cache, neox pairs (i, i+rotary_dim/2)).
"""

import pytest
import torch

from sglang.kernels.ops.attention.minimax_qknorm_rope import (
    minimax_qknorm_rope,
    minimax_qknorm_rope_grouped,
    minimax_qknorm_rope_grouped_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

dev = "cuda"
HEAD_DIM, ROTARY_DIM, BASE, EPS = 128, 64, 5_000_000, 1e-6


def _build_cache(max_pos):
    inv_freq = 1.0 / (
        BASE
        ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float, device=dev) / ROTARY_DIM)
    )  # [32]
    t = torch.arange(max_pos, dtype=torch.float, device=dev)
    freqs = torch.outer(t, inv_freq)  # [max_pos, 32]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()  # [max_pos, 64]


def _ref(q, k, wq, wk, cache, positions, nq, nk):
    T = q.shape[0]

    def norm(x, w, nh):
        x = x.reshape(T, nh, HEAD_DIM).float()
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + EPS) * (1.0 + w.float())

    cs = cache.index_select(0, positions).float()
    cos, sin = cs[:, :32], cs[:, 32:]

    def rope(x):
        x1, x2 = x[..., :32], x[..., 32:64]
        o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        return torch.cat([o1, o2, x[..., 64:]], dim=-1)

    return rope(norm(q, wq, nq)).reshape(T, -1), rope(norm(k, wk, nk)).reshape(T, -1)


@pytest.mark.parametrize("nq,nk", [(8, 1), (64, 8), (16, 2)])
@pytest.mark.parametrize("T", [1, 7, 64, 1024, 4096])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_fused_qknorm_rope(nq, nk, T, pos_dtype):
    torch.manual_seed(T * 131 + nq)
    max_pos = 8192
    cache = _build_cache(max_pos)
    wq = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    q = torch.randn(T, nq * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T, nk * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T, nk * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    positions = torch.randint(0, max_pos, (T,), device=dev, dtype=pos_dtype)

    qr, kr = _ref(q, k, wq, wk, cache, positions.long(), nq, nk)

    qkv = torch.cat([q, k, v], dim=-1).contiguous()
    minimax_qknorm_rope(qkv, wq, wk, cache, positions, nq, nk, nk, EPS)
    qf, kf, vf = qkv.split([nq * HEAD_DIM, nk * HEAD_DIM, nk * HEAD_DIM], dim=-1)

    floor = (qr.bfloat16().float() - qr.float()).abs().max().item()
    assert (qf.float() - qr.float()).abs().max().item() <= 2 * floor + 1e-3
    assert (kf.float() - kr.float()).abs().max().item() <= 2 * floor + 1e-3
    assert (vf.float() - v.float()).abs().max().item() == 0.0  # V untouched


def test_index_branch_shapes():
    # idx_q (nq=num_idx_heads, nk=0) and idx_k (nq=1) in-place calls.
    torch.manual_seed(0)
    max_pos = 4096
    cache = _build_cache(max_pos)
    w = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    T, num_idx = 33, 4
    positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int64)
    for nq in (num_idx, 1):
        x = torch.randn(T, nq * HEAD_DIM, dtype=torch.bfloat16, device=dev)
        ref, _ = _ref(x, x[:, :HEAD_DIM], w, w, cache, positions, nq, 1)
        xc = x.clone()
        minimax_qknorm_rope(xc, w, w, cache, positions, nq, 0, 0, EPS)
        floor = (ref.bfloat16().float() - ref.float()).abs().max().item()
        assert (xc.float() - ref.float()).abs().max().item() <= 2 * floor + 1e-3


def _norm_rope_one(x_heads, w, cache, positions):
    # x_heads: [T, nh, HEAD_DIM] fp32; returns same shape, GemmaRMSNorm(1+w)+rope.
    var = x_heads.pow(2).mean(-1, keepdim=True)
    y = x_heads * torch.rsqrt(var + EPS) * (1.0 + w.float())
    cs = cache.index_select(0, positions).float()
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    x1, x2 = y[..., :32], y[..., 32:64]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2, y[..., 64:]], dim=-1)


@pytest.mark.parametrize("nq,nkv,niq", [(8, 1, 1), (8, 1, 4), (16, 2, 2)])
@pytest.mark.parametrize("idx_v", [0, 1])
@pytest.mark.parametrize("T", [1, 7, 64, 1024])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_combined_main_index_grouped(nq, nkv, niq, idx_v, T, pos_dtype):
    """Combined main(q,k,v) + index(idx_q,idx_k,[idx_v]) layout in one launch.

    Mirrors the fused qkv+index_qkv GEMM output: a uniform [total_heads, 128]
    grid where Q / K main heads and index-Q / index-K heads are normed+roped in
    one pass and the V / index-V heads are left untouched.
    """
    torch.manual_seed(T * 17 + nq * 3 + niq + idx_v)
    max_pos = 8192
    cache = _build_cache(max_pos)
    positions = torch.randint(0, max_pos, (T,), device=dev, dtype=pos_dtype)

    # head layout: q(nq) k(nkv) v(nkv) idx_q(niq) idx_k(1) [idx_v(1)]
    off_q = 0
    off_k = nq
    off_v = nq + nkv
    off_iq = nq + 2 * nkv
    off_ik = off_iq + niq
    total_heads = off_ik + 1 + idx_v

    wq = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    wiq = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)
    wik = (torch.randn(HEAD_DIM, device=dev) * 0.1).to(torch.bfloat16)

    x = torch.randn(T, total_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ref = x.float().clone()
    ref[:, off_q:off_k] = _norm_rope_one(
        ref[:, off_q:off_k], wq, cache, positions.long()
    )
    ref[:, off_k:off_v] = _norm_rope_one(
        ref[:, off_k:off_v], wk, cache, positions.long()
    )
    ref[:, off_iq:off_ik] = _norm_rope_one(
        ref[:, off_iq:off_ik], wiq, cache, positions.long()
    )
    ref[:, off_ik : off_ik + 1] = _norm_rope_one(
        ref[:, off_ik : off_ik + 1], wik, cache, positions.long()
    )

    qkv = x.reshape(T, total_heads * HEAD_DIM).contiguous()
    minimax_qknorm_rope_grouped(
        qkv,
        [(wq, off_q, nq), (wk, off_k, nkv), (wiq, off_iq, niq), (wik, off_ik, 1)],
        cache,
        positions,
        EPS,
    )
    out = qkv.reshape(T, total_heads, HEAD_DIM)

    floor = (ref.bfloat16().float() - ref).abs().max().item()
    assert (out.float() - ref).abs().max().item() <= 2 * floor + 1e-3
    # V and index-V heads untouched (bit-exact).
    assert (
        out[:, off_v:off_iq].float() - x[:, off_v:off_iq].float()
    ).abs().max() == 0.0
    if idx_v:
        assert (
            out[:, off_ik + 1 :].float() - x[:, off_ik + 1 :].float()
        ).abs().max() == 0.0


# Input scale cancels in the norm, so the Q gain is what reaches the E4M3 edges:
# 1000 pushes most values past 448, -1 + 2^-8 puts them in the subnormal range.
@pytest.mark.parametrize(
    "nq,nk,niq,q_gain",
    [(8, 1, 4, 1000.0), (8, 1, 4, -1.0 + 2**-8), (64, 8, 16, None)],
)
def test_fp8_epilogue_matches_bf16_cast(nq, nk, niq, q_gain):
    """The FP8 Q outputs equal the BF16 result cast by `Tensor.to`, bit for bit,
    and the BF16 write-back is unchanged by the epilogue."""
    torch.manual_seed(11)
    T, max_pos = 64, 8192
    cache = _build_cache(max_pos)
    positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int32)
    weights = [torch.randn(HEAD_DIM, device=dev).bfloat16() * 0.1 for _ in range(4)]
    if q_gain is not None:
        # The kernel applies 1 + weight, so this sets the gain of both Q groups.
        weights[0] = weights[2] = torch.full_like(weights[0], q_gain)
    off_iq = nq + 2 * nk
    groups = list(zip(weights, (0, nq, off_iq, off_iq + niq), (nq, nk, niq, 1)))
    base = torch.randn(T, (off_iq + niq + 1) * HEAD_DIM, device=dev).bfloat16()

    bf16_only = base.clone()
    minimax_qknorm_rope_grouped(bf16_only, groups, cache, positions, EPS)
    with_fp8 = base.clone()
    fp8_q = torch.empty(T, nq, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev)
    fp8_idx_q = torch.empty(T, niq, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev)
    minimax_qknorm_rope_grouped_fp8(
        with_fp8,
        groups,
        cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=0,
        fp8_b=fp8_idx_q,
        group_b=2,
    )

    assert torch.equal(bf16_only, with_fp8)
    rows = with_fp8.reshape(T, -1, HEAD_DIM)
    q_abs = rows[:, :nq].float().abs()
    if q_gain is not None and q_gain > 0:
        assert (q_abs > 448).any(), "the case never leaves the E4M3 range"
    elif q_gain is not None:
        assert ((q_abs > 0) & (q_abs < 2**-6)).any(), "no subnormal E4M3 values"
    for out, expected in (
        (fp8_q, rows[:, :nq]),
        (fp8_idx_q, rows[:, off_iq : off_iq + niq]),
    ):
        expected = expected.to(torch.float8_e4m3fn)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
