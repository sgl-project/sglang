"""Cake MiniMax-H3 BF16 pre-attention through sglang.kernels.

Checks three things for the Cake adapter distributed by FlashInfer: the
registry resolves the explicit FlashInfer backend; the facade result is
bitwise identical to calling FlashInfer directly; and the fused result matches
a pure-torch reference (RMSNorm -> indexed AdaLN -> BF16 QKV -> per-head Q/K
RMSNorm -> partial split-half NeoX RoPE -> destination-major pack) within BF16
tolerance. The operands are the diffusion engine's own: AdaLN tables are
strided ``[rows, 5376]`` column chunks of a ``[rows, 6 * 5376]`` projection
with ``rows`` in {3, 6, 9, 12}, the row index is int64, RoPE is the request
``(cos_sin_cache [S, 96], positions int64 [M])`` pair with non-identity
positions, and ``norm1`` / Q-K-norm epsilons are passed separately. Skips
(with the reason) when the installed FlashInfer lacks the Cake module or the
GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_pre_attention as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_bf16_pre_attention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=400, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "diffusion.minimax_h3_bf16_pre_attention"
HIDDEN, NUM_HEADS, HEAD_DIM, KINDS = 5376, 56, 128, 3
ROPE_DIM, EPS = 96, 1.0e-5


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_pre_attention:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake.FI_MODULE, cake.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.diffusion_ops.minimax_h3")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake MiniMax-H3 pre-attention needs sm_100a/103a, device is {cc}")


def make_rope_cache(m, device):
    rows = torch.arange(m, dtype=torch.float32, device=device)
    axes = (
        torch.div(rows, 4096, rounding_mode="floor"),
        torch.div(rows, 64, rounding_mode="floor").remainder(64),
        rows.remainder(64),
    )
    inv_freq = 10000.0 ** (-torch.arange(16, dtype=torch.float32, device=device) / 16)
    phase = torch.cat([a[:, None] * inv_freq[None, :] for a in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def apply_rope(x, rope_cos_sin):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :48], rope_cos_sin[:, :48]), -1).float()[:, None]
    sin = torch.cat((rope_cos_sin[:, 48:], rope_cos_sin[:, 48:]), -1).float()[:, None]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    return torch.cat(((rotary * cos + rotated_half * sin).to(torch.bfloat16), tail), -1)


def modulated(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps):
    rows = adaln_scale.shape[0]
    norm = F.rms_norm(x, (HIDDEN,), x_norm_weight, eps=eps).to(torch.bfloat16)
    index = adaln_index.long()
    valid = (index >= 0) & (index < rows)
    safe = index.clamp(0, rows - 1)
    a = torch.addcmul(
        adaln_shift.index_select(0, safe),
        norm,
        (adaln_scale.index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def reference(case):
    a = modulated(
        case["x"],
        case["x_norm_weight"],
        case["adaln_scale"],
        case["adaln_shift"],
        case["adaln_index"],
        case["eps"],
    )
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qkv = (a.float() @ case["qkv_weight"].float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    m = case["x"].shape[0]
    grouped = qkv.view(m, NUM_HEADS, KINDS, HEAD_DIM)
    qk_eps = case["eps"] if case["qk_eps"] is None else case["qk_eps"]
    q = F.rms_norm(grouped[:, :, 0], (HEAD_DIM,), case["q_norm_weight"], eps=qk_eps)
    k = F.rms_norm(grouped[:, :, 1], (HEAD_DIM,), case["k_norm_weight"], eps=qk_eps)
    positions = case["rope_positions"]
    rope = (
        case["rope_cos_sin"][:m]
        if positions is None
        else case["rope_cos_sin"][positions]
    )
    q = apply_rope(q.to(torch.bfloat16), rope)
    k = apply_rope(k.to(torch.bfloat16), rope)
    fused = torch.stack((q, k, grouped[:, :, 2]), dim=2)
    p = case["ulysses_degree"]
    return (
        fused.view(m, p, NUM_HEADS // p, KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def make_case(m, p, device, seed, *, rows=9, eps=EPS, qk_eps=None, positions=True):
    """Engine operands: strided AdaLN chunks, int64 index, (cache, positions)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    r = torch.arange(m, device=device)
    adaln_index = torch.div(r * rows, m, rounding_mode="floor").clamp_max(rows - 1)
    adaln_index[m // 2] = rows  # device-guarded invalid row -> zero output row
    # The [rows, 6 * 5376] modulation projection; chunks 0 / 1 are shift / scale.
    proj = uniform((rows, 6 * HIDDEN), -0.05, 0.05)
    shift, scale = proj[:, :HIDDEN], proj[:, HIDDEN : 2 * HIDDEN]
    assert not scale.is_contiguous() and scale.stride(0) == 6 * HIDDEN
    cache_rows = m + 17 if positions else m
    rope_positions = ((r * 7 + 3) % cache_rows).to(torch.int64) if positions else None
    return {
        "x": normal((m, HIDDEN), 0.5),
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": scale,
        "adaln_shift": shift,
        "adaln_index": adaln_index,
        "qkv_weight": normal((NUM_HEADS * KINDS * HEAD_DIM, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": make_rope_cache(cache_rows, device),
        "ulysses_degree": p,
        "eps": eps,
        "qk_eps": qk_eps,
        "rope_positions": rope_positions,
    }


@pytest.mark.parametrize(
    "m,p,rows,eps,qk_eps,positions",
    [
        (129, 8, 9, EPS, None, False),  # identity RoPE rows, one eps
        (130, 2, 3, EPS, 1.0e-6, True),  # t2va: 3 AdaLN rows, split epsilons
        (131, 1, 6, 2.0e-5, 1.0e-6, True),
        (128, 4, 12, EPS, EPS, True),
    ],
)
def test_matches_flashinfer_and_reference(m, p, rows, eps, qk_eps, positions):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = make_case(
        m,
        p,
        device,
        seed=4532 + m + p,
        rows=rows,
        eps=eps,
        qk_eps=qk_eps,
        positions=positions,
    )
    out = torch.empty(
        (p, m, NUM_HEADS // p, KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    assert cake.supports_minimax_h3_bf16_pre_attention(**case, out=out), (
        "admission must accept an in-contract case"
    )
    result = cake_minimax_h3_bf16_pre_attention(**case, out=out)
    assert result is out

    from flashinfer.diffusion_ops.minimax_h3 import (
        minimax_h3_bf16_pre_attention as fi_direct,
    )

    out_fi = torch.empty_like(out)
    fi_direct(**case, out=out_fi)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    expected = reference(case)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    # The invalid AdaLN row is zero on every destination.
    assert torch.count_nonzero(out[:, m // 2]).item() == 0


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = make_case(8, 8, device, seed=1, rows=6)
    out = torch.empty((8, 8, 7, KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    ok = cake.supports_minimax_h3_bf16_pre_attention

    def rejects(**overrides):
        return not ok(**dict(case, **overrides), out=out)

    assert ok(**case, out=out)
    assert ok(**dict(case, eps=1e-6, qk_eps=None), out=out)  # eps is free
    assert ok(**dict(case, rope_positions=None), out=out)  # identity rows
    assert ok(**dict(case, adaln_scale=case["adaln_scale"].contiguous()), out=out)
    assert rejects(ulysses_degree=3)
    assert rejects(x=case["x"].float())
    # Index must be int64 [M].
    assert rejects(adaln_index=case["adaln_index"].to(torch.int32))
    assert rejects(adaln_index=case["adaln_index"][:-1])
    # Tables: same row count, unit last stride, 16-byte row pitch.
    assert rejects(adaln_shift=case["adaln_shift"][:-1])
    assert rejects(adaln_scale=case["adaln_scale"].t().contiguous().t())
    misaligned = torch.zeros((6, HIDDEN + 4), dtype=torch.bfloat16, device=device)
    assert misaligned[:, :HIDDEN].stride(0) % 8 != 0
    assert rejects(adaln_scale=misaligned[:, :HIDDEN])
    # RoPE pair: int64 positions [M] into a contiguous [S, 96] cache; identity
    # needs S >= M.
    assert rejects(rope_positions=case["rope_positions"].to(torch.int32))
    assert rejects(rope_positions=case["rope_positions"][:-1])
    assert rejects(rope_cos_sin=case["rope_cos_sin"].t().contiguous().t())
    assert rejects(rope_cos_sin=case["rope_cos_sin"][:4], rope_positions=None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
