"""Cake MiniMax-H3 BF16 pre-attention through sglang.kernels.

Checks three things for the Cake adapter distributed by FlashInfer: the
registry resolves the explicit FlashInfer backend; the facade result is
bitwise identical to calling FlashInfer directly; and the fused result matches
a pure-torch reference (RMSNorm -> indexed AdaLN -> BF16 QKV -> per-head Q/K
RMSNorm -> partial split-half NeoX RoPE -> destination-major pack) within BF16
tolerance. Skips (with the reason) when the installed FlashInfer lacks the
Cake module or the GPU is not sm_100a / sm_103a.
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
ROPE_DIM, ADALN_ROWS, EPS = 96, 9, 1.0e-5


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


def modulated(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index):
    norm = F.rms_norm(x, (HIDDEN,), x_norm_weight, eps=EPS).to(torch.bfloat16)
    index = adaln_index.long()
    valid = (index >= 0) & (index < ADALN_ROWS)
    safe = index.clamp(0, ADALN_ROWS - 1)
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
    )
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qkv = (a.float() @ case["qkv_weight"].float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    m = case["x"].shape[0]
    grouped = qkv.view(m, NUM_HEADS, KINDS, HEAD_DIM)
    q = F.rms_norm(grouped[:, :, 0], (HEAD_DIM,), case["q_norm_weight"], eps=EPS)
    k = F.rms_norm(grouped[:, :, 1], (HEAD_DIM,), case["k_norm_weight"], eps=EPS)
    q = apply_rope(q.to(torch.bfloat16), case["rope_cos_sin"])
    k = apply_rope(k.to(torch.bfloat16), case["rope_cos_sin"])
    fused = torch.stack((q, k, grouped[:, :, 2]), dim=2)
    p = case["ulysses_degree"]
    return (
        fused.view(m, p, NUM_HEADS // p, KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def make_case(m, p, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    rows = torch.arange(m, device=device)
    adaln_index = torch.div(rows * ADALN_ROWS, m, rounding_mode="floor").clamp_max(8)
    adaln_index = adaln_index.to(torch.int32)
    adaln_index[m // 2] = 9  # device-guarded invalid row -> zero output row
    return {
        "x": normal((m, HIDDEN), 0.5),
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_index": adaln_index,
        "qkv_weight": normal((NUM_HEADS * KINDS * HEAD_DIM, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": make_rope_cache(m, device),
        "ulysses_degree": p,
    }


@pytest.mark.parametrize("m,p", [(129, 8), (130, 2)])
def test_matches_flashinfer_and_reference(m, p):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = make_case(m, p, device, seed=4532 + m + p)
    out = torch.empty(
        (p, m, NUM_HEADS // p, KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    assert cake.supports_minimax_h3_bf16_pre_attention(**case, out=out, eps=EPS), (
        "admission must accept an in-contract case"
    )
    result = cake_minimax_h3_bf16_pre_attention(**case, out=out, eps=EPS)
    assert result is out

    from flashinfer.diffusion_ops.minimax_h3 import (
        minimax_h3_bf16_pre_attention as fi_direct,
    )

    out_fi = torch.empty_like(out)
    fi_direct(**case, out=out_fi, eps=EPS)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    expected = reference(case)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    # The invalid AdaLN row is zero on every destination.
    assert torch.count_nonzero(out[:, m // 2]).item() == 0


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = make_case(8, 8, device, seed=1)
    out = torch.empty((8, 8, 7, KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    assert cake.supports_minimax_h3_bf16_pre_attention(**case, out=out)
    assert not cake.supports_minimax_h3_bf16_pre_attention(**case, out=out, eps=1e-6)
    bad = dict(case, ulysses_degree=3)
    assert not cake.supports_minimax_h3_bf16_pre_attention(**bad, out=out)
    bad = dict(case, x=case["x"].float())
    assert not cake.supports_minimax_h3_bf16_pre_attention(**bad, out=out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
