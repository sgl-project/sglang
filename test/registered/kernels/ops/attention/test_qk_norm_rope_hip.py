"""ROCm fused QK-norm + RoPE (``apply_qk_norm_rope``) against the unfused
``apply_qk_norm`` + ``rotary_emb`` path it replaces in Qwen3 prefill.

q/k are strided views of a packed qkv, as ``qkv.split`` gives them. The fused
kernel reads the fp32 cos/sin cache while the unfused HIP RoPE rounds it to the
activation dtype first, so the two differ by 1-2 ulp; the fused result must
stay at least as close to an fp64 reference.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.diffusion.rope.qknorm_rope_jit as qknorm_rope_jit
import sglang.srt.models.utils as model_utils
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.utils import (
    apply_qk_norm,
    apply_qk_norm_rope,
    can_use_jit_qk_norm_rope,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")

DEVICE = "cuda"
BASE = 1e6
MAX_POSITION = 40960
# Same tolerance as the XPU and diffusion qknorm_rope tests.
ATOL = 8e-2
RTOL = 1e-2


def _exec_config(tc_compiler="eager", rl_on_policy_target=None):
    return SimpleNamespace(
        deterministic=SimpleNamespace(rl_on_policy_target=rl_on_policy_target),
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(tc_compiler=tc_compiler)
            )
        ),
    )


@pytest.fixture(autouse=True)
def exec_config(monkeypatch):
    monkeypatch.setattr(model_utils, "get_exec", lambda: _exec_config())


def _make_modules(
    head_dim, rotary_dim, is_neox, dtype, eps=1e-6, k_eps=None, rope_scaling=None
):
    rotary_emb = get_rope(
        head_dim,
        rotary_dim=rotary_dim,
        max_position=MAX_POSITION,
        base=BASE,
        is_neox_style=is_neox,
        rope_scaling=rope_scaling,
        dtype=dtype,
    ).to(DEVICE)
    q_norm = RMSNorm(head_dim, eps=eps).to(DEVICE, dtype)
    k_norm = RMSNorm(head_dim, eps=eps if k_eps is None else k_eps).to(DEVICE, dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.3)
    k_norm.weight.data.normal_(mean=1.0, std=0.3)
    return q_norm, k_norm, rotary_emb


def _make_inputs(num_tokens, num_q_heads, num_kv_heads, head_dim, dtype, pos_dtype):
    qkv = torch.randn(
        num_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim, device=DEVICE
    ).to(dtype)
    # Non-contiguous, not starting at 0.
    positions = torch.randint(
        1, MAX_POSITION, (num_tokens,), device=DEVICE, dtype=pos_dtype
    )
    sizes = [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim]
    return qkv, positions, sizes


def _unfused(qkv, positions, sizes, q_norm, k_norm, rotary_emb, head_dim):
    out = qkv.clone()
    q, k, _ = out.split(sizes, dim=-1)
    q, k = apply_qk_norm(q=q, k=k, q_norm=q_norm, k_norm=k_norm, head_dim=head_dim)
    q, k = rotary_emb(positions, q, k)
    return q, k


def _fp64_reference(x, weight, positions, eps, head_dim, rotary_dim, is_neox):
    x = x.double().view(x.size(0), -1, head_dim)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.double()
    inv_freq = BASE ** (
        -torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float64)
        / rotary_dim
    )
    freqs = positions.double()[:, None] * inv_freq[None, :]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    rot = x[..., :rotary_dim]
    if is_neox:
        x1, x2 = rot.chunk(2, dim=-1)
        rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rot = rot.flatten(-2)
    return torch.cat((rot, x[..., rotary_dim:]), dim=-1).flatten(1)


def _count_fused_calls(monkeypatch):
    calls = []
    fused = qknorm_rope_jit.fused_inplace_qknorm_rope

    def counted(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr(qknorm_rope_jit, "fused_inplace_qknorm_rope", counted)
    return calls


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 257, 4096])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    "head_dim,rotary_dim", [(128, 128), (128, 64), (64, 64), (256, 256)]
)
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 8), (8, 1), (16, 16)])
def test_matches_unfused_path(
    monkeypatch, num_q_heads, num_kv_heads, head_dim, rotary_dim, is_neox, num_tokens
):
    dtype = torch.bfloat16
    eps = 1e-6 if is_neox else 1e-5
    torch.manual_seed(num_tokens + head_dim + rotary_dim + num_q_heads)
    q_norm, k_norm, rotary_emb = _make_modules(
        head_dim, rotary_dim, is_neox, dtype, eps=eps
    )
    assert can_use_jit_qk_norm_rope(q_norm, k_norm, rotary_emb, head_dim, dtype)

    # The unfused HIP RoPE only takes int64 positions.
    qkv, positions, sizes = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, dtype, torch.int64
    )

    def fused(positions):
        out = qkv.clone()
        q_in, k_in, v = out.split(sizes, dim=-1)
        q, k = apply_qk_norm_rope(
            q=q_in,
            k=k_in,
            q_norm=q_norm,
            k_norm=k_norm,
            rotary_emb=rotary_emb,
            positions=positions,
            head_dim=head_dim,
            use_jit=True,
        )
        assert q.data_ptr() == q_in.data_ptr() and k.data_ptr() == k_in.data_ptr()
        assert torch.equal(v, qkv[:, sizes[0] + sizes[1] :])
        return out, q, k

    calls = _count_fused_calls(monkeypatch)
    out, q, k = fused(positions)
    if num_tokens == 0:
        # The unfused HIP RoPE cannot launch on zero tokens.
        assert not calls
        return
    assert len(calls) == 1
    assert torch.equal(fused(positions.to(torch.int32))[0], out)

    q_ref, k_ref = _unfused(qkv, positions, sizes, q_norm, k_norm, rotary_emb, head_dim)
    torch.testing.assert_close(q, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k, k_ref, atol=ATOL, rtol=RTOL)

    if num_tokens >= 257:
        q_in, k_in, _ = qkv.split(sizes, dim=-1)
        for x, w, got, ref in (
            (q_in, q_norm.weight, q, q_ref),
            (k_in, k_norm.weight, k, k_ref),
        ):
            exact = _fp64_reference(x, w, positions, eps, head_dim, rotary_dim, is_neox)
            fused_err = (got.double() - exact).abs().max()
            ref_err = (ref.double() - exact).abs().max()
            assert fused_err <= ref_err, (fused_err.item(), ref_err.item())


def test_fp16(monkeypatch):
    dtype, head_dim, num_tokens = torch.float16, 128, 257
    q_norm, k_norm, rotary_emb = _make_modules(head_dim, head_dim, True, dtype)
    assert can_use_jit_qk_norm_rope(q_norm, k_norm, rotary_emb, head_dim, dtype)
    qkv, positions, sizes = _make_inputs(
        num_tokens, 32, 8, head_dim, dtype, torch.int64
    )
    q_ref, k_ref = _unfused(qkv, positions, sizes, q_norm, k_norm, rotary_emb, head_dim)
    calls = _count_fused_calls(monkeypatch)
    q, k, _ = qkv.clone().split(sizes, dim=-1)
    q, k = apply_qk_norm_rope(
        q, k, q_norm, k_norm, rotary_emb, positions, head_dim, use_jit=True
    )
    assert len(calls) == 1
    torch.testing.assert_close(q, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k, k_ref, atol=ATOL, rtol=RTOL)


def test_kernel_gate_rejects_unvalidated_variants():
    can_use = qknorm_rope_jit.can_use_fused_inplace_qknorm_rope
    dtype = torch.bfloat16
    assert not can_use(128, 128, True, dtype, round_norm_before_rope=False)
    assert not can_use(128, 128, True, dtype, round_norm_before_rope=True, pack_kv=True)
    assert can_use(128, 128, False, dtype, round_norm_before_rope=False)


@pytest.mark.parametrize(
    "case",
    ["yarn", "eps_mismatch", "fp32_norm_weight", "deterministic", "inductor"],
)
def test_gate_falls_back(monkeypatch, case):
    dtype, head_dim, num_tokens = torch.bfloat16, 128, 257
    kwargs = {}
    if case == "yarn":
        kwargs["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": MAX_POSITION // 4,
        }
    elif case == "eps_mismatch":
        kwargs["k_eps"] = 1e-5
    q_norm, k_norm, rotary_emb = _make_modules(
        head_dim, head_dim, True, dtype, **kwargs
    )
    if case == "fp32_norm_weight":
        q_norm.float()
        k_norm.float()
    elif case == "inductor":
        monkeypatch.setattr(
            model_utils, "get_exec", lambda: _exec_config(tc_compiler="inductor")
        )
    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(case == "deterministic"):
        use_jit = can_use_jit_qk_norm_rope(q_norm, k_norm, rotary_emb, head_dim, dtype)
        assert not use_jit

        qkv, positions, sizes = _make_inputs(
            num_tokens, 32, 8, head_dim, dtype, torch.int64
        )
        q_ref, k_ref = _unfused(
            qkv, positions, sizes, q_norm, k_norm, rotary_emb, head_dim
        )
        calls = _count_fused_calls(monkeypatch)
        q, k, _ = qkv.clone().split(sizes, dim=-1)
        q, k = apply_qk_norm_rope(
            q, k, q_norm, k_norm, rotary_emb, positions, head_dim, use_jit=use_jit
        )
    assert not calls
    assert torch.equal(q, q_ref) and torch.equal(k, k_ref)


def test_gate_rejects_mrope():
    head_dim = 128
    q_norm, k_norm, rotary_emb = _make_modules(
        head_dim,
        head_dim,
        True,
        torch.bfloat16,
        rope_scaling={"rope_type": "default", "mrope_section": [16, 24, 24]},
    )
    assert type(rotary_emb).__name__ == "MRotaryEmbedding"
    assert not can_use_jit_qk_norm_rope(
        q_norm, k_norm, rotary_emb, head_dim, torch.bfloat16
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
