from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=300, suite="nightly-kernel-1-gpu", nightly=True)


DTYPE_FP8 = torch.float8_e4m3fn
D_V = 512
H_Q = 64
H_KV = 1
TOPK = 128
S_KV = 256


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _make_fp8_tensor(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tensor = torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32)
    return (tensor * 0.05).to(DTYPE_FP8)


def _make_case(
    d_qk: int,
    with_sink: bool,
    s_q: int = 2,
    topk: int = TOPK,
    s_kv: int = S_KV,
):
    q = _make_fp8_tensor((s_q, H_Q, d_qk), seed=1000 + d_qk + s_q * 13 + topk)
    kv = torch.zeros((s_kv + 1, H_KV, d_qk), dtype=DTYPE_FP8, device="cuda")
    kv[:s_kv] = _make_fp8_tensor((s_kv, H_KV, d_qk), seed=2000 + d_qk + s_kv)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(3000 + d_qk + s_q * 17 + topk)
    indices = torch.randint(
        0,
        s_kv,
        (s_q, H_KV, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )

    q_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    kv_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(d_qk)

    if not with_sink:
        return q, kv, indices, sm_scale, q_scale, kv_scale, None, None

    attn_sink = torch.linspace(-0.05, 0.05, H_Q, dtype=torch.float32, device="cuda")
    # Vary topk_length per query row to exercise the partial-topk path.
    lengths = [topk if i % 2 == 0 else max(topk - 32, topk // 2) for i in range(s_q)]
    topk_length = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length


def _torch_sparse_attention_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
) -> torch.Tensor:
    topk = indices.shape[-1]
    q_f32 = q.float() * q_scale.item()
    kv_f32 = kv.float() * kv_scale.item()
    out = torch.empty(
        (q.shape[0], q.shape[1], D_V), dtype=torch.float32, device=q.device
    )

    for q_idx in range(q.shape[0]):
        valid_topk = topk if topk_length is None else int(topk_length[q_idx].item())
        token_ids = indices[q_idx, 0, :valid_topk].to(torch.long)
        keys = kv_f32[token_ids, 0, :]
        values = kv_f32[token_ids, 0, :D_V]
        scores = torch.matmul(q_f32[q_idx], keys.transpose(0, 1)) * sm_scale
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        denom = exp_scores.sum(dim=-1, keepdim=True)
        if attn_sink is not None:
            denom = denom + torch.exp(attn_sink[:, None] - score_max)
        out[q_idx] = torch.matmul(exp_scores, values) / denom

    return out


def _run_and_check(d_qk, with_sink, s_q=2, topk=TOPK, s_kv=S_KV):
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk, with_sink, s_q=s_q, topk=topk, s_kv=s_kv
    )

    out, max_logits, lse = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    ref = _torch_sparse_attention_ref(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    assert out.shape == (q.shape[0], H_Q, D_V)
    assert out.dtype == torch.bfloat16
    assert max_logits.shape == (q.shape[0], H_Q)
    assert lse.shape == (q.shape[0], H_Q)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("d_qk,with_sink", [(512, False), (576, True)])
def test_sparse_mla_q8kv8_prefill_matches_reference(d_qk: int, with_sink: bool):
    _run_and_check(d_qk, with_sink)


# Corner cases: minimal s_q, larger s_q, larger topk/s_kv, and crossed
# (d_qk=512 with sink, d_qk=576 without sink) configurations. The kernel
# requires topk to be a multiple of 128, so 128 is the minimum supported.
@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize(
    "d_qk,with_sink,s_q,topk,s_kv",
    [
        (576, True, 1, TOPK, S_KV),
        (576, True, 8, TOPK, S_KV),
        (576, True, 2, 256, 512),
        (512, True, 2, TOPK, S_KV),
        (576, False, 2, TOPK, S_KV),
    ],
)
def test_sparse_mla_q8kv8_prefill_corner_cases(
    d_qk: int, with_sink: bool, s_q: int, topk: int, s_kv: int
):
    _run_and_check(d_qk, with_sink, s_q=s_q, topk=topk, s_kv=s_kv)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
