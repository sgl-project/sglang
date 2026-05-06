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


def _make_case(d_qk: int, with_sink: bool):
    s_q = 2
    q = _make_fp8_tensor((s_q, H_Q, d_qk), seed=1000 + d_qk)
    kv = torch.zeros((S_KV + 1, H_KV, d_qk), dtype=DTYPE_FP8, device="cuda")
    kv[:S_KV] = _make_fp8_tensor((S_KV, H_KV, d_qk), seed=2000 + d_qk)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(3000 + d_qk)
    indices = torch.randint(
        0,
        S_KV,
        (s_q, H_KV, TOPK),
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
    topk_length = torch.tensor([TOPK, 96], dtype=torch.int32, device="cuda")
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
    q_f32 = q.float() * q_scale.item()
    kv_f32 = kv.float() * kv_scale.item()
    out = torch.empty(
        (q.shape[0], q.shape[1], D_V), dtype=torch.float32, device=q.device
    )

    for q_idx in range(q.shape[0]):
        valid_topk = TOPK if topk_length is None else int(topk_length[q_idx].item())
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


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("d_qk,with_sink", [(512, False), (576, True)])
def test_flashmla_q8kv8_sparse_prefill_matches_reference(d_qk: int, with_sink: bool):
    from sglang.jit_kernel.flashmla_q8kv8_sparse_prefill import (
        flash_mla_sparse_q8kv8_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk, with_sink
    )

    out, max_logits, lse = flash_mla_sparse_q8kv8_fwd(
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
