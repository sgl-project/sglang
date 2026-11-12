# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import sage_attn_qk_int8_pv_fp16
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not is_xpu(), reason="Intel XPU required")


def _reference(q, k, v, scale, causal):
    qh, kh, vh = (t.transpose(1, 2).float() for t in (q, k, v))
    groups = qh.shape[1] // kh.shape[1]
    kh, vh = kh.repeat_interleave(groups, 1), vh.repeat_interleave(groups, 1)
    scores = (qh @ kh.transpose(-1, -2)) * scale
    if causal:
        q_len, kv_len = scores.shape[-2:]
        future = torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(future.triu(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2), lse


@pytest.mark.parametrize(
    "batch,qo_len,kv_len,heads,kv_heads,head_dim,causal",
    [
        (1, 1024, 1024, 8, 8, 128, False),
        (2, 1000, 1000, 4, 4, 128, False),  # partial tiles at both ends
        (1, 777, 333, 4, 4, 128, False),  # cross-attention
        (1, 1024, 1024, 8, 2, 128, False),  # GQA
        (1, 1000, 1000, 8, 2, 128, True),  # causal GQA text encoder
        (1, 1000, 1000, 4, 4, 64, False),
        (1, 600, 600, 4, 4, 72, False),  # padded head dim
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_fp32_attention(
    batch, qo_len, kv_len, heads, kv_heads, head_dim, causal, dtype
):
    torch.manual_seed(0)
    q = torch.randn(batch, qo_len, heads, head_dim, device="xpu", dtype=dtype)
    # a nonzero key mean exercises the K smoothing and its LSE correction
    k = torch.randn(batch, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype) + 0.5
    v = torch.randn(batch, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype)
    scale = head_dim**-0.5
    expected, expected_lse = _reference(q, k, v, scale, causal)

    out, lse = sage_attn_qk_int8_pv_fp16(
        q, k, v, causal=causal, sm_scale=scale, return_lse=True
    )
    out_no_lse = sage_attn_qk_int8_pv_fp16(q, k, v, causal=causal, sm_scale=scale)

    assert out.shape == q.shape and out.dtype == dtype
    assert lse.shape == (batch, heads, qo_len) and lse.dtype == torch.float32
    # error of per-block INT8 Q/K quantization
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), expected.flatten(), dim=0
    )
    assert cos > 0.999
    assert (lse - expected_lse).abs().max() < 0.05
    torch.testing.assert_close(out, out_no_lse, rtol=0, atol=0)


def test_accepts_strided_qkv_views():
    torch.manual_seed(0)
    qkv = torch.randn(1, 500, 3 * 8 * 128, device="xpu", dtype=torch.bfloat16)
    q, k, v = (t.unflatten(-1, (8, 128)) for t in qkv.chunk(3, dim=-1))
    expected, _ = _reference(q, k, v, 128**-0.5, causal=False)
    out = sage_attn_qk_int8_pv_fp16(q, k, v)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), expected.flatten(), dim=0
    )
    assert cos > 0.999


def test_rejects_unsupported_inputs():
    q = torch.randn(1, 64, 2, 256, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        sage_attn_qk_int8_pv_fp16(q, q, q)
    q = torch.randn(1, 64, 2, 128, device="xpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="FP16/BF16"):
        sage_attn_qk_int8_pv_fp16(q, q, q)
    q = torch.randn(1, 64, 2, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 128, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="causal"):
        sage_attn_qk_int8_pv_fp16(q, k, k, causal=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
