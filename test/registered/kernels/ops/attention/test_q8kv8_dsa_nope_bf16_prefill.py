"""DSA Q8KV8 sparse prefill on NoPE latents, bf16 KV pools and kpool top-k widths.

Covers the GLM-5.3-Flash geometry: qk_rope_head_dim == 0 (zero-width rope views),
512-dim bf16 KV rows, and kpool top-k rows of width index_topk + kpool - 1 = 2051.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    concat_cast_kv_fp8_pad,
    gather_cast_kv_fp8_pad_paged,
)
from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_q_fp8_pad
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

FP8 = torch.float8_e4m3fn
NOPE = 512
KPOOL_TOPK_WIDTH = 2048 + 4 - 1
SM_SCALE = 1.0 / math.sqrt(256)


def _fp8_triton_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def _sm90_available() -> bool:
    return torch.cuda.is_available() and is_sm90_supported()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(not _fp8_triton_available(), reason="Triton fp8 needs SM>=89")
@pytest.mark.parametrize("rope", [0, 64])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("tokens", [1, 437])
def test_concat_and_cast_q_fp8_pad_matches_torch_cast(rope, heads, tokens):
    """NoPE models pass a zero-width q_rope; the rope>0 bytes must stay unchanged."""
    q_nope = (
        (torch.randn(heads, tokens, NOPE, device="cuda") * 0.5)
        .to(torch.bfloat16)
        .transpose(0, 1)
    )
    q_rope = (torch.randn(tokens, heads, 256 + rope, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    q_rope = q_rope.split([256, rope], dim=-1)[1]
    dst = torch.zeros(tokens + 3, 64, NOPE + rope, dtype=FP8, device="cuda")
    _bytes(dst[:, :heads]).fill_(0x7F)
    concat_and_cast_q_fp8_pad(dst, q_nope, q_rope, heads)
    expected = torch.cat([q_nope, q_rope], dim=-1).to(FP8)
    assert torch.equal(_bytes(dst[:tokens, :heads]), _bytes(expected))
    assert (_bytes(dst[:, heads:]) == 0).all()


@pytest.mark.skipif(not _fp8_triton_available(), reason="Triton fp8 needs SM>=89")
@pytest.mark.parametrize("rope,pad_rows", [(0, 2176), (64, 2048)])
def test_concat_cast_kv_fp8_pad_writes_rows_and_zero_band(rope, pad_rows):
    tokens = 437
    head = NOPE + rope
    big = torch.empty(tokens + pad_rows + 50, head, dtype=FP8, device="cuda")
    _bytes(big).fill_(0x7F)
    out = big[: tokens + pad_rows]
    k = (torch.randn(tokens, NOPE, device="cuda") * 0.5).to(torch.bfloat16).unsqueeze(1)
    k_rope = (
        (torch.randn(tokens, head, device="cuda") * 0.5)
        .to(torch.bfloat16)[..., NOPE:]
        .unsqueeze(1)
    )
    concat_cast_kv_fp8_pad(out, k, k_rope, tokens)
    expected = torch.cat([k, k_rope], dim=-1).view(tokens, head).to(FP8)
    assert torch.equal(_bytes(out[:tokens]), _bytes(expected))
    assert (_bytes(out[tokens:]) == 0).all()


@pytest.mark.skipif(not _fp8_triton_available(), reason="Triton fp8 needs SM>=89")
@pytest.mark.parametrize("head_dim", [512, 576])
def test_gather_cast_kv_fp8_pad_paged_matches_torch_cast(head_dim):
    """A radix-hit prefix (paged gather) must produce the same fp8 bytes as the
    non-prefix cast of the same rows."""
    slots, rows, pad_rows = 4096, 333, 2176
    pool = (torch.randn(slots, 1, head_dim, device="cuda") * 50).to(torch.bfloat16)
    idx = torch.randint(0, slots, (rows,), dtype=torch.int32, device="cuda")
    idx[:8] = idx[8]
    out = torch.empty(rows + pad_rows, head_dim, dtype=FP8, device="cuda")
    _bytes(out).fill_(0x7F)
    gather_cast_kv_fp8_pad_paged(
        out=out, kv_cache=pool, page_table_1_flattened=idx, nope_dim=NOPE
    )
    expected = pool.view(-1, head_dim)[idx.long()].to(FP8)
    assert torch.equal(_bytes(out[:rows]), _bytes(expected))
    assert (_bytes(out[rows:]) == 0).all()


def _make_backend():
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend._q8kv8_born_q_stash = None
    backend._q8kv8_born_q_buf = None
    backend._q8kv8_qpad_buf = None
    backend._q8kv8_topk_pad_buf = None
    backend._q8kv8_topk_pad_width = None
    backend._q8kv8_identity_scale = None
    backend._q8kv8_kv_buf = None
    backend._q8kv8_topk_length_enabled = False
    backend._q8kv8_out_bufs = None
    backend.dsa_kv_cache_store_fp8 = False
    backend.kv_lora_rank = NOPE
    return backend


def _make_ragged_case(heads: int, seed: int):
    """Two prefix-extend requests laid out RAGGED: indices address one buffer with
    every request's full context end to end; rows are [picks][tail][-1 ...]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens, extend_lens = [300, 520], [4, 3]
    tokens, kv_rows = sum(extend_lens), sum(seq_lens)
    slots = 2 * kv_rows
    pool = (torch.randn(slots, 1, NOPE, generator=gen, device="cuda") * 0.25).to(
        torch.bfloat16
    )
    flat = torch.randperm(slots, generator=gen, device="cuda")[:kv_rows].to(torch.int32)
    ragged_kv = pool[flat.long(), 0, :]
    indices = torch.full(
        (tokens, KPOOL_TOPK_WIDTH), -1, dtype=torch.int32, device="cuda"
    )
    row, base = 0, 0
    for seq_len, ext in zip(seq_lens, extend_lens):
        for i in range(ext):
            picks = torch.randperm(seq_len, generator=gen, device="cuda")[
                : 1 + (i * 97) % 257
            ]
            indices[row, : picks.numel()] = (picks + base).to(torch.int32)
            row += 1
        base += seq_len
    q_nope = (torch.randn(tokens, heads, NOPE, generator=gen, device="cuda") * 0.25).to(
        torch.bfloat16
    )
    q_rope = q_nope.new_empty((tokens, heads, 0))
    return q_nope, q_rope, pool, flat, ragged_kv, indices


def _torch_reference(q_nope, kv, indices):
    q, kv = q_nope.float(), kv.float()
    out = torch.empty_like(q)
    for i in range(q.shape[0]):
        keys = kv[indices[i][indices[i] >= 0].long()]
        out[i] = torch.softmax((q[i] @ keys.T) * SM_SCALE, dim=-1) @ keys
    return out


@pytest.mark.skipif(not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90")
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("topk_length", [False, True])
def test_q8_helper_nope_bf16_pool_kpool_width(heads, topk_length):
    q_nope, q_rope, pool, flat, ragged_kv, indices = _make_ragged_case(heads, seed=11)
    backend = _make_backend()
    backend._q8kv8_topk_length_enabled = topk_length
    common = dict(
        q_nope=q_nope, q_rope=q_rope, v_head_dim=NOPE, sm_scale=SM_SCALE, layer_id=0
    )
    prefix = backend._forward_flashmla_sparse_q8kv8(
        kv_bf16=None,
        page_table_1=indices,
        paged_kv_cache=pool,
        page_table_1_flattened=flat,
        **common,
    ).clone()
    non_prefix = backend._forward_flashmla_sparse_q8kv8(
        kv_bf16=ragged_kv, page_table_1=indices, **common
    ).clone()
    padded = torch.cat(
        (indices, indices.new_full((indices.shape[0], 2176 - KPOOL_TOPK_WIDTH), -1)),
        dim=-1,
    )
    pre_padded = backend._forward_flashmla_sparse_q8kv8(
        kv_bf16=ragged_kv, page_table_1=padded, **common
    ).clone()
    torch.cuda.synchronize()
    assert prefix.shape == (q_nope.shape[0], heads, NOPE)
    assert torch.equal(prefix, non_prefix)
    assert torch.equal(non_prefix, pre_padded)
    ref = _torch_reference(q_nope, ragged_kv, indices)
    assert torch.isfinite(prefix.float()).all()
    assert (prefix.float() - ref).abs().mean().item() < 0.03
    torch.testing.assert_close(prefix.float(), ref, atol=2.5e-1, rtol=3.0e-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
