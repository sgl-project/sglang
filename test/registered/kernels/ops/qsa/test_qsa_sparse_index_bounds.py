"""Index-bound regressions for the QSA sparse attention kernels.

Three properties are pinned here:

- the prefill kernels must ignore a selected index that lies past the request
  they are attending for, instead of reading the next request's packed K/V;
- the decode and packing kernels must address their KV pools in 64-bit, so a
  slot or page whose product with its row stride passes 2**31 still reads its
  own row.
"""

import inspect
import sys

import pytest
import torch

from sglang.kernels.ops.attention.qsa import mqa as mqa_module
from sglang.kernels.ops.attention.qsa.mqa import triton_qsa_mqa_decode
from sglang.srt.layers.attention.qsa import sparse_attn as sparse_attn_module
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qsa_sparse_decode_triton,
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("QSA sparse attention requires CUDA")


def test_large_pool_offsets_have_non_skipping_int64_guards():
    checks = (
        (
            sparse_attn_module._qsa_sparse_decode,
            ("request_offset = req.to(tl.int64)", ").to(tl.int64)"),
        ),
        (
            sparse_attn_module._qsa_sparse_decode_splitk,
            ("request_offset = req.to(tl.int64)", ").to(tl.int64)"),
        ),
        (
            sparse_attn_module._compact_kv,
            (
                "req = tl.load(req_indices + batch).to(tl.int64)",
                ").to(tl.int64)",
            ),
        ),
        (mqa_module._qsa_mqa_decode_kernel, (").to(tl.int64)",)),
    )
    for kernel, expected in checks:
        source = inspect.getsource(kernel.fn)
        for expression in expected:
            assert expression in source, f"{kernel.fn.__name__}: {expression}"


def _reference_row(q_row, keys, values):
    scores = torch.einsum("hd,khd->hk", q_row.float(), keys.float()) * SCALE
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hk,khd->hd", probabilities, values.float()).to(q_row.dtype)


def _packed_case(lengths, topk, out_of_range_row, out_of_range_index):
    """Two packed requests where one query row selects a foreign token."""
    torch.manual_seed(20260827)
    device = torch.device("cuda")
    total = sum(lengths)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    indices = torch.full((total, topk), -1, dtype=torch.int32, device=device)
    indices[:, 0] = 0
    indices[out_of_range_row, :3] = torch.tensor(
        [0, 1, out_of_range_index], dtype=torch.int32, device=device
    )
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lengths), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return q, k, v, indices, cu_seqlens


def _expected(q, k, v, lengths, out_of_range_row):
    """Every row attends token 0 of its own request; the marked row attends 0
    and 1 of its own request and nothing else."""
    expected = torch.empty_like(q)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    for request, length in enumerate(lengths):
        start = starts[request]
        for offset in range(length):
            row = start + offset
            local = [0, 1] if row == out_of_range_row else [0]
            slots = [start + t for t in local]
            expected[row] = _reference_row(
                q[row],
                k[slots].repeat_interleave(NUM_Q_HEADS, dim=1),
                v[slots].repeat_interleave(NUM_Q_HEADS, dim=1),
            )
    return expected


def test_prefill_ignores_index_past_the_request():
    _require_cuda()
    lengths = [4, 4]
    q, k, v, indices, cu_seqlens = _packed_case(
        lengths, topk=4, out_of_range_row=3, out_of_range_index=lengths[0]
    )
    out = sparse_gqa_fwd_interface_triton(
        q, k, v, max(lengths), indices, cu_seqlens, SCALE
    )
    expected = _expected(q, k, v, lengths, out_of_range_row=3)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_chunk_prefill_ignores_index_past_the_request():
    _require_cuda()
    lengths = [4, 4]
    q, k, v, indices, cu_seqlens = _packed_case(
        lengths, topk=4, out_of_range_row=3, out_of_range_index=lengths[0]
    )
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device=q.device)
    out = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_seqlens, cu_seqlens, kv_lens, SCALE
    )
    expected = _expected(q, k, v, lengths, out_of_range_row=3)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("chunked", [False, True])
def test_prefill_all_invalid_first_block_returns_zero(chunked):
    _require_cuda()
    lengths = [4, 4]
    q, k, v, indices, cu_seqlens = _packed_case(
        lengths, topk=4, out_of_range_row=0, out_of_range_index=lengths[0]
    )
    indices[0].fill_(lengths[0])
    if chunked:
        kv_lens = torch.tensor(lengths, dtype=torch.int32, device=q.device)
        out = sparse_gqa_fwd_interface_triton_ck(
            q, k, v, indices, cu_seqlens, cu_seqlens, kv_lens, SCALE
        )
    else:
        out = sparse_gqa_fwd_interface_triton(
            q, k, v, max(lengths), indices, cu_seqlens, SCALE
        )
    assert torch.isfinite(out[0]).all()
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]))


def test_decode_slot_past_the_int32_stride_product():
    """A slot whose element offset reaches 2**31 must still read its own row."""
    _require_cuda()
    device = torch.device("cuda")
    # The decode kernel supports head dimension 128 or 256; 128 keeps the pool
    # that spans the boundary as small as it can be.
    head_dim = 128
    stride = NUM_KV_HEADS * head_dim
    wrap_slot = 2**31 // stride
    needed = (wrap_slot + 1) * stride * 2  # bfloat16, one buffer aliased as K and V
    free, _ = torch.cuda.mem_get_info(device)
    if free < needed + 2 * 2**30:
        pytest.skip(
            f"needs {needed / 2**30:.1f} GiB of free device memory for a KV pool "
            "spanning the 32-bit offset boundary"
        )
    kv = torch.zeros(
        wrap_slot + 1, NUM_KV_HEADS, head_dim, dtype=torch.bfloat16, device=device
    )
    marker = torch.arange(head_dim, dtype=torch.bfloat16, device=device) / head_dim
    kv[wrap_slot, 0] = marker

    req_to_token = torch.zeros(1, 1, dtype=torch.int32, device=device)
    req_to_token[0, 0] = wrap_slot
    q = torch.randn(1, NUM_Q_HEADS, head_dim, dtype=torch.bfloat16, device=device)
    indices = torch.zeros(1, 1, dtype=torch.int32, device=device)
    out = qsa_sparse_decode_triton(
        q,
        kv,
        kv,
        req_to_token,
        torch.zeros(1, dtype=torch.int32, device=device),
        indices,
        torch.ones(1, dtype=torch.int32, device=device),
        head_dim**-0.5,
    )
    # One selected token: softmax over a single score is 1, so the output is
    # exactly that slot's value row, repeated across query heads.
    expected = marker.expand(NUM_Q_HEADS, head_dim)
    torch.testing.assert_close(out[0].float(), expected.float(), atol=2e-2, rtol=2e-2)


def _large_kv_pool(rows, head_dim):
    device = torch.device("cuda")
    stride = NUM_KV_HEADS * head_dim
    wrap_row = 2**31 // stride
    needed = (wrap_row + 1) * stride * 2
    free, _ = torch.cuda.mem_get_info(device)
    if free < needed + 2 * 2**30:
        pytest.skip(
            f"needs {needed / 2**30:.1f} GiB of free device memory for a KV pool "
            "spanning the 32-bit offset boundary"
        )
    return (
        torch.zeros(
            wrap_row + 1,
            rows,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        wrap_row,
    )


def test_compact_slot_past_the_int32_stride_product():
    _require_cuda()
    head_dim = 64
    kv, wrap_slot = _large_kv_pool(NUM_KV_HEADS, head_dim)
    marker = torch.arange(head_dim, dtype=torch.bfloat16, device=kv.device) / head_dim
    kv[wrap_slot, 0] = marker

    req_to_token = torch.full((1, 1), wrap_slot, dtype=torch.int32, device=kv.device)
    out_k = torch.empty(1, NUM_KV_HEADS, head_dim, dtype=kv.dtype, device=kv.device)
    out_v = torch.empty_like(out_k)
    qwen_sparse_kv_extraction_compact_triton(
        kv,
        kv,
        req_to_token,
        torch.zeros(1, dtype=torch.int32, device=kv.device),
        torch.zeros(1, 1, dtype=torch.int32, device=kv.device),
        torch.ones(1, dtype=torch.int32, device=kv.device),
        torch.tensor([0, 1], dtype=torch.int32, device=kv.device),
        out_k,
        out_v,
        1,
        1,
    )
    torch.testing.assert_close(out_k[0, 0], marker, atol=0, rtol=0)
    torch.testing.assert_close(out_v[0, 0], marker, atol=0, rtol=0)


def test_mqa_page_past_the_int32_stride_product():
    _require_cuda()
    head_dim = 64
    kv, wrap_page = _large_kv_pool(NUM_KV_HEADS, head_dim)
    kv[wrap_page, 0] = 1
    q = torch.ones(1, NUM_Q_HEADS, head_dim, dtype=torch.bfloat16, device=kv.device)
    logits = triton_qsa_mqa_decode(
        q,
        kv.view(-1, 1, NUM_KV_HEADS, head_dim),
        torch.full((1, 1), wrap_page, dtype=torch.int32, device=kv.device),
        torch.ones(1, dtype=torch.int32, device=kv.device),
        max_model_len=1,
    )
    expected = NUM_Q_HEADS * head_dim / head_dim**0.5
    torch.testing.assert_close(logits, torch.full_like(logits, expected))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
