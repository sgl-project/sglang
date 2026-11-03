"""QSA K/V host offload on the GPU: pinned host store, JIT row copies and the
decode gather.

Each forward packs the selected K/V twice with the decode gather the QSA
backend runs (``qwen_sparse_kv_extraction_compact_triton``): once from the full
logical pool through ``req_to_token``, once from the offload's device rows
through its per-forward row table. The packed scratch and valid counts must be
bit-identical, under hot-cache eviction, radix remaps and chain verify.
"""

import sys

import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    compute_offload_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CR, BLOCK_TOPK, HEADS, DIM, STRIDE = 4, 512, 2, 256, 2112
TOPK = BLOCK_TOPK * CR + CR - 1


def _packed(k, v, table, row_req, indices, lens, rows, packed_dtype):
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    ok = torch.empty(rows * STRIDE, HEADS, DIM, device="cuda", dtype=packed_dtype)
    ov = torch.empty_like(ok)
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * STRIDE
    qwen_sparse_valid_counts_triton(lens, indices, counts, rows, TOPK)
    qwen_sparse_kv_extraction_compact_triton(
        k, v, table, row_req, indices, lens, cu, ok, ov, rows, TOPK,
        zero_fill_cols=STRIDE,
    )  # fmt: skip
    return counts, ok, ov


@pytest.mark.parametrize("rows_per_request", [1, 4])
@pytest.mark.parametrize(
    "dtype,packed_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
    ],
)
def test_offload_gather_matches_pool_gather(
    rows_per_request, dtype, packed_dtype, request
):
    torch.manual_seed(7)
    q, requests, width, pool_size = rows_per_request, 4, 12288, 65536
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=BLOCK_TOPK * CR,
        hot_blocks=q * BLOCK_TOPK,
        num_request_slots=requests + 1,
        rows_per_request=q,
        max_extend_tokens=4096,
        page_size=64,
    )
    shape = (layout.device_rows + 64, HEADS, DIM)
    dev_k = torch.zeros(shape, device="cuda", dtype=dtype)
    dev_v = torch.zeros_like(dev_k)
    offload = QSAKVHostOffload(
        layout=layout, k_buffers=[dev_k], v_buffers=[dev_v], host_rows=pool_size + 64
    )
    request.addfinalizer(offload.destroy)
    pool_k = torch.randn(pool_size + 64, HEADS, DIM, device="cuda").to(dtype)
    pool_v = torch.randn_like(pool_k.float()).to(dtype)
    req_to_token = torch.zeros((requests + 1, width), dtype=torch.int32, device="cuda")
    free = (torch.randperm(pool_size, device="cuda") + 64).tolist()
    lengths = {}

    def alloc(n):
        nonlocal free
        out, free = free[:n], free[n:]
        return torch.tensor(out, dtype=torch.int64, device="cuda")

    # Extend: chunks go through the scratch rows and are written through.
    offload.begin_forward(req_to_token, torch.arange(1, requests + 1, device="cuda"))
    for req in range(1, requests + 1):
        lengths[req] = int(torch.randint(3000, 9000, (1,)))
        slots = alloc(lengths[req])
        req_to_token[req, : lengths[req]] = slots.int()
        for start in range(0, lengths[req], 4096):
            chunk = slots[start : start + 4096]
            rows = offload.extend_rows(chunk.numel())
            dev_k[rows], dev_v[rows] = pool_k[chunk], pool_v[chunk]
            offload.write_through(0, rows, chunk)

    for step in range(12):
        if step % 4 == 2:  # radix remap of a resident block to new K/V
            req = 1 + step % requests
            cols = torch.arange(CR, device="cuda") + CR * (lengths[req] // CR // 2)
            slots = alloc(CR)
            req_to_token[req, cols] = slots.int()
            pool_k[slots] = torch.randn_like(pool_k[slots].float()).to(dtype)
            offload.host_k[0][slots.cpu()] = pool_k[slots].cpu()
            offload.host_v[0][slots.cpu()] = pool_v[slots].cpu()
        row_req = torch.arange(1, requests + 1, device="cuda").repeat_interleave(q)
        first = torch.tensor(
            [lengths[r] for r in range(1, requests + 1)], device="cuda"
        )
        qpos = first.repeat_interleave(q) + torch.arange(q, device="cuda").repeat(
            requests
        )
        loc = alloc(requests * q)
        req_to_token[row_req, qpos] = loc.int()
        blocks = torch.full(
            (requests * q, BLOCK_TOPK), -1, device="cuda", dtype=torch.int32
        )
        for row in range(requests * q):
            n = int(qpos[row] + 1) // CR
            chosen = torch.randperm(n, device="cuda")[:BLOCK_TOPK]
            blocks[row, : chosen.numel()] = chosen.int()
        lens = (qpos + 1).int()
        positions = expand_qsa_block_indices(blocks, qpos, lens, CR, BLOCK_TOPK * CR)

        offload.begin_forward(req_to_token)
        new_k = torch.randn(requests * q, HEADS, DIM, device="cuda").to(dtype)
        new_v = torch.randn_like(new_k.float()).to(dtype)
        pool_k[loc], pool_v[loc] = new_k, new_v
        new_rows = offload.new_token_rows(lens, row_req)
        dev_k[new_rows], dev_v[new_rows] = new_k, new_v
        offload.write_through(0, new_rows, loc)
        rows = offload.resolve_rows(0, positions, lens, row_req, q)

        expected = _packed(
            pool_k, pool_v, req_to_token, row_req.int(), positions, lens,
            requests * q, packed_dtype,
        )  # fmt: skip
        columns = torch.arange(TOPK, device="cuda", dtype=torch.int32)
        actual = _packed(
            dev_k, dev_v, rows, torch.arange(requests * q, device="cuda"),
            torch.where(rows >= 0, columns, -1),
            torch.full((requests * q,), TOPK, device="cuda", dtype=torch.int32),
            requests * q, packed_dtype,
        )  # fmt: skip
        for got, ref in zip(actual, expected):
            assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
        for req in range(1, requests + 1):  # accept between 1 and q tokens
            lengths[req] += int(torch.randint(1, q + 1, (1,)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
