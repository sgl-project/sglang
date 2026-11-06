"""QSA K/V host offload on the GPU: pinned host store, JIT row copies and the
decode gather.

Each forward packs the selected K/V twice with the decode gather the QSA
backend runs (``qwen_sparse_kv_extraction_compact_triton``): once from the full
logical pool through ``req_to_token``, once from the offload's device rows
through its per-forward row table. The packed scratch and valid counts must be
bit-identical, under hot-cache eviction, radix remaps and chain verify, and for
the MTP draft's decode rows, whose extra columns may precede the tail ring.
"""

import sys

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    compute_offload_layout,
    draft_extra_columns,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CR, BLOCK_TOPK, HEADS, DIM, STRIDE = 4, 512, 2, 256, 2112
TOPK = BLOCK_TOPK * CR + CR - 1


def _packed(k, v, table, row_req, indices, lens, rows, packed_dtype, width=TOPK):
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    ok = torch.empty(rows * STRIDE, HEADS, DIM, device="cuda", dtype=packed_dtype)
    ov = torch.empty_like(ok)
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * STRIDE
    qwen_sparse_valid_counts_triton(lens, indices, counts, rows, width)
    qwen_sparse_kv_extraction_compact_triton(
        k, v, table, row_req, indices, lens, cu, ok, ov, rows, width,
        zero_fill_cols=STRIDE,
    )  # fmt: skip
    return counts, ok, ov


@pytest.mark.parametrize("torch_resolve", [False, True])
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
    rows_per_request, dtype, packed_dtype, torch_resolve, request
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
    with envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.override(torch_resolve):
        offload = QSAKVHostOffload(
            layout=layout,
            k_buffers=[dev_k],
            v_buffers=[dev_v],
            host_rows=pool_size + 64,
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
        offload.set_num_real(requests)
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
    if offload.fused is not None:
        assert int(offload.fused.errors) == 0


@pytest.mark.parametrize("torch_resolve", [False, True])
def test_draft_decode_rows_with_extra_columns(torch_resolve, request):
    """MTP draft decode steps reuse the selection captured at an earlier
    position (its expanded top-k and partial block) plus the positions drafted
    since; once the step crosses a block boundary those columns precede the
    tail ring and must resolve to rows holding exactly the pool's K/V."""
    torch.manual_seed(13)
    steps, requests, width, pool_size = 3, 4, 12288, 65536
    extra = draft_extra_columns(compress_ratio=CR, speculative_num_steps=steps)
    row_width = TOPK + steps + 1
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=BLOCK_TOPK * CR,
        hot_blocks=BLOCK_TOPK + extra,
        num_request_slots=requests + 1,
        rows_per_request=1,
        max_extend_tokens=4096,
        page_size=64,
        extra_columns=extra,
    )
    shape = (layout.device_rows + 64, HEADS, DIM)
    dev_k = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    dev_v = torch.zeros_like(dev_k)
    with envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.override(torch_resolve):
        offload = QSAKVHostOffload(
            layout=layout,
            k_buffers=[dev_k],
            v_buffers=[dev_v],
            host_rows=pool_size + 64,
        )
    request.addfinalizer(offload.destroy)
    pool_k = torch.randn(pool_size + 64, HEADS, DIM, device="cuda").bfloat16()
    pool_v = torch.randn_like(pool_k)
    req_to_token = torch.zeros((requests + 1, width), dtype=torch.int32, device="cuda")
    free = (torch.randperm(pool_size, device="cuda") + 64).tolist()

    def alloc(n):
        nonlocal free
        out, free = free[:n], free[n:]
        return torch.tensor(out, dtype=torch.int64, device="cuda")

    row_req = torch.arange(1, requests + 1, device="cuda")
    lengths = torch.randint(3000, 9000, (requests,), device="cuda")
    # Shorter than the top-k: the capture holds fewer than BLOCK_TOPK blocks.
    lengths[0] = 1000
    offload.begin_forward(req_to_token, row_req)
    for i in range(requests):
        slots = alloc(int(lengths[i]))
        req_to_token[i + 1, : slots.numel()] = slots.int()
        for start in range(0, slots.numel(), 4096):
            chunk = slots[start : start + 4096]
            rows = offload.extend_rows(chunk.numel())
            dev_k[rows], dev_v[rows] = pool_k[chunk], pool_v[chunk]
            offload.write_through(layer_idx=0, rows=rows, loc=chunk)

    columns = torch.arange(row_width, device="cuda", dtype=torch.int32)
    tail = torch.arange(steps + 1, device="cuda")
    crossed = 0
    for iteration in range(6):
        # Draft extend: the capture at each request's newest position.
        anchor = lengths - 1
        blocks = torch.full(
            (requests, BLOCK_TOPK), -1, device="cuda", dtype=torch.int32
        )
        for row in range(requests):
            chosen = torch.randperm(int(anchor[row] + 1) // CR, device="cuda")
            blocks[row, : min(BLOCK_TOPK, chosen.numel())] = chosen[:BLOCK_TOPK].int()
        captured = expand_qsa_block_indices(
            blocks, anchor, (anchor + 1).int(), CR, BLOCK_TOPK * CR
        )
        for step in range(steps):
            qpos = lengths.clone()
            loc = alloc(requests)
            req_to_token[row_req, qpos] = loc.int()
            pool_k[loc] = torch.randn(requests, HEADS, DIM, device="cuda").bfloat16()
            lens = (qpos + 1).int()
            drafted = (anchor + 1)[:, None] + tail
            positions = torch.cat(
                [captured, torch.where(drafted <= qpos[:, None], drafted, -1).int()],
                dim=1,
            ).contiguous()
            crossed += int((qpos // CR != anchor // CR).sum())
            offload.begin_forward(req_to_token)
            offload.set_num_real(requests)
            new_rows = offload.new_token_rows(row_seq_lens=lens, row_req=row_req)
            dev_k[new_rows], dev_v[new_rows] = pool_k[loc], pool_v[loc]
            offload.write_through(layer_idx=0, rows=new_rows, loc=loc)
            rows = offload.resolve_rows(
                layer_idx=0,
                positions=positions,
                row_seq_lens=lens,
                row_req=row_req,
                rows_per_request=1,
            )
            expected = _packed(
                pool_k, pool_v, req_to_token, row_req.int(), positions, lens,
                requests, torch.bfloat16, width=row_width,
            )  # fmt: skip
            actual = _packed(
                dev_k, dev_v, rows, torch.arange(requests, device="cuda"),
                torch.where(rows >= 0, columns, -1),
                torch.full((requests,), row_width, device="cuda", dtype=torch.int32),
                requests, torch.bfloat16, width=row_width,
            )  # fmt: skip
            for got, ref in zip(actual, expected):
                assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
            lengths += 1
        # The verify accepts a prefix of the drafts: rewind before the next round.
        lengths -= torch.randint(0, steps, (requests,), device="cuda")
    assert crossed > 0
    if offload.fused is not None:
        assert int(offload.fused.errors) == 0


def test_fused_resolve_in_cuda_graph_with_padding(request):
    """The fused resolve captured once in a CUDA graph and replayed with new
    positions, new req_to_token mappings and a smaller num_real: real rows gather
    exactly the pool's K/V; padded rows are -1 and change no table."""
    torch.manual_seed(11)
    requests, width, pool_size = 4, 8192, 32768
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=BLOCK_TOPK * CR,
        hot_blocks=BLOCK_TOPK,
        num_request_slots=requests + 1,
        rows_per_request=1,
        max_extend_tokens=4096,
        page_size=64,
    )
    shape = (layout.device_rows + 64, HEADS, DIM)
    dev_k = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    dev_v = torch.zeros_like(dev_k)
    offload = QSAKVHostOffload(
        layout=layout, k_buffers=[dev_k], v_buffers=[dev_v], host_rows=pool_size + 64
    )
    request.addfinalizer(offload.destroy)
    assert offload.use_fused
    pool_k = torch.randn(pool_size + 64, HEADS, DIM, device="cuda").bfloat16()
    pool_v = torch.randn_like(pool_k)
    req_to_token = torch.zeros((requests + 1, width), dtype=torch.int32, device="cuda")
    free = (torch.randperm(pool_size, device="cuda") + 64).tolist()
    lengths = [0] + [3000 + 700 * r for r in range(requests)]

    def alloc(n):
        nonlocal free
        out, free = free[:n], free[n:]
        return torch.tensor(out, dtype=torch.int64, device="cuda")

    offload.begin_forward(req_to_token, torch.arange(1, requests + 1, device="cuda"))
    for req in range(1, requests + 1):
        slots = alloc(lengths[req])
        req_to_token[req, : lengths[req]] = slots.int()
        for start in range(0, lengths[req], 4096):
            chunk = slots[start : start + 4096]
            rows = offload.extend_rows(chunk.numel())
            dev_k[rows], dev_v[rows] = pool_k[chunk], pool_v[chunk]
            offload.write_through(layer_idx=0, rows=rows, loc=chunk)

    positions = torch.full((requests, TOPK), -1, dtype=torch.int32, device="cuda")
    lens = torch.ones(requests, dtype=torch.int32, device="cuda")
    row_req = torch.arange(1, requests + 1, device="cuda")

    def stage(real):
        """New tokens of the real requests, their selection into the static inputs."""
        qpos = torch.tensor(lengths[1 : real + 1], device="cuda")
        loc = alloc(real)
        req_to_token[row_req[:real], qpos] = loc.int()
        pool_k[loc] = torch.randn(real, HEADS, DIM, device="cuda").bfloat16()
        blocks = torch.full((real, BLOCK_TOPK), -1, device="cuda", dtype=torch.int32)
        for row in range(real):
            chosen = torch.randperm(int(qpos[row] + 1) // CR, device="cuda")
            blocks[row, : min(BLOCK_TOPK, chosen.numel())] = chosen[:BLOCK_TOPK].int()
        lens[:real] = (qpos + 1).int()
        positions[:real] = expand_qsa_block_indices(
            blocks, qpos, lens[:real], CR, BLOCK_TOPK * CR
        )
        new_rows = offload.new_token_rows(
            row_seq_lens=lens[:real], row_req=row_req[:real]
        )
        dev_k[new_rows], dev_v[new_rows] = pool_k[loc], pool_v[loc]
        offload.write_through(layer_idx=0, rows=new_rows, loc=loc)
        for r in range(real):
            lengths[r + 1] += 1

    def resolve():
        return offload.resolve_rows(
            layer_idx=0,
            positions=positions,
            row_seq_lens=lens,
            row_req=row_req,
            rows_per_request=1,
        )

    stage(requests)
    offload.set_num_real(requests)
    resolve()  # warm up the Triton kernels outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows = resolve()
    fused = offload.fused
    columns = torch.arange(TOPK, device="cuda", dtype=torch.int32)
    for step in range(8):
        real = requests if step % 2 == 0 else requests - 1
        stage(real)
        if step == 4:  # radix remap of a resident block of request 1
            cols = torch.arange(CR, device="cuda") + CR * 100
            slots = alloc(CR)
            req_to_token[1, cols] = slots.int()
            offload.host_k[0][slots.cpu()] = pool_k[slots].cpu()
            offload.host_v[0][slots.cpu()] = pool_v[slots].cpu()
        padded = requests
        before = [
            t[0, padded].clone()
            for t in (fused.blk_slot, fused.slot_blk, fused.ring, fused.ring_src)
        ]
        offload.set_num_real(real)
        graph.replay()
        expected = _packed(
            pool_k, pool_v, req_to_token, row_req[:real].int(), positions[:real],
            lens[:real], real, torch.bfloat16,
        )  # fmt: skip
        actual = _packed(
            dev_k, dev_v, rows[:real], torch.arange(real, device="cuda"),
            torch.where(rows[:real] >= 0, columns, -1),
            torch.full((real,), TOPK, device="cuda", dtype=torch.int32),
            real, torch.bfloat16,
        )  # fmt: skip
        for got, ref in zip(actual, expected):
            assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
        real_rows = offload.real_rows(num_rows=requests, rows_per_request=1)
        assert real_rows.tolist() == [r < real for r in range(requests)]
        if real < requests:
            assert bool((rows[real:] == -1).all())
            after = (fused.blk_slot, fused.slot_blk, fused.ring, fused.ring_src)
            for old, new in zip(before, after):
                assert torch.equal(old, new[0, padded])
    assert int(fused.errors) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
