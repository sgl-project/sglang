"""Check parallel page-table construction against independent pool arithmetic."""

import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("pages", [129, 8192])
def test_page_tiles_and_graph_replay(rows, pages):
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_graph_row_metadata_kernel,
    )

    torch.manual_seed(19)
    ratio, full_page = 4, 32
    page_ids = torch.randint(1, 10000, (6, pages), device="cuda")
    pool = (
        page_ids[:, :, None] * full_page
        + torch.arange(full_page, device="cuda")[None, None, :]
    ).flatten(1)
    seq = torch.tensor([1, 4, 251, 2500][:rows], dtype=torch.int32, device="cuda")
    req = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")

    def buffers():
        return [
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty((rows, pages), dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int64, device="cuda"),
            torch.empty((rows, ratio), dtype=torch.int32, device="cuda"),
        ]

    serial, tiled = buffers(), buffers()

    def run(outputs, tiles):
        _qsa_graph_row_metadata_kernel[(rows, tiles)](
            seq,
            req,
            *outputs,
            pool,
            pool.stride(0),
            pages,
            RATIO=ratio,
            FULL_PAGE=full_page,
            PAGE_BLOCK=128,
            num_warps=1,
        )

    run(serial, 1)
    run(tiled, triton.cdiv(pages, 128))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(tiled, triton.cdiv(pages, 128))
    for increment in [0, 1, 2, 3]:
        seq.add_(increment)
        req.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) + increment % 2)
        graph.replay()
        run(serial, 1)
        current = seq - 1
        last = pool[req.long(), current.long()]
        expected = [
            seq // ratio,
            torch.where(seq % ratio == 0, last // ratio, 0),
            page_ids[req.long()],
            current,
            req.long() * ratio + current % ratio,
            req[:, None] * ratio
            + (
                current[:, None] - torch.arange(ratio - 1, -1, -1, device="cuda")
            ).clamp_min(0)
            % ratio,
        ]
        for actual, old, reference in zip(tiled, serial, expected):
            torch.testing.assert_close(actual, old, rtol=0, atol=0)
            torch.testing.assert_close(
                actual, reference.to(actual.dtype), rtol=0, atol=0
            )


@pytest.mark.parametrize("bs", [1, 2])
def test_draft_extend_layout_replay_with_changing_lengths(bs):
    from sglang.srt.layers.attention.qsa.graph_metadata import _qsa_graph_layout_kernel

    capacity = bs * 4
    seq = torch.full((bs,), 25000, dtype=torch.int32, device="cuda")
    req = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda")
    extend = torch.full_like(seq, 4)
    outputs = [
        torch.empty(capacity, dtype=torch.int32, device="cuda") for _ in range(3)
    ]

    def run():
        _qsa_graph_layout_kernel[(bs + 1,)](
            seq, req, extend, *outputs, bs, capacity, 0, 0, MODE=2, num_warps=1
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lens in ([4] * bs, [1] * bs, [2, 3][:bs], [0] * bs, [4] * bs):
        extend.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        seq.add_(1)
        req.add_(1)
        graph.replay()
        expected = [[], [], []]
        for length, total, slot in zip(lens, seq.tolist(), req.tolist()):
            expected[0].extend(range(total - length + 1, total + 1))
            expected[1].extend([total - length] * length)
            expected[2].extend([slot] * length)
        padding = capacity - sum(lens)
        for actual, values, fill in zip(outputs, expected, [1, 0, 0]):
            reference = torch.tensor(
                values + [fill] * padding, device="cuda", dtype=torch.int32
            )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
