"""PyHIP BF16 full-prefill integration: ragged pages, tails, padding, FlyPA parity and streams."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.aiter_utils import (
    is_gfx942,
    run_mimo_flypa_prefill,
    run_mimo_pyhip_prefill,
)

PAGE, HEADS, QDIM, VDIM = 64, 16, 192, 128


def _require_gfx942():
    if not torch.cuda.is_available() or not is_gfx942():
        pytest.skip("gfx942 required")


def _layer():
    return SimpleNamespace(
        tp_q_head_num=HEADS,
        tp_k_head_num=1,
        qk_head_dim=QDIM,
        v_head_dim=VDIM,
        scaling=QDIM**-0.5,
    )


@pytest.mark.parametrize("physical_rows", [0, 4])
def test_empty_attention_group(physical_rows):
    q = torch.empty(physical_rows, HEADS * QDIM, dtype=torch.bfloat16)
    batch = SimpleNamespace(batch_size=0, extend_seq_lens_cpu=[])
    output = run_mimo_pyhip_prefill(None, q, _layer(), batch, None, None, None)
    assert output.shape == (physical_rows, HEADS * VDIM)
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize(
    "qlens,kvlens,padding",
    [
        ([257], [8193], 17),
        ([1, 63, 257], [17, 129, 4097], 19),
        ([128, 256], [128, 256], 0),
        ([5000, 3000, 1, 4096], [5000, 9000, 70, 4096], 3),
    ],
)
def test_pyhip_ragged_parity_and_streams(qlens, kvlens, padding):
    _require_gfx942()
    torch.manual_seed(714)
    npages = [(n + PAGE - 1) // PAGE for n in kvlens]
    total_pages = sum(npages)
    kd = (
        torch.randn(total_pages + 2, PAGE, 1, QDIM, dtype=torch.bfloat16, device="cuda")
        * 0.2
    )
    vd = (
        torch.randn(total_pages + 2, PAGE, 1, VDIM, dtype=torch.bfloat16, device="cuda")
        * 0.2
    )
    order = torch.randperm(total_pages, device="cuda")
    tables, offset = [], 0
    for pages, length in zip(npages, kvlens):
        physical = order[offset : offset + pages]
        tables.append(physical)
        if length % PAGE:
            kd[physical[-1], length % PAGE :].fill_(float("nan"))
            vd[physical[-1], length % PAGE :].fill_(float("nan"))
        offset += pages
    kd[-2:].fill_(float("nan"))
    vd[-2:].fill_(float("nan"))
    key = kd.view(-1, PAGE, 1, QDIM // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    value = vd.view(-1, PAGE // 8, 8, 1, VDIM).permute(0, 3, 1, 4, 2).contiguous()
    cuq, indptr = [0], [0]
    for ql, pages in zip(qlens, npages):
        cuq.append(cuq[-1] + ql)
        indptr.append(indptr[-1] + pages)
    indices = torch.zeros(indptr[-1] + 256, dtype=torch.int32, device="cuda")
    indices[: indptr[-1]] = torch.cat(tables).to(torch.int32)
    metadata = SimpleNamespace(
        paged_kv_indptr=torch.tensor(indptr, device="cuda", dtype=torch.int32),
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.tensor(
            [(n - 1) % PAGE + 1 for n in kvlens], device="cuda", dtype=torch.int32
        ),
        max_q_len=max(qlens),
        max_kv_len=max(kvlens),
    )
    backend = SimpleNamespace(
        qo_indptr=torch.tensor(cuq, device="cuda", dtype=torch.int32),
        page_size=PAGE,
        input_dtype=torch.bfloat16,
    )
    batch = SimpleNamespace(batch_size=len(qlens), extend_seq_lens_cpu=qlens)
    layer = _layer()
    pool = SimpleNamespace(dtype=torch.bfloat16, store_dtype=torch.bfloat16)
    packed_qkv = (
        torch.randn(
            cuq[-1] + padding,
            HEADS * QDIM + QDIM + VDIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    query = packed_qkv[:, : HEADS * QDIM]

    def run(q):
        return run_mimo_pyhip_prefill(backend, q, layer, batch, key, value, metadata)

    def check(output, q):
        live = cuq[-1]
        assert torch.isfinite(output[:live]).all()
        assert torch.equal(output[live:], torch.zeros_like(output[live:]))
        flypa = run_mimo_flypa_prefill(
            backend, q, layer, len(qlens) + 1, pool, key, value, metadata
        )
        torch.testing.assert_close(output[:live], flypa[:live], atol=0.02, rtol=0.02)
        for index, ql in enumerate(qlens):
            sample = torch.tensor(sorted({0, ql // 2, ql - 1}), device="cuda")
            pages = metadata.paged_kv_indices[indptr[index] : indptr[index + 1]].long()
            k = kd[pages].reshape(-1, QDIM)[: kvlens[index]].float()
            v = vd[pages].reshape(-1, VDIM)[: kvlens[index]].float()
            qq = q[cuq[index] + sample].view(-1, HEADS, QDIM).float()
            scores = (qq.reshape(-1, QDIM) @ k.T).view(
                -1, HEADS, len(k)
            ) * layer.scaling
            limit = kvlens[index] - ql + sample + 1
            columns = torch.arange(len(k), device="cuda")[None, None, :]
            scores.masked_fill_(columns >= limit[:, None, None], -float("inf"))
            expected = torch.softmax(scores, -1).reshape(-1, len(k)) @ v
            actual = output[cuq[index] + sample].view(-1, VDIM).float()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    eager = run(query)
    check(eager, query)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = run(query)
    graph.replay()
    torch.testing.assert_close(replay, eager, atol=0, rtol=0)
    query.neg_()
    # Reorder every page but the partial last one, whose tail holds NaNs.
    for a, b in zip(indptr, indptr[1:]):
        metadata.paged_kv_indices[a : b - 1] = metadata.paged_kv_indices[
            a : b - 1
        ].flip(0)
    graph.replay()
    check(replay, query)

    opposite = -query
    torch.cuda.synchronize()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        output1 = run(query)
    with torch.cuda.stream(s2):
        output2 = run(opposite)
    torch.cuda.synchronize()
    check(output1, query)
    check(output2, opposite)
