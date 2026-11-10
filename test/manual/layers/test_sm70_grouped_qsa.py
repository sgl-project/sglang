"""Grouped Page4 attention must preserve visibility and dynamic paged mapping."""


import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Grouped QSA requires V100",
)


@pytest.mark.parametrize("prefix,rows", [(0, 8), (2048, 16), (25000, 16), (262000, 8)])
def test_grouped_qsa_reference_and_graph_replay(prefix, rows):
    from sglang.srt.layers.attention.tilelang_fa_v100._grouped_qsa import (
        grouped_qsa_prefill,
        load_grouped_qsa_ops,
    )

    assert load_grouped_qsa_ops() is not None, (
        "Install or select the grouped QSA test library"
    )
    torch.manual_seed(751 + prefix)
    max_length = prefix + rows + 3
    capacity = ((max_length + 15) // 16) * 16
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(capacity, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    pages = capacity // 16
    table = torch.stack(
        [
            (
                torch.randperm(pages, device="cuda", dtype=torch.int32)[:, None] * 16
                + torch.arange(16, device="cuda", dtype=torch.int32)
            ).flatten()
            for _ in range(2)
        ]
    )
    req = torch.zeros(1, device="cuda", dtype=torch.int32)
    lens = torch.tensor([prefix + rows], device="cuda", dtype=torch.int32)
    indices = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)

    def update(iteration):
        length = prefix + rows + iteration
        q.normal_()
        req.fill_(iteration % 2)
        lens.fill_(length)
        indices.fill_(-1)
        expected = []
        for row in range(rows):
            visible = length - rows + row + 1
            complete = min(512, visible // 4)
            ids = torch.randperm(visible // 4, device="cuda")[:complete]
            selected = (ids[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
            indices[row, : selected.numel()] = selected.int()
            tail = visible % 4
            indices[row, complete * 4 : complete * 4 + tail] = torch.arange(
                visible - tail, visible, device="cuda", dtype=torch.int32
            )
            # Drop entire groups; the compressed-QSA ABI never leaves holes
            # inside a complete Page4 group.
            if complete > 8:
                indices[row, 12:16] = -1
            if row == 0:
                indices[row].fill_(-1)
            selected = indices[row]
            selected = selected[(selected >= 0) & (selected < visible)].long()
            if selected.numel() == 0:
                expected.append(torch.zeros(6, 256, device="cuda"))
            else:
                slots = table[iteration % 2, selected].long()
                scores = q[row].float() @ k[slots, 0].float().T * 0.0625
                expected.append(scores.softmax(-1) @ v[slots, 0].float())
        return torch.stack(expected)

    update(0)
    grouped_qsa_prefill(q, k, v, table, req, indices, lens, 0.0625)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = grouped_qsa_prefill(q, k, v, table, req, indices, lens, 0.0625)
    for iteration in range(3):
        expected = update(iteration)
        graph.replay()
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[0]) == 0
        torch.testing.assert_close(output.float(), expected, rtol=0.005, atol=0.001)
        assert (output.float() - expected).norm() / expected.norm() < 0.001


def test_grouped_dispatch_is_limited_to_structured_fp16_page4(monkeypatch):
    from sglang.srt.layers.attention.tilelang_fa_v100._grouped_qsa import (
        can_use_grouped_qsa,
    )

    monkeypatch.setenv("SGLANG_SM70_QSA_GROUPED_PREFILL", "1")
    q = torch.zeros(2048, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.zeros(4096, 1, 256, device="cuda", dtype=torch.float16)
    ids = torch.zeros(2048, 2051, device="cuda", dtype=torch.int32)
    assert can_use_grouped_qsa(q, k, k, ids, 16)
    assert not can_use_grouped_qsa(q, k, k, ids, None)
    assert not can_use_grouped_qsa(q, k, k, ids, 1)
    assert not can_use_grouped_qsa(q[:128], k, k, ids[:128], 16)
    assert not can_use_grouped_qsa(q, k, k, ids[:, :2048], 16)
    fp8 = k.to(torch.float8_e5m2)
    assert not can_use_grouped_qsa(q, fp8, fp8, ids, 16)
    monkeypatch.setenv("SGLANG_SM70_QSA_GROUPED_PREFILL", "0")
    assert not can_use_grouped_qsa(q, k, k, ids, 16)
