"""Actual paged backend GBSA routing, causal tail and changing graph inputs."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires ops930 and an idle NPU", allow_module_level=True)

from test_qsa_compact_prefill_npu import make_case
from test_qsa_request_li_npu import runtime_context  # noqa: F401
import torch
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend, QwenSparseAttnMetadata


@pytest.mark.parametrize("extends,prefixes,marker", [
    ([4], [12000], False),
    ([1] * 32, [12000 + i % 4 for i in range(32)], False),
    ([4] * 32, [12000 + i % 4 for i in range(32)], False),
    ([1] * 8, [0, 1, 2, 3, 2043, 2047, 2048, 2050], False),
    ([4] * 4, [60000] * 4, False),
    ([4], [4096], True),
])
def test_actual_paged_consumer(monkeypatch, extends, prefixes, marker):
    c = make_case(extends, [p + 8 for p in prefixes])
    rows = c.q.shape[0]
    request_pages = (c.table[:, ::128] // 128).int().contiguous()
    pages = request_pages.index_select(0, c.sequence_ids.long()).contiguous()
    lengths = torch.empty(rows, dtype=torch.int32, device="npu")
    row_ids = torch.arange(rows, dtype=torch.int32, device="npu")
    cu_q = torch.arange(rows + 1, dtype=torch.int64, device="npu")
    layer_cache = SimpleNamespace(key=lambda: c.k.unsqueeze(1), value=lambda: c.v.unsqueeze(1))
    pool = SimpleNamespace(qsa_compressed_page_size=32, qsa_compress_ratio=4,
                           layer_router=SimpleNamespace(read=lambda _: layer_cache))
    index = QSAIndexerMetadata(
        sequence_lengths=lengths, token_to_batch_idx=row_ids, token_slot_table=c.table,
        out_cache_loc=row_ids, token_to_kv_pool=pool, compress_ratio=4, block_topk=512,
        is_cuda_graph=True, graph_compressed_page_table=pages,
        decode_logical_positions=c.positions, npu_gbsa_cu_q=cu_q,
    )
    metadata = QwenSparseAttnMetadata(sequence_lengths=lengths, token_to_batch_idx=row_ids,
        token_slot_table=c.table, indexer_metadata=index, is_cuda_graph=True)
    backend = QwenSparseAttnBackend()
    backend.token_to_kv_pool = pool
    backend.forward_metadata = metadata
    backend.qsa_profile = SimpleNamespace(block_topk=512, budget=2048, compress_ratio=4)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=3, head_dim=256, scaling=1 / 16)

    def update(step):
        positions, selected = [], torch.full((rows, 512), -1, dtype=torch.int32)
        row = 0
        for prefix, extend in zip(prefixes, extends):
            for pos in range(prefix + step, prefix + step + extend):
                full = (pos + 1) // 4
                groups = torch.arange(512) if marker else torch.randperm(full)[:512]
                selected[row, :groups.numel()] = groups.int()
                positions.append(pos)
                row += 1
        c.positions.copy_(torch.tensor(positions, dtype=torch.int32))
        lengths.copy_(c.positions + 1)
        c.groups.copy_(selected)
        if step:
            pages.add_(1).remainder_(c.k.shape[0] // 128)
            c.q.mul_(-0.875)
            c.k.mul_(0.75)
            c.v.add_(0.125)

    update(0)
    if marker:
        c.q.zero_(); c.k.zero_(); c.v.zero_()
        last = torch.arange(2044, 2048, device="npu")
        tail = torch.arange(4096, 4100, device="npu")
        c.v[pages[0, last // 128].long() * 128 + last % 128, 0, 0] = 1024
        c.v[pages[0, tail // 128].long() * 128 + tail % 128, 0, 1] = 1024

    def run():
        return backend._forward_paged_attention(c.q, layer, None, c.groups)

    graphs, outputs = {}, {}
    for name, enabled in (("native", "0"), ("gbsa", "1")):
        monkeypatch.setenv("SGLANG_NPU_QSA_PAGED_GBSA", enabled)
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=False):
            outputs[name] = run()
        graphs[name] = graph
    for step in range(1 if marker else 3):
        if step:
            update(step)
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs["gbsa"], outputs["native"], atol=0.004, rtol=0.02)
        assert bool(torch.isfinite(outputs["gbsa"]).all())
        if marker:
            tails = (c.positions + 1) % 4
            expected = torch.stack((4096 / (2048 + tails), 1024 * tails / (2048 + tails)), -1)
            torch.testing.assert_close(outputs["gbsa"].view(rows, 3, 256)[:, 0, :2].float(), expected.float(), atol=0.008, rtol=0.01)
    # The draft's expanded token contract still uses its original consumer.
    tokens = expand_qsa_block_indices(c.groups, c.positions, lengths, 4, 2048)
    assert backend._try_npu_paged_gbsa(c.q, c.k.unsqueeze(1), c.v.unsqueeze(1), metadata, tokens, 1 / 16) is None
    expected = backend._forward_paged_attention(c.q, layer, None, tokens)
    torch.testing.assert_close(outputs["gbsa"], expected, atol=0.004, rtol=0.02)
