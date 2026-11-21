"""NPU slot mapping with live graph tables and exact compact expansion."""

import os
from types import MethodType, SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU", allow_module_level=True)

import sgl_kernel_npu  # noqa: F401 -- register vendors before NPU init
import torch
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode


def make_mapping_case(rows, compact, indirect, strided):
    torch.npu.set_device(0)
    torch.manual_seed(103)
    dtype = torch.int64 if strided else torch.int32
    lengths = torch.tensor([0, 1, 3, 2049, 16383], dtype=dtype)
    sequence_ids = torch.arange(rows, dtype=dtype) % len(lengths)
    positions = (lengths[sequence_ids.long()] - 1).clamp_min(0)
    requests = torch.tensor([4, 2, 0, 6, 1], dtype=dtype) if indirect else None
    page_ids = torch.randint(1, 8000, (7 if indirect else 5, 128), dtype=dtype)
    table = (page_ids[:, :, None] * 128 + torch.arange(128)).reshape(-1, 16384)
    width = 512 if compact else (2055 if indirect else 2051)
    indices = torch.full((rows, width), -1, dtype=dtype)
    if compact:
        for row, pos in enumerate(positions.tolist()):
            count = min(512, (pos + 1) // 4, int(lengths[sequence_ids[row]]) // 4)
            # Deliberate repeated groups: mapping must preserve order/duplicates.
            if count:
                indices[row, :count] = torch.arange(count) // 2
    else:
        indices = torch.randint(-3, 16392, (rows, width), dtype=dtype)
        indices[:, :4] = torch.tensor([-1, 0, 1, 16384])

    def to_device(x):
        if x is None:
            return None
        if not strided:
            return x.npu()
        storage = torch.empty(tuple(n * 2 + 1 for n in x.shape), dtype=x.dtype, device="npu")
        view = storage[tuple(slice(1, n * 2 + 1, 2) for n in x.shape)]
        view.copy_(x)
        return view

    indices, sequence_ids, lengths, table, requests, positions = map(
        to_device, (indices, sequence_ids, lengths, table, requests, positions)
    )
    meta = SimpleNamespace(
        token_to_batch_idx=sequence_ids, sequence_lengths=lengths,
        token_slot_table=torch.full_like(table, -2) if indirect else table,
        is_cuda_graph=indirect, req_to_token=table if indirect else None,
        row_req_pool_indices=requests,
    )
    pool = SimpleNamespace(qsa_compressed_page_size=32)
    page_view = to_device(torch.empty(5, 128, dtype=dtype))
    indexer = SimpleNamespace(
        token_to_kv_pool=pool, sequence_lengths=lengths, compress_ratio=4,
        token_to_batch_idx=torch.arange(5, dtype=torch.int32, device="npu"),
        graph_compressed_lengths=torch.empty(5, dtype=torch.int32, device="npu"),
        graph_write_locs=torch.empty(5, dtype=torch.int32, device="npu"),
        decode_logical_positions=torch.empty(5, dtype=torch.int32, device="npu"),
        pending_ring_slots=torch.empty(5, dtype=torch.int64, device="npu"),
        graph_ring_group_locs=torch.empty(5, 4, dtype=torch.int32, device="npu"),
        graph_compressed_page_table=page_view if indirect else None,
        decode_page_table=None if indirect else page_view,
    )
    meta.indexer_metadata = indexer
    indexer.get_seqlens_expanded = lambda: meta.sequence_lengths[meta.token_to_batch_idx.long()]
    backend = SimpleNamespace(
        token_to_kv_pool=pool, compress_ratio=4, req_to_token=table,
        qsa_profile=SimpleNamespace(block_topk=512, budget=2048),
        _logical_to_physical=QwenSparseAttnBackend._logical_to_physical,
    )
    for name in ("_uses_block_indices", "_expand_block_indices"):
        setattr(backend, name, MethodType(getattr(QwenSparseAttnBackend, name), backend))
    c = SimpleNamespace(indices=indices, meta=meta, backend=backend, positions=positions if compact else None)
    refresh_pages(c)
    return c


def refresh_pages(c):
    if c.meta.is_cuda_graph:
        # Use the real replay updater, so changed request IDs/slots reach the
        # exact page buffer shared with LightningIndexer.
        QwenSparseAttnBackend._update_qsa_cuda_graph_metadata(
            c.backend, c.meta.indexer_metadata, c.meta.row_req_pool_indices
        )
    else:
        c.meta.indexer_metadata.decode_page_table.copy_(c.meta.token_slot_table[:, ::128] // 128)


def mapped(c, indices=None, positions=None):
    return QwenSparseAttnBackend._npu_indices_to_physical(
        c.backend, c.indices if indices is None else indices, c.meta,
        c.positions if indices is None else positions,
    )


def cpu_reference(c):
    ids = c.indices.cpu()
    seq = c.meta.token_to_batch_idx.cpu().long()
    lengths = c.meta.sequence_lengths.cpu()[seq]
    if c.positions is not None:
        tokens = torch.full((ids.shape[0], 2051), -1, dtype=torch.int64)
        for row, pos in enumerate(c.positions.cpu().tolist()):
            selected = [int(b) * 4 + off for b in ids[row].tolist() if b >= 0 for off in range(4)]
            selected += list(range((pos + 1) // 4 * 4, min(pos + 1, int(lengths[row]))))
            tokens[row, :len(selected)] = torch.tensor(selected, dtype=torch.int64)
        ids = tokens
    table = (c.meta.req_to_token if c.meta.is_cuda_graph else c.meta.token_slot_table).cpu()
    req = c.meta.row_req_pool_indices.cpu().long()[seq] if c.meta.is_cuda_graph else seq
    slots = table[req[:, None], ids.clamp(0, table.shape[1] - 1).long()]
    return torch.where((ids >= 0) & (ids < lengths[:, None]), slots, -1).int()


@pytest.mark.parametrize("rows,compact,indirect,strided", [
    (0, False, False, False), (1, False, True, True),
    (4, True, True, False), (32, False, True, False),
    (33, True, True, False), (31, False, True, False),
    (128, True, True, False),
    (128, True, True, True), (128, False, False, True),
    (4096, True, False, False), (4096, False, False, False),
])
def test_mapping_live_replay(rows, compact, indirect, strided):
    c = make_mapping_case(rows, compact, indirect, strided)

    def run():
        return mapped(c)

    torch.testing.assert_close(run().cpu(), cpu_reference(c), atol=0, rtol=0)
    if not rows:
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        output = run()
    for iteration in range(3):
        if iteration:
            table = c.meta.req_to_token if indirect else c.meta.token_slot_table
            table.add_(128)
            if indirect:
                c.meta.row_req_pool_indices.add_(1).remainder_(table.shape[0])
            if compact:
                # Keep valid IDs a prefix and preserve all visibility bounds.
                c.indices.copy_(torch.where(c.indices >= 0, c.indices // 2, -1))
                lengths = c.meta.sequence_lengths
                lengths.copy_(torch.where(lengths > 0, lengths + 1, lengths).clamp_max(table.shape[1]))
                c.positions.copy_((lengths[c.meta.token_to_batch_idx.long()] - 1).clamp_min(0))
            else:
                c.indices.add_(1)
                c.meta.token_to_batch_idx.add_(1).remainder_(5)
                c.meta.sequence_lengths.sub_(1).clamp_min_(0)
            refresh_pages(c)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), cpu_reference(c), atol=0, rtol=0)


def test_compact_matches_existing_expansion_and_token_consumer():
    c = make_mapping_case(128, True, True, False)
    meta = c.meta
    logical = expand_qsa_block_indices(
        c.indices, c.positions, meta.sequence_lengths[meta.token_to_batch_idx.long()], 4, 2048
    )
    expanded = QwenSparseAttnBackend._logical_to_physical(logical, meta)
    fused = mapped(c)
    torch.testing.assert_close(fused, expanded, atol=0, rtol=0)


def test_decode_deferral_keeps_draft_reuse_unchanged():
    backend = SimpleNamespace(
        _fused_kv_pool_eligible=False, _mtp_shared_sparse_indices=None,
        qsa_profile=SimpleNamespace(block_topk=512), compress_ratio=4,
    )
    eligible = QwenSparseAttnBackend._can_defer_block_expansion
    assert eligible(backend, ForwardMode.TARGET_VERIFY)
    assert eligible(backend, ForwardMode.DECODE)
    assert not eligible(backend, ForwardMode.DRAFT_EXTEND_V2)
    backend._mtp_shared_sparse_indices = object()
    assert not eligible(backend, ForwardMode.DECODE)
    assert not eligible(backend, ForwardMode.TARGET_VERIFY)
    backend._fused_kv_pool_eligible = True
    assert not eligible(backend, ForwardMode.DECODE)


def test_paged_consumer_attention_matches_expanded_tokens():
    from test_qsa_compact_prefill_npu import make_case

    c = make_case([128], [4097])
    cache = SimpleNamespace(key=lambda: c.k.unsqueeze(1), value=lambda: c.v.unsqueeze(1))
    pool = SimpleNamespace(qsa_compressed_page_size=32, layer_router=SimpleNamespace(read=lambda _: cache))
    meta = SimpleNamespace(
        token_to_batch_idx=c.sequence_ids, sequence_lengths=c.lengths,
        token_slot_table=c.table, is_cuda_graph=False,
        indexer_metadata=SimpleNamespace(decode_logical_positions=c.positions, decode_page_table=c.table[:, ::128] // 128),
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool, _resolve_metadata=lambda _: meta,
        qsa_profile=SimpleNamespace(block_topk=512), compress_ratio=4,
        _logical_to_physical=QwenSparseAttnBackend._logical_to_physical,
    )
    backend._uses_block_indices = MethodType(QwenSparseAttnBackend._uses_block_indices, backend)
    backend._npu_indices_to_physical = MethodType(QwenSparseAttnBackend._npu_indices_to_physical, backend)
    layer = SimpleNamespace(layer_id=0, scaling=1/16)
    run = QwenSparseAttnBackend._forward_paged_attention
    compact = run(backend, c.q, layer, None, c.groups)
    expanded = run(backend, c.q, layer, None, c.logical)
    torch.testing.assert_close(compact, expanded, atol=0, rtol=0)
