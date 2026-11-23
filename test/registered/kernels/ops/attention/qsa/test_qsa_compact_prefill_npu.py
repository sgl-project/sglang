"""Opt-in compact QSA producer/GBSA/fallback checks on an idle NPU die."""

import os
from types import MethodType, SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires ops-transformer9.3 and an idle NPU", allow_module_level=True)

import sgl_kernel_npu  # noqa: F401 -- register native vendors before NPU init
import torch
from sgl_kernel_npu.operators.attention import qsa_sparse_attention
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sglang.kernels.ops.attention.qsa_gbsa_prefill import qsa_gbsa_prefill
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode


def make_case(extends, prefixes):
    torch.npu.set_device(0)
    torch.manual_seed(53)
    lengths = [e + p for e, p in zip(extends, prefixes)]
    pool_rows = sum((n + 127) // 128 * 128 for n in lengths) + 128
    q = torch.randn(sum(extends), 3, 256, device="npu", dtype=torch.bfloat16) * 0.2
    k = torch.randn(pool_rows, 1, 256, device="npu", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    pages = torch.randperm(pool_rows // 128)
    table = torch.zeros(len(extends), max(lengths), dtype=torch.int32)
    groups = torch.full((sum(extends), 512), -1, dtype=torch.int32)
    positions, sequence_ids, row, page = [], [], 0, 0
    for seq, (extend, prefix, length) in enumerate(zip(extends, prefixes, lengths)):
        num_pages = (length + 127) // 128
        table[seq, :length] = (pages[page:page + num_pages, None] * 128 + torch.arange(128)).flatten()[:length].int()
        page += num_pages
        for pos in range(prefix, length):
            selected = torch.randperm((pos + 1) // 4)[:512]
            groups[row, :selected.numel()] = selected.int()
            positions.append(pos)
            sequence_ids.append(seq)
            row += 1
    positions = torch.tensor(positions, dtype=torch.int32, device="npu")
    sequence_ids = torch.tensor(sequence_ids, dtype=torch.int32, device="npu")
    lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")
    extends = torch.tensor(extends, dtype=torch.int32, device="npu")
    groups, table = groups.npu(), table.npu()
    logical = expand_qsa_block_indices(groups, positions, lengths[sequence_ids.long()], 4, 2048)
    physical = table[sequence_ids.long()[:, None], logical.clamp_min(0).long()]
    physical[logical < 0] = -1
    return SimpleNamespace(q=q, k=k, v=v, groups=groups, table=table,
                           lengths=lengths, extends=extends, positions=positions,
                           sequence_ids=sequence_ids, logical=logical, physical=physical)


def gbsa(c, compact):
    return qsa_gbsa_prefill(c.q, c.k, c.v, c.groups if compact else c.logical,
                          c.table, c.lengths, c.extends, c.q.shape[0], indices_are_blocks=compact)


@pytest.mark.parametrize("extends,prefixes", [([128], [0]), ([128], [2048]), ([128], [11001]), ([17, 111], [0, 3001])])
def test_compact_tail_multisequence_and_replay(extends, prefixes):
    c = make_case(extends, prefixes)
    legacy = gbsa(c, False)
    compact = gbsa(c, True)
    torch.testing.assert_close(compact, legacy, atol=0, rtol=0)
    reference = qsa_sparse_attention(c.q, c.k, c.v, c.physical, impl="torch_native")
    torch.testing.assert_close(compact, reference, atol=0.004, rtol=0.02)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        captured = gbsa(c, True)
    c.q.mul_(0.7)
    c.v.mul_(0.5)
    # A new valid selection in the same buffers must be observed on replay.
    # Sort inside GBSA preserves the exact previous reduction order.
    next_groups = c.groups.cpu()
    for row, pos in enumerate(c.positions.cpu().tolist()):
        valid = next_groups[row] >= 0
        # Rotate IDs within the visible full groups. For long prefixes this
        # changes the selected set, as well as its order, without duplication.
        visible = (pos + 1) // 4
        if visible:
            next_groups[row, valid] = (next_groups[row, valid] + 1) % visible
    c.groups.copy_(next_groups.npu())
    c.logical.copy_(expand_qsa_block_indices(c.groups, c.positions, c.lengths[c.sequence_ids.long()], 4, 2048))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, gbsa(c, False), atol=0, rtol=0)


@pytest.mark.parametrize("rows,num_keys", [(128, 0), (128, 64), (128, 600)])
def test_prefill_producer_keeps_groups(rows, num_keys):
    indexer = SimpleNamespace(token_topk=2048, compress_ratio=4, block_topk=512)
    torch.manual_seed(71)
    q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
    keys = torch.randn(num_keys, 1, 128, device="npu", dtype=torch.bfloat16)
    positions = torch.arange(num_keys * 4, num_keys * 4 + rows, device="npu", dtype=torch.int32)
    lengths = positions + 1
    starts = torch.zeros(rows, device="npu", dtype=torch.int32)
    ends = torch.full_like(starts, num_keys)
    args = (indexer, q, keys, starts, ends, positions, lengths)
    compact = QSAIndexer.select_prefill_tokens(*args, defer_expansion=True)
    legacy = QSAIndexer.select_prefill_tokens(*args)
    assert compact.shape == (rows, 512)
    torch.testing.assert_close(expand_qsa_block_indices(compact, positions, lengths, 4, 2048), legacy, atol=0, rtol=0)


@pytest.mark.parametrize("rows,enabled,padded", [(4, True, 0), (128, False, 0), (128, True, 4)])
def test_attention_consumer_fallback_and_padding(rows, enabled, padded, monkeypatch):
    c = make_case([rows], [2051])
    monkeypatch.setenv("SGLANG_NPU_QSA_PREFILL_GBSA", "1" if enabled else "0")
    layer_cache = SimpleNamespace(key=lambda: c.k.unsqueeze(1), value=lambda: c.v.unsqueeze(1))
    pool = SimpleNamespace(qsa_compressed_page_size=32, layer_router=SimpleNamespace(read=lambda _: layer_cache))
    idx_meta = QSAIndexerMetadata(
        sequence_lengths=c.lengths, token_to_batch_idx=c.sequence_ids, token_slot_table=c.table,
        out_cache_loc=torch.zeros(rows, device="npu", dtype=torch.int32), token_to_kv_pool=pool,
        compress_ratio=4, block_topk=512, defer_prefill_block_expansion=True,
    )
    meta = QwenSparseAttnMetadata(sequence_lengths=c.lengths, token_to_batch_idx=c.sequence_ids, token_slot_table=c.table, indexer_metadata=idx_meta)
    backend = SimpleNamespace(token_to_kv_pool=pool, compress_ratio=4, qsa_profile=SimpleNamespace(block_topk=512, budget=2048),
                              _resolve_metadata=lambda _: meta, _is_speculative_paged_mode=lambda _: False,
                              _logical_to_physical=QwenSparseAttnBackend._logical_to_physical,
                              _pad_extend_output=QwenSparseAttnBackend._pad_extend_output)
    backend._uses_block_indices = MethodType(QwenSparseAttnBackend._uses_block_indices, backend)
    backend._expand_block_indices = MethodType(QwenSparseAttnBackend._expand_block_indices, backend)
    backend._npu_indices_to_physical = MethodType(QwenSparseAttnBackend._npu_indices_to_physical, backend)
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=c.positions, extend_seq_lens=c.extends)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=3, head_dim=256, scaling=1 / 16)
    q = torch.cat([c.q, c.q.new_zeros(padded, 3, 256)])
    actual = QwenSparseAttnBackend.forward_extend(backend, q, None, None, layer, batch, save_kv_cache=False, topk_indices=c.groups)
    expected = QwenSparseAttnBackend.forward_extend(backend, q, None, None, layer, batch, save_kv_cache=False, topk_indices=c.logical)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if padded:
        assert not actual[-padded:].count_nonzero().item()


def test_draft_prefill_retains_mtp_token_contract(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_QSA_PREFILL_GBSA", "1")
    backend = SimpleNamespace(
        _mtp_shared_sparse_indices=None,
        compress_ratio=4,
        token_to_kv_pool=SimpleNamespace(qsa_block_topk=512, qsa_compressed_page_size=32),
    )
    eligible = MethodType(QwenSparseAttnBackend._can_defer_prefill_block_expansion, backend)
    assert eligible(ForwardMode.EXTEND)
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        assert not eligible(mode)
    # A draft's ordinary EXTEND also captures the next iteration's token IDs.
    # Enabling compact groups here would silently seed 512 group IDs into the
    # 2051-wide token buffer, corrupting the following draft decode selection.
    backend._mtp_shared_sparse_indices = object()
    assert not eligible(ForwardMode.EXTEND)
