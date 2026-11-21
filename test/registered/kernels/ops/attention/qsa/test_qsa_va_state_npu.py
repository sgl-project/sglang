"""Compare native and fused QSA cache updates for live and padded rows."""

import os
from functools import partial
from types import MethodType, SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU die", allow_module_level=True)

from test_qsa_va_norm_npu import make_case, runtime_context  # noqa: F401
import torch
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode


class Pool:
    qsa_compressed_dtype = torch.bfloat16
    index_state_dtype = torch.bfloat16
    qsa_compress_ratio = 4
    qsa_compressed_page_size = 32

    def __init__(self, request_slots=35, context=256):
        self.key_state = torch.zeros(request_slots * 4, 1, 128, device="npu", dtype=torch.bfloat16)
        self.qsa_rope_position_buffer = torch.zeros(request_slots * 4, 3, device="npu", dtype=torch.int64)
        self.compressed = torch.zeros(request_slots * context // 4, 1, 128, device="npu", dtype=torch.bfloat16)

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    set_qsa_key_state_buffer = QSATokenToKVPool.set_qsa_key_state_buffer
    set_qsa_rope_position_buffer = QSATokenToKVPool.set_qsa_rope_position_buffer
    set_qsa_compressed_k_buffer = QSATokenToKVPool.set_qsa_compressed_k_buffer


def indexer_pair(rows, axes):
    cases = [make_case(rows, axes, torch.bfloat16) for _ in range(2)]
    names = ("_use_fused_compress", "_pending_ring_slots", "_group_ring_slots",
             "_rope_from_matrix", "_compress_decode_cuda_graph", "update_key_state_and_compress",
             "_pool_group_keys", "_store_compressed_keys")
    for c in cases:
        c.obj.layer_id = 0
        c.obj.compress_ratio = 4
        c.obj.rotary_emb.get_cos_sin_with_position(c.positions)
        for name in names:
            setattr(c.obj, name, MethodType(getattr(QSAIndexer, name), c.obj))
    return cases


def capture(fn):
    for _ in range(2):
        fn()
        torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        fn()
    return graph


def assert_live_equal(pools, cases):
    a, b = pools
    torch.testing.assert_close(a.key_state, b.key_state, atol=0, rtol=0)
    torch.testing.assert_close(a.qsa_rope_position_buffer, b.qsa_rope_position_buffer, atol=0, rtol=0)
    # Pool contract reserves compressed slot0; native repeated index writes
    # and VA per-row writes do not define its final contents.
    torch.testing.assert_close(a.compressed[1:], b.compressed[1:], atol=0, rtol=0)
    for field in ("position_cos", "position_sin"):
        torch.testing.assert_close(getattr(cases[0].obj.rotary_emb, field),
                                   getattr(cases[1].obj.rotary_emb, field), atol=0, rtol=0)


@pytest.mark.parametrize("raw_state", ["unique_target", "shared_native_state"])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("axes", [1, 3])
def test_actual_graph_state_acceptance_padding_and_slot_reuse(monkeypatch, batch, axes, raw_state):
    rows = batch * 4
    cases = indexer_pair(rows, axes)
    pools = [Pool(), Pool()]
    requests = torch.arange(1, batch + 1, device="npu", dtype=torch.int32)
    seq = torch.ones(rows, device="npu", dtype=torch.int32)
    req_rows = torch.zeros_like(seq)
    position = torch.zeros_like(seq)
    state_slots = torch.zeros(rows, device="npu", dtype=torch.int64)
    groups = torch.zeros(rows, 4, device="npu", dtype=torch.int32)
    write = torch.zeros_like(seq)
    compressed_lens = torch.zeros_like(seq)
    pages = torch.zeros(rows, 2, device="npu", dtype=torch.int32)
    req_table = torch.arange(35 * 256, device="npu", dtype=torch.int32).reshape(35, 256)
    backend = SimpleNamespace(req_to_token=req_table)
    metadata = [SimpleNamespace(
        token_to_kv_pool=pool, is_cuda_graph=True, compress_member_rows=None,
        sequence_lengths=seq, token_to_batch_idx=torch.arange(rows, device="npu", dtype=torch.int32),
        req_pool_indices=req_rows, compress_ratio=4, graph_write_locs=write,
        graph_ring_group_locs=groups, graph_compressed_lengths=compressed_lens,
        graph_compressed_page_table=pages, decode_logical_positions=position,
        pending_ring_slots=state_slots,
    ) for pool in pools]
    token_k = cases[0].qk[:, -128:].reshape(rows, 1, 128)
    rope = cases[0].positions
    graphs = []
    for use_fused, case, meta in zip((False, True), cases, metadata):
        monkeypatch.setattr(qsa_indexer, "_use_va_compress", use_fused)
        update = partial(
            case.obj.update_key_state_and_compress,
            token_k, position, rope, meta,
            state_slots=state_slots,
            state_stored=raw_state == "shared_native_state",
        )
        graphs.append(capture(update))
    for pool in pools:
        pool.key_state.zero_()
        pool.qsa_rope_position_buffer.zero_()
        pool.compressed.zero_()
    # Replay varied request fan-out, accepted lengths, padding aliases, then
    # reuse a request slot with different physical pages and model inputs.
    for step in range(6):
        mode = ForwardMode.TARGET_VERIFY if raw_state == "unique_target" or step % 2 == 0 else ForwardMode.DRAFT_EXTEND_V2
        accepted = [1 + (i + step) % 4 for i in range(batch)]
        spec = SimpleNamespace(draft_token_num=4, extend_seq_lens_cpu=accepted)
        base = torch.tensor([101 + i % 4 + step * 4 for i in range(batch)], dtype=torch.int32)
        if step == 4:
            requests.copy_(requests.roll(1))
            req_table.copy_(req_table.roll(1, 0))
        if step == 5:
            requests[0] = 33
        lengths, mapped, _ = QwenSparseAttnBackend._graph_speculative_layout(
            batch, rows, requests, base, mode, spec,
            num_padding=1 if raw_state == "shared_native_state" and batch > 1 and step == 3 else 0,
        )
        seq.copy_(lengths)
        req_rows.copy_(mapped)
        QwenSparseAttnBackend._update_qsa_cuda_graph_metadata(backend, metadata[0], req_rows)
        token_k.add_(0.015625)
        rope.add_(5).remainder_(65536)
        for c in cases:
            c.obj.k_layernorm.weight.add_(0.015625)
        if raw_state == "shared_native_state":
            # The unchanged native raw writer is nondeterministic for different
            # values aliasing a live slot. Run it once and feed that SAME state
            # to both compression paths; this does not qualify duplicate raw
            # writes as correct. unique_target separately tests full methods.
            pools[0].set_qsa_key_state_buffer(0, state_slots, token_k)
            pools[0].set_qsa_rope_position_buffer(state_slots, rope)
            pools[1].key_state.copy_(pools[0].key_state)
            pools[1].qsa_rope_position_buffer.copy_(pools[0].qsa_rope_position_buffer)
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        assert_live_equal(pools, cases)


@pytest.mark.parametrize("rows", [4, 128, 4096])
def test_actual_eager_prefill_compression(monkeypatch, rows):
    cases = indexer_pair(rows, 3)
    pools = [Pool(request_slots=35, context=4096), Pool(request_slots=35, context=4096)]
    groups = rows // 4
    member_rows = torch.cat([torch.arange(groups, device="npu") * 4, torch.zeros(2, device="npu", dtype=torch.int64)])
    locations = torch.cat([torch.arange(32, 32 + groups, device="npu"), torch.zeros(2, device="npu", dtype=torch.int64)]).int()
    source = cases[0].qk[:, -128:].reshape(rows, 1, 128)
    rope = cases[0].positions
    logical = torch.arange(rows, device="npu", dtype=torch.int32)
    state_slots = logical.long() % 4  # complete prefill groups leave no pending tail
    metadata = [SimpleNamespace(
        token_to_kv_pool=pool, compress_member_rows=member_rows, is_cuda_graph=False,
        write_locs=locations, compress_group_positions=(member_rows + 3).int(),
        extend_rope_matrix=build_rope_position_matrix(rope, rows),
    ) for pool in pools]
    for step in range(3):
        if step:
            source.mul_(-0.75)
            rope.add_(5).remainder_(65536)
            for m in metadata:
                m.extend_rope_matrix.copy_(build_rope_position_matrix(rope, rows))
        with torch.no_grad():
            for i in range(2):
                monkeypatch.setattr(qsa_indexer, "_use_va_compress", bool(i))
                cases[i].obj.update_key_state_and_compress(source, logical, rope, metadata[i], state_slots=state_slots)
        torch.npu.synchronize()
        # Raw reserved ring writes can race; none of these slots is live.
        torch.testing.assert_close(pools[0].compressed[1:], pools[1].compressed[1:], atol=0, rtol=0)
