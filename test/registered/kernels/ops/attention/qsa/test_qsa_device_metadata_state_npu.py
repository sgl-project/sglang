"""Actual device metadata and full indexer raw-store/compression integration."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU", allow_module_level=True)

from test_qsa_request_li_npu import make_backend, make_batch, runtime_context  # noqa: F401
from test_qsa_va_state_npu import Pool, indexer_pair
import torch
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode


@pytest.mark.parametrize("mode", [ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2, ForwardMode.DECODE])
@pytest.mark.parametrize("bs", [1, 32])
def test_actual_gpu_layout_and_raw_writer(mode, bs):
    backend, compressed = make_backend()
    raw_storage = Pool(context=16384)
    pool = object.__new__(QSATokenToKVPool)
    pool.__dict__.update(vars(backend.token_to_kv_pool))
    pool.index_state_dtype = torch.bfloat16
    pool.qsa_rope_position_buffer = raw_storage.qsa_rope_position_buffer
    pool.get_qsa_key_state_buffer = raw_storage.get_qsa_key_state_buffer
    pool.get_qsa_compressed_k_buffer = raw_storage.get_qsa_compressed_k_buffer
    backend.token_to_kv_pool = pool
    reference_pool = Pool(context=16384)
    batch = make_batch(bs, 1 if mode.is_decode() else 4)
    batch.forward_mode = mode
    batch.spec_info.extend_seq_lens_cpu = [4] * bs
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    metadata = backend.forward_metadata.indexer_metadata
    assert metadata.npu_unique_live_ring_writes
    assert bool((metadata.req_pool_indices == 0).all())
    rows = batch.input_ids.numel()
    reference = SimpleNamespace(**{name: getattr(metadata, name) for name in metadata.__struct_fields__})
    reference.token_to_kv_pool = reference_pool
    reference.npu_unique_live_ring_writes = False
    cases = indexer_pair(rows, 3)
    token_k = cases[0].qk[:, -128:].view(rows, 1, 128)
    rope = cases[0].positions
    graphs = []
    for i, md in enumerate((reference, metadata)):
        def run():
            cases[i].obj.update_key_state_and_compress(
                token_k, md.decode_logical_positions, rope, md,
                state_slots=md.pending_ring_slots)
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=False):
            run()
        graphs.append(graph)
    for storage in (raw_storage, reference_pool):
        storage.key_state.zero_()
        storage.qsa_rope_position_buffer.zero_()
        storage.compressed.zero_()
    for step in range(6):
        batch.num_padding = (0, 0, bs // 2, bs, 0, 0)[step]
        batch.seq_lens_cpu.copy_(torch.tensor([101 + i % 4 + step * 4 for i in range(bs)], dtype=torch.int32))
        batch.seq_lens.copy_(batch.seq_lens_cpu)
        batch.spec_info.extend_seq_lens_cpu = [1 + (i + step) % 4 for i in range(bs)]
        if step == 4:
            batch.req_pool_indices.copy_(batch.req_pool_indices.roll(1))
            backend.req_to_token.copy_(backend.req_to_token.roll(128, 1))
        if step == 5:
            batch.req_pool_indices[0] = 33
        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        assert backend.forward_metadata.indexer_metadata is metadata
        token_k.add_(0.015625)
        rope.copy_(metadata.decode_logical_positions.expand(3, -1))
        for c in cases:
            c.obj.k_layernorm.weight.add_(0.015625)
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(raw_storage.key_state[4:], reference_pool.key_state[4:], atol=0, rtol=0)
        torch.testing.assert_close(raw_storage.qsa_rope_position_buffer[4:], reference_pool.qsa_rope_position_buffer[4:], atol=0, rtol=0)
        torch.testing.assert_close(raw_storage.compressed[1:], reference_pool.compressed[1:], atol=0, rtol=0)


def test_raw_writer_requires_qualified_layout(monkeypatch):
    backend, _ = make_backend()
    # A non-QSA pool cannot enable device replay merely by setting the env var.
    batch = make_batch(1)
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    assert not backend.forward_metadata.indexer_metadata.npu_unique_live_ring_writes
