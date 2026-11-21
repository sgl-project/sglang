"""Actual backend capture/replay and actual indexer LI call, against row LI."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU die", allow_module_level=True)

import sgl_kernel_npu  # noqa: F401
import torch
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def runtime_context():
    torch.npu.set_device(0)
    publish(ServerArgs(model_path="dummy", device="npu", attention_backend="ascend",
                       speculative_num_draft_tokens=4), role="test")
    try:
        yield
    finally:
        reset_context()


def make_backend():
    torch.manual_seed(939)
    capacity, request_slots = 16384, 35
    cache = torch.randn(request_slots * capacity // 4, 1, 128, device="npu", dtype=torch.bfloat16) * 0.1
    cache[:32].zero_()
    pool = SimpleNamespace(
        qsa_compress_ratio=4, qsa_block_topk=512, qsa_index_kv_heads=1,
        qsa_index_head_dim=128, qsa_compressed_page_size=32,
        qsa_compressed_dtype=torch.bfloat16, get_qsa_compressed_k_buffer=lambda _: cache,
    )
    backend = QwenSparseAttnBackend()
    backend.device, backend.token_to_kv_pool = torch.device("npu"), pool
    backend.req_to_token = torch.arange(request_slots * capacity, device="npu", dtype=torch.int32).reshape(request_slots, capacity)
    backend.qsa_profile = QSAProfile(n_heads=4, kv_heads=1, head_dim=128, budget=2048, compress_ratio=4)
    backend.max_context_len = 65536
    backend.init_cuda_graph_state(32, 128)
    return backend, cache


def make_batch(batch_size, width=4):
    lengths = torch.tensor([12000 + i % 4 for i in range(batch_size)], dtype=torch.int32)
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, batch_size=batch_size,
        input_ids=torch.zeros(batch_size * width, dtype=torch.int64, device="npu"),
        req_pool_indices=torch.arange(1, batch_size + 1, dtype=torch.int32, device="npu"),
        seq_lens=lengths.npu(), seq_lens_cpu=lengths,
        spec_info=SimpleNamespace(topk=1, draft_token_num=width), num_padding=0,
    )


def capture_bucket(backend, batch):
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    metadata = backend.forward_metadata.indexer_metadata
    prepared = metadata.npu_lightning_inputs
    assert prepared is not None
    assert prepared.table.shape[0] == batch.batch_size
    assert metadata.graph_compressed_page_table.shape[0] == batch.input_ids.numel()
    q = torch.randn(batch.input_ids.numel(), 5, 128, device="npu", dtype=torch.bfloat16)[:, :4] * 0.1
    indexer = SimpleNamespace(token_topk=2048, block_topk=512, compress_ratio=4)

    def select(inputs):
        k, table, lengths, capacity = metadata.get_decode_mqa_inputs(0)
        return QSAIndexer.select_decode_tokens(
            indexer, q, k, table, lengths, capacity,
            metadata.decode_logical_positions, metadata.sequence_lengths,
            defer_expansion=True, lightning_inputs=inputs,
        )

    for _ in range(2):
        select(prepared)
        select(None)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        actual, expected = select(prepared), select(None)
    return SimpleNamespace(batch=batch, metadata=metadata, prepared=prepared,
                           q=q, graph=graph, actual=actual, expected=expected)


def check_replay(backend, case, step):
    batch = case.batch
    bs = batch.batch_size
    if step == 0:
        batch.seq_lens_cpu.copy_(torch.tensor([12000 + i % 4 for i in range(bs)]))
        batch.num_padding = 0
    elif step == 1:
        batch.seq_lens_cpu.copy_(torch.tensor([2043 + i % 8 for i in range(bs)]))
        batch.num_padding = 0
        batch.req_pool_indices.copy_(batch.req_pool_indices.roll(1))
    elif step == 2:
        batch.seq_lens_cpu.copy_(torch.tensor([i % 4 for i in range(bs)]))
        batch.num_padding = bs // 2
    elif step == 3:
        batch.num_padding = bs
    else:
        batch.num_padding = 0
        batch.seq_lens_cpu.fill_(12007)
        batch.req_pool_indices[0] = 33
    batch.seq_lens.copy_(batch.seq_lens_cpu)
    case.q.mul_(-0.875)
    before_ptrs = [v.data_ptr() for v in (case.prepared.table, case.prepared.lengths,
                                        case.prepared.residual, case.prepared.valid_output)]
    backend.init_forward_metadata_out_graph(batch, in_capture=False)
    assert backend.forward_metadata.indexer_metadata is case.metadata
    assert before_ptrs == [v.data_ptr() for v in (case.prepared.table, case.prepared.lengths,
                                                 case.prepared.residual, case.prepared.valid_output)]
    case.graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(case.actual, case.expected, atol=0, rtol=0)
    visible = case.metadata.graph_compressed_lengths.cpu()
    result = case.actual.cpu()
    valid = result >= 0
    assert bool(((result < visible[:, None]) | ~valid).all())
    torch.testing.assert_close(valid.sum(-1), visible.clamp_max(512).long(), atol=0, rtol=0)
    for row in result:
        assert row[row >= 0].unique().numel() == int((row >= 0).sum())


@pytest.mark.parametrize("batch_size,width", [(1, 4), (3, 4), (32, 4), (4, 1), (4, 2), (4, 3)])
def test_target_backend_updates_match_original_row_selection(batch_size, width):
    backend, cache = make_backend()
    case = capture_bucket(backend, make_batch(batch_size, width))
    for step in range(5):
        if step:
            cache.mul_(0.75)
            # Rotate complete physical pages, retaining the page128 contract.
            backend.req_to_token.copy_(backend.req_to_token.roll(128, dims=1))
        check_replay(backend, case, step)


def test_captured_buckets_have_separate_prepared_views():
    backend, _ = make_backend()
    cases = [capture_bucket(backend, make_batch(bs)) for bs in (1, 32)]
    assert cases[0].prepared.table.data_ptr() != cases[1].prepared.table.data_ptr()
    for step in range(5):
        for case in cases if step % 2 else cases[::-1]:
            check_replay(backend, case, step)


def test_draft_decode_routes_remain_separate():
    backend, _ = make_backend()
    spec = SimpleNamespace(draft_token_num=4)
    for mode in (ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2, ForwardMode.EXTEND):
        assert backend._make_npu_verify_lightning_inputs(mode, spec, 128, 512) is None
