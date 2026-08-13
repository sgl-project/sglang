from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.page_table import (
    build_page_table_snapshot,
)
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _row_from_pages(pages: list[int], page_size: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                page * page_size,
                (page + 1) * page_size,
                dtype=torch.int32,
                device="cuda",
            )
            for page in pages
        ]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="page-table snapshot requires CUDA"
)
def test_page_table_snapshot_uses_active_rows_and_draft_delta():
    page_size = 4
    max_slots = 64
    req_to_token = torch.empty((3, 16), dtype=torch.int32, device="cuda")
    req_to_token[0] = _row_from_pages([2, 7, 4, 9], page_size)
    req_to_token[1] = _row_from_pages([6, 8, 11, 0], page_size)
    req_to_token[2] = _row_from_pages([5, 1, 10, 3], page_size)

    req_pool_indices = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([5, 9], dtype=torch.int32, device="cuda")
    snapshot = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    original_ptr = snapshot.data_ptr()

    build_page_table_snapshot(
        snapshot,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_size,
        max_slots,
        seq_len_delta=3,
    )

    assert snapshot.data_ptr() == original_ptr
    torch.testing.assert_close(
        snapshot.cpu(),
        torch.tensor([[5, 1, 0, 0], [2, 7, 4, 0]], dtype=torch.int32),
    )


def test_backend_slices_max_batch_buffers_before_snapshot(monkeypatch):
    captured = {}

    def fake_snapshot(
        page_table,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_size,
        max_slots,
        *,
        seq_len_delta=0,
    ):
        captured.update(
            page_table=page_table,
            req_pool_indices=req_pool_indices.clone(),
            seq_lens=seq_lens.clone(),
            page_size=page_size,
            max_slots=max_slots,
            seq_len_delta=seq_len_delta,
        )

    monkeypatch.setattr(
        "sglang.kernels.ops.attention.minimax_sparse.page_table."
        "build_page_table_snapshot",
        fake_snapshot,
    )

    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.is_npu = False
    backend._msa_dec_meta = None
    backend._active_page_table = None
    backend._cuda_graph_page_table = torch.empty((4, 4), dtype=torch.int32)
    backend._msa_owns_decode = False
    backend.req_to_token = torch.zeros((4, 16), dtype=torch.int32)
    backend.kv_pool = SimpleNamespace(size=60)
    backend.page_size = 4
    backend.max_context_len = 16
    backend.speculative_num_draft_tokens = 8
    backend.speculative_num_steps = 7

    mode = SimpleNamespace(
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
        is_decode_or_idle=lambda: True,
    )
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        batch_size=2,
        req_pool_indices=torch.tensor([2, 1, 99, 99], dtype=torch.int64),
        seq_lens=torch.tensor([8, 7, 999, 999], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 7, 999, 999], dtype=torch.int32),
        extend_seq_lens_cpu=None,
    )

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=False)

    assert backend._active_page_table.shape == (2, 4)
    assert captured["page_table"].shape == (2, 2)
    torch.testing.assert_close(
        captured["req_pool_indices"], torch.tensor([2, 1], dtype=torch.int64)
    )
    torch.testing.assert_close(
        captured["seq_lens"], torch.tensor([8, 7], dtype=torch.int32)
    )
    assert backend._max_seqlen_k == 8

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    assert backend._active_page_table.shape == (2, 4)
    assert captured["page_table"].shape == (2, 4)
    assert backend._max_seqlen_k == 16


def test_full_prefill_capture_preserves_page_table_backing_for_decode(monkeypatch):
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.minimax_sparse.page_table."
        "build_page_table_snapshot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.minimax_sparse.prefill.scheduler."
        "build_query_block_to_req",
        lambda *args, **kwargs: None,
    )

    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.is_npu = False
    backend._msa_dec_meta = None
    backend._active_page_table = None
    backend._cuda_graph_page_table = None
    backend._active_query_block_to_req = None
    backend._full_cg_query_block_to_req = None
    backend._decode_cuda_graph_max_bs = 4
    backend._full_cg_query_block_capacity = 2
    backend._msa_owns_decode = False
    backend.req_to_token = torch.zeros((4, 16), dtype=torch.int32)
    backend.kv_pool = SimpleNamespace(size=60)
    backend.page_size = 4
    backend.max_context_len = 16
    backend.block_size_q = 1

    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.empty(2, dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 8], dtype=torch.int32),
        extend_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        extend_seq_lens_cpu=[1, 1],
    )

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    captured_ptr = backend._active_page_table.data_ptr()
    packed_query_ptr = backend._active_query_block_to_req.data_ptr()
    assert backend._cuda_graph_page_table.shape == (4, 4)

    backend.init_cuda_graph_state(max_bs=4, max_num_tokens=4)
    assert backend._cuda_graph_page_table.data_ptr() == captured_ptr
    assert backend._full_cg_query_block_to_req.data_ptr() == packed_query_ptr


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="page-table replay requires CUDA"
)
def test_full_prefill_graph_replay_reads_updated_sparse_metadata():
    page_size = 4
    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.is_npu = False
    backend._msa_dec_meta = None
    backend._active_page_table = None
    backend._cuda_graph_page_table = None
    backend._active_query_block_to_req = None
    backend._full_cg_query_block_to_req = None
    backend._decode_cuda_graph_max_bs = 2
    backend._full_cg_query_block_capacity = 8
    backend._msa_owns_decode = False
    backend.req_to_token = torch.stack(
        [
            _row_from_pages([2, 7], page_size),
            _row_from_pages([5, 1], page_size),
        ]
    )
    backend.kv_pool = SimpleNamespace(size=32)
    backend.page_size = page_size
    backend.max_context_len = 8
    backend.block_size_q = 1

    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.empty(8, dtype=torch.int32, device="cuda"),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        seq_lens=torch.tensor([8, 0], dtype=torch.int32, device="cuda"),
        seq_lens_cpu=torch.tensor([8, 0], dtype=torch.int32),
        extend_seq_lens=torch.tensor([8, 0], dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=[8, 0],
    )
    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    captured_ptr = backend._active_page_table.data_ptr()
    packed_query_ptr = backend._active_query_block_to_req.data_ptr()

    replay_output = torch.empty((2, 1), dtype=torch.int32, device="cuda")
    replay_query_schedule = torch.empty(8, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output.copy_(backend._active_page_table[:, :1])
        replay_query_schedule.copy_(backend._active_query_block_to_req)

    forward_batch.seq_lens.copy_(torch.tensor([4, 4], device="cuda"))
    forward_batch.seq_lens_cpu.copy_(torch.tensor([4, 4]))
    forward_batch.extend_seq_lens.copy_(torch.tensor([4, 4], device="cuda"))
    forward_batch.extend_seq_lens_cpu = [4, 4]
    forward_batch.forward_mode = ForwardMode.MIXED
    backend.init_forward_metadata_out_graph(forward_batch, in_capture=False)
    assert backend._active_page_table.data_ptr() == captured_ptr
    assert backend._active_query_block_to_req is None
    assert backend._full_cg_query_block_to_req.data_ptr() == packed_query_ptr
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        replay_output.cpu(), torch.tensor([[2], [5]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        replay_query_schedule.cpu(),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
    )
