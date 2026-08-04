from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.page_table import (
    build_page_table_snapshot,
)
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)


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
    backend._msa_prefill_meta_cache = {}
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


def test_msa_prefill_metadata_is_reused_within_forward(monkeypatch):
    from sglang.srt.layers.attention.minimax_sparse_ops import msa

    calls = {"pack": 0, "plan": 0, "fmha": 0}

    def fake_pack(page_table, seq_lens, page_size):
        calls["pack"] += 1
        return page_table.flatten().to(torch.int32)

    def fake_plan(*args, **kwargs):
        calls["plan"] += 1
        return object()

    def fake_fmha(q, k, v, plan, **kwargs):
        calls["fmha"] += 1
        return q, None

    monkeypatch.setattr(msa, "_pack_page_table", fake_pack)
    monkeypatch.setattr(msa, "_run_fmha_sm100_plan", fake_plan)
    monkeypatch.setattr(msa, "_load_fmha_sm100", lambda: (fake_fmha, None))

    q = torch.zeros(4, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(256, 1, 128, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    topk_idx = torch.zeros(1, 4, 1, dtype=torch.int32)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    seq_lens = torch.tensor([132], dtype=torch.int32)
    prefix_lens = torch.tensor([128], dtype=torch.int32)
    cache = {}

    kwargs = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=topk_idx,
        page_table=page_table,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        block_size_k=128,
        meta_cache=cache,
    )
    msa.msa_sparse_prefill_main(**kwargs)
    msa.msa_sparse_prefill_main(**kwargs)
    assert calls == {"pack": 1, "plan": 1, "fmha": 2}

    cache.clear()
    msa.msa_sparse_prefill_main(**kwargs)
    assert calls == {"pack": 2, "plan": 2, "fmha": 3}
