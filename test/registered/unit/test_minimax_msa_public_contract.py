from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_ops import msa


class _RecordingGraphState:
    def __init__(self):
        self.workspace = object()
        self.staged = {}

    def stage(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.staged[name] = value
        return value


def _production_inputs():
    batch_size = 2
    total_q = 4
    num_q_heads = 16
    num_kv_heads = 1
    head_dim = 128
    page_size = 128
    num_pages = 2
    q = torch.zeros(total_q, num_q_heads, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(
        num_pages * page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v = torch.zeros_like(k)
    q2k = torch.zeros(num_kv_heads, total_q, 16, dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([128, 256], dtype=torch.int32)
    return q, k, v, q2k, page_table, seq_lens


def test_topk16_only_provider_selection(monkeypatch):
    monkeypatch.setenv("SGLANG_MINIMAX_MSA_BACKEND", "auto")
    monkeypatch.setattr(msa, "flashinfer_msa_available", lambda: True)
    monkeypatch.setattr(msa, "fmha_sm100_available", lambda: True)

    assert msa.selected_msa_backend(torch.bfloat16, 1, 16) == "flashinfer"
    assert msa.selected_msa_backend(torch.bfloat16, 1, 8) == "fmha_sm100"

    monkeypatch.setenv("SGLANG_MINIMAX_MSA_BACKEND", "flashinfer")
    with pytest.raises(msa.MSAUnavailableError, match="top-k 16"):
        msa.selected_msa_backend(torch.bfloat16, 1, 8)


def test_prefill_forwards_tp4_public_contract(monkeypatch):
    q, k, v, q2k, page_table, seq_lens = _production_inputs()
    public_prefill = Mock(return_value=q)
    monkeypatch.setattr(
        msa,
        "_load_flashinfer_msa",
        lambda: (public_prefill, Mock(), Mock(), Mock()),
    )
    graph_state = _RecordingGraphState()
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    q_offset = torch.tensor([126, 254], dtype=torch.int32)

    out = msa._flashinfer_prefill(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=q2k,
        req_to_token=torch.empty(2, 256, dtype=torch.int32),
        slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens=cu_q,
        seq_lens=seq_lens,
        prefix_lens=q_offset,
        block_size_k=128,
        sm_scale=128**-0.5,
        page_table=page_table,
        graph_state=graph_state,
    )

    assert out is q
    call = public_prefill.call_args.kwargs
    assert call["q"].shape == (4, 16, 128)
    assert call["k"].shape == call["v"].shape == (2, 1, 128, 128)
    assert call["q2k_indices"].shape == (1, 4, 16)
    assert call["cu_seqlens_q"].shape == (3,)
    assert call["page_table"].shape == (2, 2)
    assert call["seqused_k"].shape == call["q_offset"].shape == (2,)
    assert call["workspace"] is graph_state.workspace
    assert set(graph_state.staged) == {
        "q",
        "q2k_indices",
        "cu_seqlens_q",
        "seqused_k",
        "q_offset",
    }


def test_decode_forwards_tp4_public_contract(monkeypatch):
    q, k, v, q2k, page_table, seq_lens = _production_inputs()
    q = q[:2]
    q2k = q2k[:, :2]
    public_decode = Mock(return_value=q)
    monkeypatch.setattr(
        msa,
        "_load_flashinfer_msa",
        lambda: (Mock(), public_decode, Mock(), Mock()),
    )
    graph_state = _RecordingGraphState()

    out = msa._flashinfer_decode(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=q2k,
        req_to_token=torch.empty(2, 256, dtype=torch.int32),
        slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_lens,
        block_size_k=128,
        sm_scale=128**-0.5,
        page_table=page_table,
        graph_state=graph_state,
    )

    assert out is q
    call = public_decode.call_args.kwargs
    assert call["q"].shape == (2, 16, 128)
    assert call["k"].shape == call["v"].shape == (2, 1, 128, 128)
    assert call["q2k_indices"].shape == (1, 2, 16)
    assert call["page_table"].shape == (2, 2)
    assert call["seqused_k"].shape == (2,)
    assert call["seqlen_q"] == 1
    assert call["workspace"] is graph_state.workspace
    assert set(graph_state.staged) == {"q", "q2k_indices", "seqused_k"}


def test_graph_state_and_page_table_keep_stable_addresses(monkeypatch):
    workspace = object()
    monkeypatch.setattr(
        msa,
        "_load_flashinfer_msa",
        lambda: (Mock(), Mock(), Mock(), lambda _device: workspace),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    state = msa.FlashInferMSAGraphState("cpu")
    first = state.stage("q", torch.ones(2, 16, 128, dtype=torch.bfloat16))
    address = first.data_ptr()
    second = state.stage("q", torch.zeros_like(first))
    assert state.workspace is workspace
    assert second.data_ptr() == address
    assert torch.count_nonzero(second) == 0

    req_to_token = torch.stack(
        (torch.arange(128, 384), torch.arange(512, 768))
    ).to(torch.int32)
    page_table = torch.full((2, 2), -1, dtype=torch.int32)
    result = msa.build_flashinfer_page_table(
        req_to_token,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([256, 256], dtype=torch.int32),
        128,
        out=page_table,
    )
    assert result.data_ptr() == page_table.data_ptr()
    assert result.tolist() == [[1, 2], [4, 5]]
