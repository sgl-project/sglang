from __future__ import annotations

import sys
from types import ModuleType

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse


def test_prefill_routes_only_step3_to_sgl_native_q8kv8(monkeypatch):
    q = torch.empty(1, 4, 128)
    k_cache = torch.empty(1, 1, 128)
    v_cache = torch.empty_like(k_cache)
    idx_q = torch.empty(1, 1, 128)
    idx_k_cache = torch.empty(1, 1, 128)
    req_to_token = torch.zeros(1, 1, dtype=torch.int32)
    slot_ids = torch.zeros(1, dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    prefix_lens = torch.zeros(1, dtype=torch.int32)
    topk_idx = torch.zeros(1, 1, 1, dtype=torch.int32)
    idx_output = torch.full((1, 1, 128), 3.0)
    native_output = torch.full((1, 4, 128), 7.0)
    captured = {}

    def fake_index_step(**kwargs):
        captured["index_q"] = kwargs["q"]
        return idx_output, topk_idx

    def fake_native_step(**kwargs):
        captured["native_kwargs"] = kwargs
        return native_output

    def fail_triton_step(**kwargs):
        raise AssertionError("Triton Step 3 must not run after native success")

    fake_module = ModuleType(
        "sglang.srt.layers.attention.minimax_sparse_ops.sgl_native_q8kv8"
    )
    fake_module.SglNativeQ8KV8UnavailableError = RuntimeError
    fake_module.sgl_native_q8kv8_sparse_prefill_main = fake_native_step
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.minimax_sparse_ops.sgl_native_q8kv8",
        fake_module,
    )
    monkeypatch.setattr(
        minimax_sparse, "flash_prefill_with_topk_index", fake_index_step
    )
    monkeypatch.setattr(
        minimax_sparse, "flash_prefill_with_gqa_share_sparse", fail_triton_step
    )

    actual_idx_output, actual_main_output = minimax_sparse.minimax_sparse_prefill(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=None,
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        idx_v_cache=None,
        idx_sink=None,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=1,
        max_seqlen_k=1,
        block_size_q=1,
        block_size_k=128,
        topk=1,
        init_blocks=0,
        local_blocks=1,
        use_sgl_native_q8kv8=True,
        cu_seqblocks_q=torch.tensor([0, 1], dtype=torch.int32),
        max_seqblock_q=1,
        all_seqblock_q=1,
    )

    assert actual_idx_output is idx_output
    assert actual_main_output is native_output
    assert captured["index_q"] is idx_q
    assert captured["native_kwargs"]["topk_idx"] is topk_idx
    assert captured["native_kwargs"]["q"] is q


def test_native_q8kv8_is_opt_in_by_default():
    assert hasattr(envs, "SGLANG_ENABLE_MINIMAX_SGL_NATIVE_Q8KV8")
    assert not envs.SGLANG_ENABLE_MINIMAX_SGL_NATIVE_Q8KV8.get()


def test_prefill_falls_back_when_native_contract_is_unavailable(monkeypatch):
    q = torch.empty(1, 4, 128)
    k_cache = torch.empty(1, 1, 128)
    v_cache = torch.empty_like(k_cache)
    idx_q = torch.empty(1, 1, 128)
    idx_k_cache = torch.empty(1, 1, 128)
    topk_idx = torch.zeros(1, 1, 1, dtype=torch.int32)
    triton_output = torch.full((1, 4, 128), 11.0)

    class ExpectedUnavailableError(RuntimeError):
        pass

    def fake_index_step(**kwargs):
        return torch.empty(1, 1, 128), topk_idx

    def unavailable_native_step(**kwargs):
        raise ExpectedUnavailableError("unsupported test contract")

    def fake_triton_step(**kwargs):
        return triton_output

    fake_module = ModuleType(
        "sglang.srt.layers.attention.minimax_sparse_ops.sgl_native_q8kv8"
    )
    fake_module.SglNativeQ8KV8UnavailableError = ExpectedUnavailableError
    fake_module.sgl_native_q8kv8_sparse_prefill_main = unavailable_native_step
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.minimax_sparse_ops.sgl_native_q8kv8",
        fake_module,
    )
    monkeypatch.setattr(
        minimax_sparse, "flash_prefill_with_topk_index", fake_index_step
    )
    monkeypatch.setattr(
        minimax_sparse, "flash_prefill_with_gqa_share_sparse", fake_triton_step
    )
    monkeypatch.setattr(minimax_sparse, "_sgl_native_q8kv8_fallback_warned", False)

    _, actual_output = minimax_sparse.minimax_sparse_prefill(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=None,
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        idx_v_cache=None,
        idx_sink=None,
        req_to_token=torch.zeros(1, 1, dtype=torch.int32),
        slot_ids=torch.zeros(1, dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        prefix_lens=torch.zeros(1, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=1,
        block_size_q=1,
        block_size_k=128,
        topk=1,
        init_blocks=0,
        local_blocks=1,
        use_sgl_native_q8kv8=True,
        cu_seqblocks_q=torch.tensor([0, 1], dtype=torch.int32),
        max_seqblock_q=1,
        all_seqblock_q=1,
    )

    assert actual_output is triton_output
