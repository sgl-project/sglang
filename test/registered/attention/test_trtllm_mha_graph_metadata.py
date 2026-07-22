"""Correctness tests for the fused TRTLLM-MHA cuda-graph metadata kernel.

Validates the single-launch triton kernel against a pure-aten reference that
mirrors the exact semantics of the triton port: cache_seqlens / cu_seqlens_k /
cu_seqlens_q (all 3 q-modes) / page_table / swa_page_table / swa_out_cache_loc,
with the SWA -1 sentinel guard.
"""

from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.trtllm_mha_backend as trtllm_mha_backend
from sglang.kernels.ops.kvcache.trtllm_mha_graph_metadata import (
    Q_MODE_CUMSUM,
    Q_MODE_NONE,
    Q_MODE_STRIDED,
    update_trtllm_mha_graph_metadata,
)
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

# trtllm_mha kernels are sm100-only; run this kernel-unit test on Blackwell.
register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

DEVICE = "cuda"
PAGE_SIZE = 128


def _make_backend_for_hook_test(speculative_num_draft_tokens=None):
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.device = torch.device("cpu")
    backend.max_context_len = 1024
    backend.page_size = PAGE_SIZE
    backend.max_num_pages = 8
    backend.req_to_token = torch.zeros(4, 1024, dtype=torch.int32)
    backend.use_sliding_window_kv_pool = False
    backend._swa_kv_pool = None
    backend._swa_full_to_swa_mapping = None
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = speculative_num_draft_tokens
    backend.decode_cuda_graph_metadata = {}
    backend.target_verify_metadata = {}
    backend.draft_extend_metadata = {}
    backend.init_cuda_graph_state(max_bs=4, max_num_tokens=16)
    return backend


def test_cuda_graph_metadata_launch_runs_in_graph_hook(monkeypatch):
    calls = []

    def fake_update(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        trtllm_mha_backend, "update_trtllm_mha_graph_metadata", fake_update
    )
    backend = _make_backend_for_hook_test()
    fb = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.arange(2, dtype=torch.int64),
        seq_lens=torch.ones(2, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        positions=torch.arange(2, dtype=torch.int64),
        out_cache_loc=torch.arange(2, dtype=torch.int64),
    )

    backend.init_forward_metadata_out_graph(fb, in_capture=True)
    assert calls == []
    assert backend.forward_metadata is backend.decode_cuda_graph_metadata[2]

    backend.init_forward_metadata_in_graph(fb)
    assert len(calls) == 1
    assert calls[0]["out_cache_loc"] is fb.out_cache_loc

    calls.clear()
    backend.init_forward_metadata_out_graph(fb)
    assert calls == []
    assert backend.forward_metadata is backend.decode_cuda_graph_metadata[2]


def test_draft_extend_in_graph_uses_captured_static_q_stride(monkeypatch):
    calls = []

    def fake_update(**kwargs):
        calls.append(kwargs)

    class ExplodingAcceptTokens:
        def __getitem__(self, key):
            raise AssertionError("in-graph metadata must not inspect accept tokens")

    monkeypatch.setattr(
        trtllm_mha_backend, "update_trtllm_mha_graph_metadata", fake_update
    )
    backend = _make_backend_for_hook_test(speculative_num_draft_tokens=4)
    fb = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.arange(2, dtype=torch.int64),
        seq_lens=torch.ones(2, dtype=torch.int32),
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        spec_info=SimpleNamespace(
            num_tokens_per_req=4,
            num_accept_tokens=ExplodingAcceptTokens(),
        ),
        positions=torch.arange(8, dtype=torch.int64),
        out_cache_loc=torch.arange(8, dtype=torch.int64),
    )

    backend.init_forward_metadata_out_graph(fb, in_capture=True)
    # The in-graph body must use the captured static stride, not replay-time state.
    fb.spec_info.num_tokens_per_req = 0
    backend.init_forward_metadata_in_graph(fb)

    assert len(calls) == 1
    assert calls[0]["q_mode"] == Q_MODE_STRIDED
    assert calls[0]["q_stride"] == 4


def test_hybrid_wrappers_forward_in_graph_hook():
    """Hybrid wrappers must forward init_forward_metadata_in_graph to the
    wrapped backend(s) — the inherited no-op would leave the fused metadata
    rebuild out of the captured graph (stale page table on every replay)."""
    from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    def make_fake(name, calls):
        return SimpleNamespace(
            token_to_kv_pool=None,
            req_to_token_pool=None,
            needs_cpu_seq_lens=False,
            init_forward_metadata_in_graph=lambda fb: calls.append(name),
        )

    fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    calls = []
    hybrid = HybridAttnBackend(
        SimpleNamespace(
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=None,
            req_to_token_pool=None,
            server_args=SimpleNamespace(speculative_attention_mode="decode"),
        ),
        prefill_backend=make_fake("prefill", calls),
        decode_backend=make_fake("decode", calls),
    )
    hybrid.init_forward_metadata_in_graph(fb)
    assert calls == ["decode"]

    calls = []
    hybrid_linear = HybridLinearAttnBackend(
        full_attn_backend=make_fake("full", calls),
        linear_attn_backend=make_fake("linear", calls),
        full_attn_layers=[0],
    )
    hybrid_linear.init_forward_metadata_in_graph(fb)
    assert calls == ["full", "linear"]


def test_metadata_update_records_inside_cuda_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    backend = _make_backend_for_hook_test()
    backend.device = torch.device(DEVICE)
    backend.page_size = 2
    backend.max_num_pages = 4
    backend.req_to_token = torch.arange(32, dtype=torch.int32, device=DEVICE).reshape(
        4, 8
    )
    backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2)

    fb = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.arange(2, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([3, 4], dtype=torch.int32, device=DEVICE),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        positions=torch.arange(2, dtype=torch.int64, device=DEVICE),
        out_cache_loc=torch.arange(2, dtype=torch.int64, device=DEVICE),
    )

    backend.init_forward_metadata_out_graph(fb, in_capture=True)
    backend.init_forward_metadata_in_graph(fb)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.init_forward_metadata_in_graph(fb)

    fb.seq_lens.copy_(torch.tensor([5, 6], dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        backend.forward_metadata.cache_seqlens_int32,
        torch.tensor([5, 6], dtype=torch.int32, device=DEVICE),
        rtol=0,
        atol=0,
    )


def _build_inputs(bs, pool_size, max_num_pages, max_seq_pages, seq_max, seed):
    """Build random pool / indices / seq_lens consistent with backend buffers."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    req_to_token_stride = max_num_pages * PAGE_SIZE
    # int32 token ids in [0, pool_token_cap); -1 allowed in unused tails.
    pool_token_cap = pool_size * req_to_token_stride
    req_to_token = torch.randint(
        0,
        pool_token_cap,
        (pool_size, req_to_token_stride),
        generator=g,
        dtype=torch.int32,
    ).to(DEVICE)
    req_pool_indices = torch.randperm(pool_size, generator=g)[:bs].to(
        DEVICE, dtype=torch.int64
    )
    seq_lens = torch.randint(1, seq_max + 1, (bs,), generator=g, dtype=torch.int32).to(
        DEVICE
    )
    return req_to_token, req_pool_indices, seq_lens, req_to_token_stride, pool_token_cap


def _ref_cache_seqlens(seq_lens, seqlen_offset):
    return (seq_lens.to(torch.int32) + seqlen_offset).to(torch.int32)


def _ref_page_table(req_to_token, req_pool_indices, max_seq_pages):
    strided = torch.arange(0, max_seq_pages * PAGE_SIZE, PAGE_SIZE, device=DEVICE)
    gathered = req_to_token[req_pool_indices[:, None], strided[None, :]]
    return gathered // PAGE_SIZE, gathered


def _ref_swa_page_table(gathered_tokens, swa_mapping):
    # mimic mapping[-1]=-1 sentinel: token<0 -> -1, else mapping[token]//page
    tok = gathered_tokens.to(torch.int64)
    safe = torch.where(tok >= 0, tok, torch.zeros_like(tok))
    swa_token = swa_mapping[safe]
    swa_token = torch.where(tok >= 0, swa_token, torch.full_like(swa_token, -1))
    swa_page = torch.where(
        swa_token < 0,
        torch.full_like(swa_token, -1),
        swa_token // PAGE_SIZE,
    )
    return swa_page.to(torch.int32)


def _make_swa_mapping(pool_token_cap, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    # Random non-negative SWA pool ids, with a -1 sentinel appended (index -1).
    mapping = torch.randint(
        0, pool_token_cap, (pool_token_cap + PAGE_SIZE + 1,), generator=g
    ).to(DEVICE, dtype=torch.int64)
    mapping[-1] = -1  # sentinel for wrapped -1 index
    return mapping


@pytest.mark.parametrize("bs", [1, 3, 8, 17])
@pytest.mark.parametrize("seqlen_offset", [0, 1, 4])
@pytest.mark.parametrize("q_mode", [Q_MODE_NONE, Q_MODE_CUMSUM, Q_MODE_STRIDED])
@pytest.mark.parametrize("with_swa", [False, True])
# static_width=True exercises the production path: the backend passes the STATIC
# max_num_pages (full upper bound), not a per-batch dynamic width. The kernel
# self-guards on the device-side seqlen: pages past cdiv(cache_seqlen, PAGE_SIZE)
# keep stale values the attention kernel never reads (it is bounded by
# cache_seqlens), so the page-table checks compare the live prefix per row.
@pytest.mark.parametrize("static_width", [False, True])
def test_metadata_correctness(bs, seqlen_offset, q_mode, with_swa, static_width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    seed = (
        1234
        + bs * 31
        + seqlen_offset * 7
        + q_mode * 3
        + int(with_swa)
        + 1000 * int(static_width)
    )
    pool_size = 64
    max_num_pages = 16
    seq_max = (max_num_pages - 2) * PAGE_SIZE  # leave headroom for seqlen_offset
    seq_max = min(seq_max, 1500)

    (
        req_to_token,
        req_pool_indices,
        seq_lens,
        req_to_token_stride,
        pool_token_cap,
    ) = _build_inputs(bs, pool_size, max_num_pages, None, seq_max, seed)

    cache_seqlens_ref = _ref_cache_seqlens(seq_lens, seqlen_offset)
    max_seq_len_k = int(cache_seqlens_ref.max().item())
    if static_width:
        # Production passes the static upper bound (self.max_num_pages), not a
        # dynamic per-batch width — write the whole table every replay.
        max_seq_pages = max_num_pages
    else:
        max_seq_pages = (max_seq_len_k + PAGE_SIZE - 1) // PAGE_SIZE

    # Pre-allocated output buffers (mirror init_cuda_graph_state).
    cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    page_table = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=DEVICE)

    cu_seqlens_q = None
    qlens = None
    q_stride = 0
    if q_mode == Q_MODE_CUMSUM:
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        g = torch.Generator(device="cpu").manual_seed(seed + 99)
        qlens = torch.randint(1, 6, (bs,), generator=g, dtype=torch.int32).to(DEVICE)
    elif q_mode == Q_MODE_STRIDED:
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        q_stride = 4

    swa_mapping = None
    swa_page_table = None
    swa_out_cache_loc = None
    out_cache_loc = None
    if with_swa:
        swa_mapping = _make_swa_mapping(pool_token_cap, seed)
        swa_page_table = torch.zeros(
            bs, max_num_pages, dtype=torch.int32, device=DEVICE
        )
        num_out = bs  # one written token per request (decode-like)
        swa_out_len = num_out + 5  # extra padding tail to validate zero-fill
        swa_out_cache_loc = torch.full(
            (swa_out_len,), 123, dtype=torch.int64, device=DEVICE
        )
        g = torch.Generator(device="cpu").manual_seed(seed + 555)
        out_cache_loc = torch.randint(
            0, pool_token_cap, (num_out,), generator=g, dtype=torch.int64
        ).to(DEVICE)
        # inject a -1 entry to exercise the sentinel path
        out_cache_loc[0] = -1

    update_trtllm_mha_graph_metadata(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table=page_table,
        bs=bs,
        seqlen_offset=seqlen_offset,
        max_seq_pages=max_seq_pages,
        page_size=PAGE_SIZE,
        swa_mapping=swa_mapping,
        swa_page_table=swa_page_table,
        out_cache_loc=out_cache_loc,
        swa_out_cache_loc=swa_out_cache_loc,
        cu_seqlens_q=cu_seqlens_q,
        qlens=qlens,
        q_stride=q_stride,
        q_mode=q_mode,
    )
    torch.cuda.synchronize()

    # ---- cache_seqlens ----
    torch.testing.assert_close(cache_seqlens, cache_seqlens_ref, rtol=0, atol=0)

    # ---- cu_seqlens_k ----
    cu_k_ref = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    cu_k_ref[1:] = torch.cumsum(cache_seqlens_ref, dim=0, dtype=torch.int32)
    torch.testing.assert_close(cu_seqlens_k, cu_k_ref, rtol=0, atol=0)

    # ---- page_table (live [:pages(cache_seqlen)] prefix per row) ----
    pt_ref, gathered = _ref_page_table(req_to_token, req_pool_indices, max_seq_pages)
    live_pages = torch.clamp(
        (cache_seqlens_ref.to(torch.int64) + PAGE_SIZE - 1) // PAGE_SIZE,
        max=max_seq_pages,
    )
    live_mask = torch.arange(max_seq_pages, device=DEVICE).view(
        1, -1
    ) < live_pages.view(-1, 1)
    torch.testing.assert_close(
        page_table[:, :max_seq_pages][live_mask], pt_ref[live_mask], rtol=0, atol=0
    )

    # ---- cu_seqlens_q ----
    if q_mode == Q_MODE_CUMSUM:
        cu_q_ref = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        cu_q_ref[1:] = torch.cumsum(qlens, dim=0, dtype=torch.int32)
        torch.testing.assert_close(cu_seqlens_q, cu_q_ref, rtol=0, atol=0)
    elif q_mode == Q_MODE_STRIDED:
        cu_q_ref = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        cu_q_ref[1:] = (
            torch.arange(1, bs + 1, device=DEVICE, dtype=torch.int32) * q_stride
        )
        torch.testing.assert_close(cu_seqlens_q, cu_q_ref, rtol=0, atol=0)

    # ---- swa_page_table / swa_out_cache_loc ----
    if with_swa:
        swa_pt_ref = _ref_swa_page_table(gathered, swa_mapping)
        torch.testing.assert_close(
            swa_page_table[:, :max_seq_pages][live_mask],
            swa_pt_ref[live_mask],
            rtol=0,
            atol=0,
        )

        # swa_out_cache_loc reference: translate real prefix, zero-fill tail.
        num_out = out_cache_loc.shape[0]
        swa_out_len = swa_out_cache_loc.shape[0]
        num_real = min(num_out, swa_out_len)
        out_ref = torch.zeros(swa_out_len, dtype=torch.int64, device=DEVICE)
        loc = out_cache_loc[:num_real].to(torch.int64)
        safe = torch.where(loc >= 0, loc, torch.zeros_like(loc))
        translated = swa_mapping[safe]
        translated = torch.where(loc >= 0, translated, torch.full_like(translated, -1))
        out_ref[:num_real] = translated
        torch.testing.assert_close(swa_out_cache_loc, out_ref, rtol=0, atol=0)


def test_bs_zero_noop():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # bs == 0 should be a no-op (early return).
    cache_seqlens = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    page_table = torch.zeros(0, 4, dtype=torch.int32, device=DEVICE)
    update_trtllm_mha_graph_metadata(
        req_pool_indices=torch.zeros(0, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.zeros(0, dtype=torch.int32, device=DEVICE),
        req_to_token=torch.zeros(4, 4 * PAGE_SIZE, dtype=torch.int32, device=DEVICE),
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table=page_table,
        bs=0,
        seqlen_offset=0,
        max_seq_pages=0,
        page_size=PAGE_SIZE,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
