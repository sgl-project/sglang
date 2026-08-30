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
    backend.dcp_size = 1
    backend.dcp_rank = 0
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = speculative_num_draft_tokens
    backend.expand_encoder_only_verify = False
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


@pytest.mark.parametrize(
    ("forward_mode", "seq_lens", "q_len", "expected_prefix", "expected_total"),
    [
        (ForwardMode.TARGET_VERIFY, [9, 17], 4, [9, 17], [13, 21]),
        (ForwardMode.DRAFT_EXTEND_V2, [13, 21], 4, [9, 17], [13, 21]),
    ],
)
def test_dcp_spec_metadata_keeps_global_prefix_and_local_total(
    forward_mode, seq_lens, q_len, expected_prefix, expected_total
):
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.dcp_size = 4
    backend.dcp_rank = 2
    backend.speculative_step_id = 0
    backend.page_size = PAGE_SIZE
    backend.max_num_pages = 2
    backend._swa_kv_pool = None
    backend.expand_encoder_only_verify = False
    backend.use_sliding_window_kv_pool = False
    backend._fill_page_table_device = lambda *_args, **_kwargs: None
    backend._maybe_build_cp_zigzag_page_tables = lambda *_args, **_kwargs: None

    batch_size = len(seq_lens)
    spec_info = SimpleNamespace(
        num_tokens_per_req=q_len,
        ragged_verify_layout=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        forward_mode=forward_mode,
        spec_info=spec_info,
        input_ids=torch.zeros(batch_size * q_len, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
        out_cache_loc=None,
    )

    backend.init_forward_metadata(forward_batch)
    metadata = backend.forward_metadata
    expected_total = torch.tensor(expected_total, dtype=torch.int32)
    expected_local = expected_total // backend.dcp_size + (
        backend.dcp_rank < expected_total % backend.dcp_size
    )
    torch.testing.assert_close(
        metadata.causal_seqlens_kv_global,
        torch.tensor(expected_prefix, dtype=torch.int32),
    )
    torch.testing.assert_close(metadata.cache_seqlens_int32, expected_local)
    assert metadata.max_seq_len_q == q_len
    torch.testing.assert_close(
        metadata.cu_seqlens_q,
        torch.arange(0, batch_size * q_len + 1, q_len, dtype=torch.int32),
    )


@pytest.mark.parametrize("q_len", [1, 4])
def test_dcp_spec_decode_forwards_cake_contract_and_base2_lse(monkeypatch, q_len):
    kernel_calls = []
    merge_calls = []

    def fake_decode(**kwargs):
        kernel_calls.append(kwargs)
        query = kwargs["query"]
        return (
            torch.zeros_like(query),
            torch.zeros(query.shape[:2], dtype=torch.float32),
        )

    def fake_merge(out, lse, group, **kwargs):
        merge_calls.append((lse, group, kwargs))
        return out[:, : out.shape[1] // group.world_size]

    monkeypatch.setattr(
        trtllm_mha_backend,
        "flashinfer",
        SimpleNamespace(
            decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache=fake_decode)
        ),
    )
    monkeypatch.setattr(trtllm_mha_backend, "cp_lse_ag_out_rs_mha", fake_merge)

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.dcp_size = 4
    backend.dcp_rank = 1
    backend.dcp_group = SimpleNamespace(world_size=4)
    backend.dcp_max_context_len = 32768
    backend.max_context_len = 131072
    backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
    backend.q_data_type = torch.bfloat16
    backend._multi_ctas_kv_counter_buffer = torch.zeros(1, dtype=torch.int32)
    backend.decode_seq_len_splits = 8
    backend.dcp_cuda_graph_out_buffer = None
    backend.dcp_cuda_graph_lse_buffer = None

    batch_size, num_heads, head_dim = 2, 16, 256
    query = torch.zeros(batch_size * q_len, num_heads, head_dim, dtype=torch.bfloat16)
    causal_prefix = torch.tensor([1024, 2048], dtype=torch.int32)
    output = backend._run_fixed_q_len_decode(
        query,
        (torch.empty(0), torch.empty(0)),
        torch.zeros(batch_size, 1, dtype=torch.int32),
        torch.tensor([257, 513], dtype=torch.int32),
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        window_left=-1,
        sinks=None,
        q_len_per_req=q_len,
        causal_seqlens_kv_global=causal_prefix,
    )

    assert output.shape == (batch_size * q_len, num_heads // 4, head_dim)
    assert len(kernel_calls) == 1
    call = kernel_calls[0]
    assert call["cp_world"] == 4
    assert call["cp_rank"] == 1
    assert call["causal_seqlens_kv_global"] is causal_prefix
    assert call["q_len_per_req"] == q_len
    assert call["max_seq_len"] == backend.dcp_max_context_len
    assert call["return_lse"] is True
    assert call["skip_softmax_threshold_scale_factor"] is None
    assert len(merge_calls) == 1
    assert merge_calls[0][2] == {"is_lse_base_on_e": False}


def test_dcp_spec_q_gather_reuses_graph_stable_buffer():
    class FakeGroup:
        def all_gather_into_tensor(self, output, query):
            rows = query.shape[0]
            for rank in range(4):
                output[rank * rows : (rank + 1) * rows].copy_(query + rank * 100)

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.dcp_size = 4
    backend.dcp_group = FakeGroup()
    backend.num_q_heads = 2
    backend.head_dim = 3
    backend.dcp_cuda_graph_q_gather_buffer = torch.empty(6 * 4, 2, 3)
    backend.dcp_cuda_graph_q_buffer = torch.empty(6, 8, 3)

    query = torch.arange(2 * 2 * 3, dtype=torch.float32).view(2, 2, 3)
    stable_query = backend._all_gather_dcp_spec_q(query)
    expected = torch.cat([query + rank * 100 for rank in range(4)], dim=1)
    torch.testing.assert_close(stable_query, expected)

    stable_ptr = stable_query.data_ptr()
    stable_query = backend._all_gather_dcp_spec_q(query + 1)
    assert stable_query.data_ptr() == stable_ptr
    torch.testing.assert_close(
        stable_query,
        torch.cat([query + 1 + rank * 100 for rank in range(4)], dim=1),
    )


def test_hybrid_wrappers_forward_in_graph_hook():
    # The hybrid backend reads the mode from the published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="decode")
    override.install()
    try:
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
                model_config=SimpleNamespace(context_len=2048),
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
    finally:
        override.restore()


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


def test_graph_read_done_event_fences_slot_mutation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    backend = _make_backend_for_hook_test()
    backend.device = torch.device(DEVICE)
    backend.page_size = 2
    backend.max_num_pages = 2
    backend.use_sliding_window_kv_pool = True
    backend._swa_kv_pool = object()
    backend.req_to_token = torch.tensor(
        [[0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.int32, device=DEVICE
    )
    backend._swa_full_to_swa_mapping = (
        torch.arange(32, dtype=torch.int64, device=DEVICE) * 2
    )
    backend.init_cuda_graph_state(max_bs=1, max_num_tokens=1)
    forward_batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([4], dtype=torch.int32, device=DEVICE),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        positions=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        out_cache_loc=torch.tensor([3], dtype=torch.int64, device=DEVICE),
    )

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    backend.init_forward_metadata_in_graph(forward_batch)
    torch.cuda.synchronize()

    read_done = torch.cuda.Event(external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.init_forward_metadata_in_graph(forward_batch)
        read_done.record()

    fence_stream = torch.cuda.Stream()
    mutation_done = torch.cuda.Event()
    graph.replay()
    with torch.cuda.stream(fence_stream):
        fence_stream.wait_event(read_done)
        backend.req_to_token.copy_(
            torch.tensor(
                [[8, 9, 10, 11], [0, 1, 2, 3]],
                dtype=torch.int32,
                device=DEVICE,
            )
        )
        backend._swa_full_to_swa_mapping.add_(64)
        mutation_done.record()
    torch.cuda.current_stream().wait_event(mutation_done)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        backend.forward_metadata.page_table,
        torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(
        backend.forward_metadata.swa_page_table,
        torch.tensor([[0, 2]], dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(
        backend.forward_metadata.swa_out_cache_loc,
        torch.tensor([6], dtype=torch.int64, device=DEVICE),
    )

    graph.replay()
    with torch.cuda.stream(fence_stream):
        fence_stream.wait_event(read_done)
        backend.req_to_token.copy_(
            torch.tensor(
                [[0, 1, 2, 3], [8, 9, 10, 11]],
                dtype=torch.int32,
                device=DEVICE,
            )
        )
        backend._swa_full_to_swa_mapping.sub_(64)
        mutation_done.record()
    torch.cuda.current_stream().wait_event(mutation_done)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        backend.forward_metadata.page_table,
        torch.tensor([[4, 5]], dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(
        backend.forward_metadata.swa_page_table,
        torch.tensor([[40, 42]], dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(
        backend.forward_metadata.swa_out_cache_loc,
        torch.tensor([70], dtype=torch.int64, device=DEVICE),
    )


def test_swa_cache_write_uses_metadata_slot_snapshot():
    snapshot = torch.tensor([6], dtype=torch.int64)

    def translate_live_mapping(_):
        raise AssertionError("cache writes must not read the live SWA mapping")

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend._swa_kv_pool = SimpleNamespace(
        layers_mapping={1: (0, True)},
        translate_loc_from_full_to_swa=translate_live_mapping,
    )
    backend.forward_metadata = SimpleNamespace(swa_out_cache_loc=snapshot)
    forward_batch = SimpleNamespace(out_cache_loc=torch.tensor([3], dtype=torch.int64))

    cache_loc = backend._get_layer_cache_loc(SimpleNamespace(layer_id=1), forward_batch)

    torch.testing.assert_close(cache_loc, snapshot, rtol=0, atol=0)


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
# max_num_pages upper bound, not a per-batch dynamic width. The kernel self-guards
# on the device-side seqlen, so the checks compare only the live prefix per row.
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


def test_dcp_metadata_uses_local_lens_and_page_table():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    bs = 3
    dcp_size = 4
    dcp_rank = 2
    global_lens = torch.tensor([1, 257, 1027], dtype=torch.int32, device=DEVICE)
    max_local_len = (int(global_lens.max().item()) + dcp_size - 1) // dcp_size
    max_seq_pages = (max_local_len + PAGE_SIZE - 1) // PAGE_SIZE
    req_to_token = torch.arange(bs * 2048, dtype=torch.int32, device=DEVICE).reshape(
        bs, 2048
    )
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=DEVICE)
    cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    causal_seqlens_kv_global = torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    page_table = torch.full((bs, max_seq_pages), -1, dtype=torch.int32, device=DEVICE)

    update_trtllm_mha_graph_metadata(
        req_pool_indices=req_pool_indices,
        seq_lens=global_lens,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table=page_table,
        bs=bs,
        seqlen_offset=0,
        max_seq_pages=max_seq_pages,
        page_size=PAGE_SIZE,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        causal_seqlen_offset=-1,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    torch.cuda.synchronize()

    local_lens = (global_lens // dcp_size + (dcp_rank < global_lens % dcp_size)).to(
        torch.int32
    )
    torch.testing.assert_close(cache_seqlens, local_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        causal_seqlens_kv_global, global_lens - 1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cu_seqlens_k[1:],
        torch.cumsum(local_lens, dim=0, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    for req_idx in range(bs):
        num_pages = (int(local_lens[req_idx].item()) + PAGE_SIZE - 1) // PAGE_SIZE
        global_positions = (
            torch.arange(num_pages, device=DEVICE, dtype=torch.int64)
            * PAGE_SIZE
            * dcp_size
            + dcp_rank
        )
        expected = (req_to_token[req_idx, global_positions] // dcp_size) // PAGE_SIZE
        torch.testing.assert_close(
            page_table[req_idx, :num_pages], expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
