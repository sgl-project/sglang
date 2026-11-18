"""ROCm coverage for the QSA packed-varlen decode fallback and indexer scoring.

On HIP there is neither flash_attn (FA2) nor flash-attn-4, so
``_resolve_flash_attn_varlen_func`` resolves to aiter's FA2-compatible varlen
kernel. The first two tests are device-independent (fake modules); the third
runs the real aiter kernel against a torch reference and only executes on ROCm.
The fourth checks the grouped verify scoring of the indexer on gfx950; the last
two check the rows per request the indexer passes it for verify and draft-extend
graphs and the draft-extend graph layout the grouping relies on.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")


def _fake_aiter(varlen_func):
    module = ModuleType("aiter")
    module.flash_attn_varlen_func = varlen_func
    return module


@pytest.mark.parametrize("returns_tuple", [False, True], ids=["tensor", "tuple"])
def test_qsa_hip_resolves_aiter_varlen_kernel(monkeypatch, returns_tuple):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    sentinel = object()
    calls = []

    def aiter_varlen(*args, **kwargs):
        calls.append(kwargs)
        return (sentinel, None) if returns_tuple else sentinel

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", _fake_aiter(aiter_varlen))

    try:
        func = resolver()
        assert func(q=1, causal=True) is sentinel
        assert calls == [{"q": 1, "causal": True}]
    finally:
        resolver.cache_clear()


def test_qsa_hip_without_aiter_falls_through_to_flash_attn(monkeypatch):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    fa2_func = object()
    flash_attn = ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = fa2_func

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", None)  # makes `import aiter` fail
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    try:
        assert resolver() is fa2_func
    finally:
        resolver.cache_clear()


def test_qsa_hip_aiter_varlen_matches_torch_reference():
    if not (is_hip() and torch.cuda.is_available()):
        pytest.skip("ROCm-only kernel")
    pytest.importorskip("aiter")

    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    torch.manual_seed(2028)
    device = torch.device("cuda")
    # Qwen3.8-Flash-Next full-attention shape: 24 query heads, 2 KV heads, d=256.
    batch, topk = 3, 2048
    num_q_heads, num_kv_heads, head_dim = 24, 2, 256
    valid_counts = torch.tensor([5, topk, 700], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.cumsum(valid_counts, 0)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    packed_k = torch.randn(
        int(valid_counts.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    packed_v = torch.randn_like(packed_k)
    scale = head_dim**-0.5

    try:
        # Mirrors the decode call in QwenSparseAttnBackend._forward_paged_attention.
        out = resolver()(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=scale,
            causal=True,
        )
    finally:
        resolver.cache_clear()

    assert out.shape == (batch, num_q_heads, head_dim)
    repeats = num_q_heads // num_kv_heads
    for row in range(batch):
        start, end = int(cu_seqlens_k[row]), int(cu_seqlens_k[row + 1])
        keys = packed_k[start:end].float().repeat_interleave(repeats, dim=1)
        values = packed_v[start:end].float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        expected = torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values)
        torch.testing.assert_close(out[row].float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("requests", [1, 7, 16, 32])
@pytest.mark.parametrize("min_scored_len", [0, 512])
@pytest.mark.parametrize("padded_heads", [8, 16])
def test_qsa_gfx950_grouped_verify_scoring_is_bitwise(
    requests, min_scored_len, padded_heads
):
    """The indexer scores the four verify rows of a request from one key load on
    gfx950; every written logit must equal the per-row kernel's bit for bit."""
    from sglang.srt.layers.attention.qsa.mqa import (
        HAS_TILELANG,
        tilelang_qsa_mqa_decode,
    )
    from sglang.srt.utils import is_gfx95_supported

    if not (HAS_TILELANG and is_gfx95_supported()):
        pytest.skip("gfx950 TileLang-only path")
    torch.manual_seed(requests)
    device = torch.device("cuda")
    rows, max_pages, page_size = 4 * requests, 256, 16
    max_model_len = max_pages * page_size
    # Verify rows of a request share its pages and end one token apart.
    lens = torch.randint(1, max_model_len - 3, (requests,)).repeat_interleave(4)
    lens = (lens + torch.arange(4).repeat(requests)).to(device, torch.int32)
    page_table = torch.randint(0, 1024, (requests, max_pages), dtype=torch.int32)
    page_table = page_table.repeat_interleave(4, dim=0).to(device)
    cache = torch.randn(1024, page_size, 1, 128, device=device).to(torch.bfloat16)
    # The indexer projects 4 heads, zero-padded to the kernel's head alignment.
    q = torch.zeros(rows, padded_heads, 128, device=device, dtype=torch.bfloat16)
    q[:, :4] = torch.randn(rows, 4, 128, device=device).to(torch.bfloat16)

    args = (q, cache, page_table, lens, max_model_len)
    ref = tilelang_qsa_mqa_decode(*args, min_scored_len=min_scored_len)
    out = tilelang_qsa_mqa_decode(
        *args, min_scored_len=min_scored_len, rows_per_request=4, num_query_heads=4
    )
    for row in range(rows):
        length = int(lens[row])
        if length <= min_scored_len:
            continue
        assert torch.equal(
            out[row, :length].view(torch.int32), ref[row, :length].view(torch.int32)
        )


class _ForwardMode:
    def __init__(self, mode):
        self.mode = mode

    def is_decode(self):
        return self.mode == "decode"

    def is_target_verify(self):
        return self.mode == "verify"

    def is_draft_extend_v2(self):
        return self.mode == "draft_extend"


@pytest.mark.parametrize(
    "mode, graph, expected",
    [
        ("draft_extend", True, 4),
        ("draft_extend", False, 1),
        ("verify", True, 4),
        ("verify", False, 1),
        ("decode", True, 1),
    ],
)
def test_qsa_decode_rows_per_request(mode, graph, expected):
    """Verify and draft-extend graphs pass their per-request row count to the
    decode scorer; eager batches and plain decode score row by row."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    rows, seen = 8, []

    def select_decode_tokens(*args, rows_per_request, **kwargs):
        seen.append(rows_per_request)

    indexer = SimpleNamespace(
        layer_id=0,
        index_n_heads=4,
        project_qk=lambda *args, **kwargs: (torch.zeros(rows, 8, 128), None, True),
        update_key_state_and_compress=lambda *args, **kwargs: None,
        select_decode_tokens=select_decode_tokens,
    )
    metadata = SimpleNamespace(
        is_cuda_graph=graph,
        defer_block_expansion=False,
        token_to_kv_pool=None,
        decode_logical_positions=torch.arange(rows),
        pending_ring_slots=torch.zeros(rows, dtype=torch.int64),
        get_token_to_batch_idx=lambda: torch.arange(rows) // 4,
        get_decode_mqa_inputs=lambda layer_id: (None, None, None, 1),
        get_seqlens_int32=lambda: torch.ones(rows, dtype=torch.int32),
    )
    # Each mode's spec info carries only its own field: the draft-extend window
    # width (and logprob rows), or verify's draft_token_num. Decode carries both.
    spec_info = {
        "draft_extend": SimpleNamespace(
            num_tokens_per_req=4, num_tokens_for_logprob_per_req=1
        ),
        "verify": SimpleNamespace(draft_token_num=4),
        "decode": SimpleNamespace(num_tokens_per_req=4, draft_token_num=4),
    }[mode]
    batch = SimpleNamespace(forward_mode=_ForwardMode(mode), spec_info=spec_info)
    QSAIndexer._forward_impl(
        indexer,
        torch.zeros(rows, 16),
        torch.zeros(rows, dtype=torch.int64),
        batch,
        metadata,
    )
    assert seen == [expected]


def test_qsa_draft_extend_graph_layout_groups_rows_by_request():
    """The draft-extend graph layout the grouped decode scorer reads: each request's
    window rows back to back on its request slot, the last row the longest. Request
    0 is shorter than the window; the last two are the graph runners' padding
    requests (slot 0, length 1)."""
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_graph_layout_kernel,
    )

    if not torch.cuda.is_available():
        pytest.skip("Triton kernel")
    torch.manual_seed(0)
    device = torch.device("cuda")
    width, bs, padding = 4, 12, 2
    rows = bs * width
    # Post-extend lengths and request slots.
    seq_lens = torch.randint(width, 70000, (bs,), device=device)
    req_pool = torch.randperm(bs, device=device) + 1
    seq_lens[0] = 2
    seq_lens[-padding:] = 1
    req_pool[-padding:] = 0
    extend_lens = torch.full((bs,), width, dtype=torch.int32, device=device)
    row_seq_lens, row_prefix_lens, row_req_pool = (
        torch.full((rows,), -1, dtype=torch.int32, device=device) for _ in range(3)
    )
    _qsa_graph_layout_kernel[(bs + 1,)](
        seq_lens,
        req_pool,
        extend_lens,
        row_seq_lens,
        row_prefix_lens,
        row_req_pool,
        bs,
        rows,
        0,
        num_warps=1,
    )
    groups = row_seq_lens.view(bs, width)
    assert torch.equal(
        row_req_pool.view(bs, width), req_pool[:, None].expand(bs, width).int()
    )
    assert bool((groups.diff(dim=1) >= 0).all())
    assert torch.equal(groups[:, -1], seq_lens.int())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
