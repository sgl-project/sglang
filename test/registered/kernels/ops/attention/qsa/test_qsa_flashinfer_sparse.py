"""FlashInfer prims-ts QToken sparse decode for QSA (Qwen4-Exp).

``--qsa-decode-attention-backend flashinfer_sparse`` against the QSA Torch
reference, eagerly and under CUDA-graph replay, plus the backend routing.
"""

import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.flashinfer_sparse import (
    FlashInferSparseDecode,
    flashinfer_sparse_supports,
    plan_rows,
    resolve_flashinfer_sparse,
)
from sglang.srt.layers.attention.qsa.kernel import (
    expand_qsa_block_indices,
    qsa_sparse_attention_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mem_cache_utils import routed

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS, KV_HEADS, DIM, RATIO, PAGE, BLOCK_TOPK, BUDGET = 3, 1, 256, 4, 64, 512, 2048
SCALE = DIM**-0.5
LENGTH = 8192
LENS = [1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 2047, 2048, 2049, 2052, 4096, 8191]

needs_sparse = pytest.mark.skipif(
    resolve_flashinfer_sparse() is None,
    reason="needs a compute-capability-10.x GPU and flashinfer.attention.prims_ts",
)


def make_problem(lens, seed):
    """Page-aligned pool with shuffled physical pages, block ids and queries."""
    torch.manual_seed(seed)
    rows = len(lens)
    pages = -(-LENGTH // PAGE)
    # Physical page 0 is the pool's padding page; requests get pages 1.. shuffled.
    page_table = (
        (torch.randperm(rows * pages, device="cuda") + 1).view(rows, pages).int()
    )
    req_to_token = (
        page_table.long()[:, :, None] * PAGE + torch.arange(PAGE, device="cuda")
    ).view(rows, pages * PAGE)
    slots = (rows * pages + 1) * PAGE
    k_cache = torch.randn(slots, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(slots, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    blocks = torch.full((rows, BLOCK_TOPK), -1, device="cuda", dtype=torch.int32)
    for row, n in enumerate(lens):
        chosen = torch.randperm(n // RATIO, device="cuda")[:BLOCK_TOPK]
        blocks[row, : chosen.numel()] = chosen.int()
    q = torch.randn(rows, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    return q, k_cache, v_cache, page_table, req_to_token, seq_lens, blocks


def reference(q, k_cache, v_cache, req_to_token, seq_lens, blocks):
    """Expand the blocks as the default path does, then the Torch reference."""
    logical = expand_qsa_block_indices(blocks, seq_lens - 1, seq_lens, RATIO, BUDGET)
    physical = req_to_token.gather(1, logical.clamp_min(0).long())
    physical = torch.where(logical >= 0, physical, torch.full_like(physical, -1))
    return qsa_sparse_attention_reference(q, k_cache, v_cache, physical, SCALE)


def make_sparse(max_context_len):
    return FlashInferSparseDecode(
        num_qo_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=DIM,
        page_size=PAGE,
        compress_ratio=RATIO,
        block_topk=BLOCK_TOPK,
        max_context_len=max_context_len,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


def assert_matches(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@needs_sparse
# LENGTH: the table is already full width and passes through. 4 * LENGTH: it is
# narrower than the planned context and is copied, like an eager batch table.
@pytest.mark.parametrize("max_context_len", [LENGTH, 4 * LENGTH])
def test_matches_reference(max_context_len):
    q, k, v, page_table, req_to_token, seq_lens, blocks = make_problem(LENS, 1)
    rows = len(LENS)
    sparse = make_sparse(max_context_len)
    out = sparse.forward(
        q.view(rows, -1), k, v, page_table, blocks, seq_lens, SCALE, 0, exact_rows=True
    )
    assert out.shape == (rows, HEADS * DIM)
    assert_matches(
        out.view(rows, HEADS, DIM), reference(q, k, v, req_to_token, seq_lens, blocks)
    )


@needs_sparse
def test_eager_batches_pad_to_plan_rows():
    assert [plan_rows(n) for n in (1, 16, 17, 24, 25, 64, 65, 96, 256, 257, 320)] == [
        1,
        16,
        24,
        24,
        32,
        64,
        96,
        96,
        256,
        320,
        320,
    ]
    sparse = make_sparse(LENGTH)
    for rows in (20, 18):
        lens = (LENS + [100, 200, 300, 1000])[:rows]
        q, k, v, page_table, req_to_token, seq_lens, blocks = make_problem(lens, rows)
        out = sparse.forward(
            q.view(rows, -1), k, v, page_table, blocks, seq_lens, SCALE, 0
        )
        assert out.shape == (rows, HEADS * DIM)
        assert_matches(
            out.view(rows, HEADS, DIM),
            reference(q, k, v, req_to_token, seq_lens, blocks),
        )
    # Both batches shared the one 24-row plan; the shrunken batch's tail rows went inactive.
    assert list(sparse._plans) == [(24, 0)]


@needs_sparse
def test_cuda_graph_replay_follows_live_inputs():
    rows = len(LENS)
    q, k, v, page_table, req_to_token, seq_lens, blocks = make_problem(LENS, 2)
    sparse = make_sparse(LENGTH)
    args = (q.view(rows, -1), k, v, page_table, blocks, seq_lens, SCALE, 0)
    # The eager run plans, compiles and binds the tensor ABI; capture reuses it.
    out = sparse.forward(*args, exact_rows=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sparse.forward(*args, exact_rows=True)
    assert captured.data_ptr() == out.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    assert_matches(
        captured.view(rows, HEADS, DIM),
        reference(q, k, v, req_to_token, seq_lens, blocks),
    )
    # The next decode step: new queries, one more token per request, new block ids.
    q.copy_(torch.randn_like(q))
    seq_lens.copy_((seq_lens + 1).clamp_max(LENGTH))
    for row, n in enumerate(seq_lens.tolist()):
        blocks[row] = -1
        chosen = torch.randperm(n // RATIO, device="cuda")[:BLOCK_TOPK]
        blocks[row, : chosen.numel()] = chosen.int()
    graph.replay()
    torch.cuda.synchronize()
    assert_matches(
        captured.view(rows, HEADS, DIM),
        reference(q, k, v, req_to_token, seq_lens, blocks),
    )


@needs_sparse
def test_layers_get_their_own_plans():
    lens = LENS[:8]
    rows = len(lens)
    q, k0, v0, page_table, req_to_token, seq_lens, blocks = make_problem(lens, 3)
    k1, v1 = torch.randn_like(k0), torch.randn_like(v0)
    sparse = make_sparse(LENGTH)
    for _ in range(2):
        out0 = sparse.forward(
            q.view(rows, -1),
            k0,
            v0,
            page_table,
            blocks,
            seq_lens,
            SCALE,
            0,
            exact_rows=True,
        ).clone()
        out1 = sparse.forward(
            q.view(rows, -1),
            k1,
            v1,
            page_table,
            blocks,
            seq_lens,
            SCALE,
            1,
            exact_rows=True,
        ).clone()
    assert set(sparse._plans) == {(rows, 0), (rows, 1)}
    assert_matches(
        out0.view(rows, HEADS, DIM),
        reference(q, k0, v0, req_to_token, seq_lens, blocks),
    )
    assert_matches(
        out1.view(rows, HEADS, DIM),
        reference(q, k1, v1, req_to_token, seq_lens, blocks),
    )


@needs_sparse
def test_graph_padding_rows_stay_finite():
    """CUDA-graph padding rows carry length 1, request slot 0 and no block ids."""
    lens = LENS[:7] + [1]
    rows = len(lens)
    q, k, v, page_table, req_to_token, seq_lens, blocks = make_problem(lens, 4)
    page_table[-1] = 0
    req_to_token[-1] = torch.arange(req_to_token.shape[1], device="cuda") % PAGE
    blocks[-1] = -1
    sparse = make_sparse(LENGTH)
    out = sparse.forward(
        q.view(rows, -1), k, v, page_table, blocks, seq_lens, SCALE, 0, exact_rows=True
    )
    assert torch.isfinite(out).all()
    assert_matches(
        out.view(rows, HEADS, DIM), reference(q, k, v, req_to_token, seq_lens, blocks)
    )


def test_supported_geometry():
    ok = dict(compress_ratio=4, block_topk=512, head_dim=256, page_size=64)
    assert flashinfer_sparse_supports(**ok)
    assert flashinfer_sparse_supports(**{**ok, "compress_ratio": 8, "page_size": 128})
    assert not flashinfer_sparse_supports(**{**ok, "compress_ratio": 3})
    assert not flashinfer_sparse_supports(**{**ok, "block_topk": 513})
    assert not flashinfer_sparse_supports(**{**ok, "head_dim": 96})
    assert not flashinfer_sparse_supports(**{**ok, "page_size": 62})
    assert not flashinfer_sparse_supports(**{**ok, "compress_ratio": 128})


class _Pool:
    """Stand-in MHA pool: flat ``[slots, Hkv, D]`` buffers and the router's write hook."""

    def __init__(self, cache):
        self.cache = cache
        self.stored = []

    def get_key_buffer(self, layer_id):
        return self.cache

    def get_value_buffer(self, layer_id):
        return self.cache

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        self.stored.append((layer.layer_id, loc, k, v))


class _Sparse:
    num_qo_heads, num_kv_heads, head_dim, page_size = HEADS, KV_HEADS, DIM, PAGE
    dtype = torch.bfloat16

    def __init__(self, marker):
        self.marker = marker
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.marker


def _routing_case(choice):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
    from sglang.srt.model_executor.forward_mode import ForwardMode

    rows = 2
    cache = torch.zeros(4 * PAGE, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    pool = _Pool(cache)
    sparse = _Sparse(
        torch.zeros(rows, HEADS * DIM, device="cuda", dtype=torch.bfloat16)
    )
    backend = module.QwenSparseAttnBackend.__new__(module.QwenSparseAttnBackend)
    backend.qsa_profile = SimpleNamespace(block_topk=BLOCK_TOPK, budget=BUDGET)
    backend.compress_ratio = RATIO
    backend._qsa_decode_attention_backend = choice
    backend._flashinfer_sparse_resolved = True
    backend._flashinfer_sparse = sparse
    backend.token_to_kv_pool = routed(pool)
    lengths = torch.tensor([65, 130], device="cuda", dtype=torch.int32)
    table = torch.tensor([[1, 2, 0], [3, 1, 2]], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=lengths,
        is_cuda_graph=False,
        token_slot_table=None,
        indexer_metadata=SimpleNamespace(decode_page_table=table),
    )
    backend._resolve_metadata = lambda _: metadata
    layer = SimpleNamespace(
        layer_id=3, tp_q_head_num=HEADS, head_dim=DIM, scaling=SCALE
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.zeros(rows, device="cuda", dtype=torch.int32),
        out_cache_loc=torch.tensor([64, 129], device="cuda"),
    )
    q = torch.randn(rows, HEADS * DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    blocks = torch.full((rows, BLOCK_TOPK), -1, device="cuda", dtype=torch.int32)
    return backend, pool, sparse, cache, table, lengths, layer, batch, q, k, v, blocks


def test_backend_routes_decode_to_flashinfer_sparse():
    backend, pool, sparse, cache, table, lengths, layer, batch, q, k, v, blocks = (
        _routing_case("flashinfer_sparse")
    )
    out = backend.forward_decode(q, k, v, layer, batch, topk_indices=blocks)
    assert out is sparse.marker
    # The new K/V reached the pool through the layer router before attention.
    assert [stored[0] for stored in pool.stored] == [3]
    assert torch.equal(pool.stored[0][1], batch.out_cache_loc)
    ((args, kwargs),) = sparse.calls
    assert args[0].shape == (2, HEADS, DIM) and args[1] is cache and args[2] is cache
    assert args[3] is table
    assert torch.equal(args[4], blocks)
    assert args[5] is lengths and args[6] == SCALE and args[7] == 3
    assert kwargs == {"exact_rows": False}


def test_backend_keeps_default_path_unless_selected():
    backend, pool, sparse, *_, layer, batch, q, k, v, blocks = _routing_case("auto")
    assert backend._try_flashinfer_sparse_decode(q, k, v, layer, batch, blocks) is None
    assert sparse.calls == [] and pool.stored == []


def test_backend_declines_token_indices():
    backend, pool, sparse, *_, layer, batch, q, k, v, _ = _routing_case(
        "flashinfer_sparse"
    )
    tokens = torch.full((2, BUDGET + RATIO - 1), -1, device="cuda", dtype=torch.int32)
    assert backend._try_flashinfer_sparse_decode(q, k, v, layer, batch, tokens) is None
    assert sparse.calls == [] and pool.stored == []


def test_resolve_declines_ineligible_pool(caplog):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module

    backend = module.QwenSparseAttnBackend.__new__(module.QwenSparseAttnBackend)
    backend._flashinfer_sparse_resolved = False
    backend._flashinfer_sparse = None
    backend._fused_kv_pool_eligible = False
    backend._kv_shard_pool = None
    backend.qsa_profile = SimpleNamespace(block_topk=BLOCK_TOPK, budget=BUDGET)
    backend.compress_ratio = RATIO
    backend.max_context_len = 1024
    backend.token_to_kv_pool = SimpleNamespace(page_size=PAGE)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=HEADS, head_dim=DIM)
    cache = torch.zeros(PAGE, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING):
        assert backend._resolve_flashinfer_sparse(layer, cache) is None
    assert "does not apply" in caplog.text
    assert backend._flashinfer_sparse_resolved
    assert backend._resolve_flashinfer_sparse(layer, cache) is None


def test_resolve_declines_fp8_pool(caplog):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module

    backend = module.QwenSparseAttnBackend.__new__(module.QwenSparseAttnBackend)
    backend._flashinfer_sparse_resolved = False
    backend._flashinfer_sparse = None
    backend._fused_kv_pool_eligible = True
    backend._kv_shard_pool = None
    backend.qsa_profile = SimpleNamespace(block_topk=BLOCK_TOPK, budget=BUDGET)
    backend.compress_ratio = RATIO
    backend.max_context_len = 1024
    backend.token_to_kv_pool = SimpleNamespace(page_size=PAGE)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=HEADS, head_dim=DIM)
    cache = torch.zeros(PAGE, KV_HEADS, DIM, device="cuda").to(torch.float8_e4m3fn)
    with caplog.at_level(logging.WARNING):
        assert backend._resolve_flashinfer_sparse(layer, cache) is None
    assert "only BF16 pools" in caplog.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
