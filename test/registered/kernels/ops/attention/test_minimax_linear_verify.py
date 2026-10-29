"""MiniMax-M3 EAGLE verify: each draft row sees exactly its causal prefix."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# One TP4 rank of MiniMax-M3: 16 query heads, 1 KV head, 1 index head.
NQH, NKH, NIH, HD = 16, 1, 1, 128
BLK, TOPK, INIT_BLOCKS, LOCAL_BLOCKS, PAGE = 128, 16, 0, 1, 128
NDT = 4
MAX_CONTEXT = 64 * 1024
# ReqToTokenPool width with speculative decoding: context + 4 + ndt.
WIDTH = MAX_CONTEXT + 4 + NDT
LAYER_ID = 3


def _paged_req_to_token(num_reqs, first_page=0):
    pages_per_req = (WIDTH + PAGE - 1) // PAGE
    rows = []
    for r in range(num_reqs):
        base = first_page + r * pages_per_req
        pages = torch.randperm(pages_per_req, device="cuda") + base
        slots = pages[:, None] * PAGE + torch.arange(PAGE, device="cuda")
        rows.append(slots.reshape(-1)[:WIDTH])
    return torch.stack(rows).to(torch.int32), first_page + num_reqs * pages_per_req


def _caches(num_slots, with_value):
    nan = float("nan")
    k = torch.full((num_slots, NKH, HD), nan, dtype=torch.bfloat16, device="cuda")
    v = torch.full_like(k, nan)
    idx_k = torch.full((num_slots, 1, HD), nan, dtype=torch.bfloat16, device="cuda")
    idx_v = torch.full_like(idx_k, nan) if with_value else None
    return k, v, idx_k, idx_v


class _Pool:
    """The slice of MiniMaxSparseKVPool that forward_decode touches."""

    def __init__(self, k, v, idx_k, idx_v):
        self.k, self.v, self.idx_k, self.idx_v = k, v, idx_k, idx_v

    def set_fused_kv_index_buffer(self, layer, loc, k, v, idx_k, idx_v, *scales):
        self.k[loc] = k.view(-1, *self.k.shape[1:])
        self.v[loc] = v.view(-1, *self.v.shape[1:])
        self.idx_k[loc] = idx_k.view(-1, *self.idx_k.shape[1:])
        if idx_v is not None:
            self.idx_v[loc] = idx_v.view(-1, *self.idx_v.shape[1:])

    def get_kv_buffer(self, layer_id):
        return self.k, self.v

    def get_index_k_buffer(self, layer_id):
        return self.idx_k

    def get_index_kv_buffer(self, layer_id):
        return self.idx_k, self.idx_v


def _backend(req_to_token, pool, with_value):
    b = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
    b.is_npu = False
    b._linear_verify = True
    b.speculative_num_draft_tokens = NDT
    b.max_context_len = MAX_CONTEXT
    b.req_to_token = req_to_token
    b.kv_pool = pool
    b.page_size = PAGE
    b.block_size_k, b.topk_blocks = BLK, TOPK
    b.init_blocks, b.local_blocks = INIT_BLOCKS, LOCAL_BLOCKS
    b.score_type = "max"
    b.disable_value_layer_ids = set() if with_value else {LAYER_ID}
    b.use_dense_sparse_decode = False
    # MSA decode on (no plan is built for verify): verify rows must still run on Triton.
    b._use_msa_decode, b._msa_owns_decode = True, False
    b._msa_dec_meta = None
    b.fp8_attn_gemm = False
    b.index_cache_enabled = False
    b._decode_topk_buf = {}
    b._topk_is_source = {}
    b.hisparse_coordinator = None
    b.indexer_cp = None
    b._linear_verify_meta = None
    b._prefill_seqblock_meta = None
    b._max_seqlen_q = b._max_seqlen_k = 1
    return b


_LAYER = SimpleNamespace(
    layer_id=LAYER_ID,
    q_scale_float=None,
    k_scale_float=None,
    v_scale_float=None,
    idx_q_scale_float=None,
    idx_k_scale_float=None,
    idx_v_scale_float=None,
)


class _Case:
    def __init__(self, prefixes, with_value, seed=0):
        torch.manual_seed(seed)
        self.with_value = with_value
        self.bs = len(prefixes)
        self.prefix = torch.tensor(prefixes, dtype=torch.int64, device="cuda")
        self.req_to_token, num_pages = _paged_req_to_token(self.bs)
        self.pool = _Pool(*_caches(num_pages * PAGE, with_value))
        rows = self.bs * NDT
        self.q = torch.randn(rows, NQH, HD, dtype=torch.bfloat16, device="cuda")
        self.k = torch.randn(rows, NKH, HD, dtype=torch.bfloat16, device="cuda")
        self.v = torch.randn_like(self.k)
        self.idx_q = torch.randn(rows, NIH, HD, dtype=torch.bfloat16, device="cuda")
        self.idx_k = torch.randn(rows, 1, HD, dtype=torch.bfloat16, device="cuda")
        self.idx_v = torch.randn_like(self.idx_k) if with_value else None
        caches = [self.pool.k, self.pool.v, self.pool.idx_k, self.pool.idx_v]
        for r, p in enumerate(prefixes):
            slots = self.req_to_token[r, :p].long()
            for cache in caches:
                if cache is not None:
                    cache[slots] = torch.randn(
                        p, *cache.shape[1:], dtype=cache.dtype, device="cuda"
                    )
        draft_pos = self.prefix[:, None] + torch.arange(NDT, device="cuda")
        self.out_cache_loc = (
            self.req_to_token.gather(1, draft_pos).reshape(-1).to(torch.int64)
        )

    def batch(self):
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=self.prefix.clone(),
            seq_lens_cpu=self.prefix.cpu(),
            req_pool_indices=torch.arange(self.bs, dtype=torch.int64, device="cuda"),
            out_cache_loc=self.out_cache_loc.clone(),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            extend_prefix_lens=None,
            minimax_m3_precached_sparse_layers=None,
        )

    def reference(self):
        """Decode every draft row alone, on a copy that ends at its own token."""
        rows = self.bs * NDT
        ref_r2t, num_pages = _paged_req_to_token(rows)
        k, v, idx_k, idx_v = _caches(num_pages * PAGE, self.with_value)
        pairs = [
            (k, self.pool.k, self.k),
            (v, self.pool.v, self.v),
            (idx_k, self.pool.idx_k, self.idx_k),
            (idx_v, self.pool.idx_v, self.idx_v),
        ]
        seq_lens = []
        for r in range(self.bs):
            p = int(self.prefix[r])
            for j in range(NDT):
                row = r * NDT + j
                dst_prefix = ref_r2t[row, :p].long()
                dst_draft = ref_r2t[row, p : p + j + 1].long()
                src = self.req_to_token[r, :p].long()
                for dst, pool_cache, draft in pairs:
                    if dst is None:
                        continue
                    dst[dst_prefix] = pool_cache[src]
                    dst[dst_draft] = draft[r * NDT : row + 1].view(-1, *dst.shape[1:])
                seq_lens.append(p + j + 1)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")
        idx_o, o = minimax_sparse_decode(
            self.q,
            None,
            k,
            v,
            self.idx_q,
            None,
            idx_k,
            idx_v,
            ref_r2t,
            torch.arange(rows, dtype=torch.int64, device="cuda"),
            seq_lens,
            int(seq_lens.max()),
            1,
            BLK,
            TOPK,
            INIT_BLOCKS,
            LOCAL_BLOCKS,
            score_type="max",
            disable_index_value=not self.with_value,
            page_size=PAGE,
        )
        return (
            None if idx_o is None else idx_o.reshape(rows, -1),
            o.reshape(rows, -1),
        )


def _verify(backend, batch, q, k, v, idx_q, idx_k, idx_v):
    backend.init_forward_metadata_in_graph(batch)
    return backend.forward_extend(
        q, k, v, _LAYER, batch, idx_q=idx_q, idx_k=idx_k, idx_v=idx_v
    )


def _assert_same(got, want, exact, rows=None):
    for g, w in zip(got, want):
        if w is None:
            assert g is None
            continue
        assert not torch.isnan(g).any()
        if rows is not None:
            g, w = g[rows], w[rows]
        if exact:
            assert torch.equal(g, w)
        else:
            torch.testing.assert_close(g, w, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("with_value", [False, True])
def test_eager_verify_matches_causal_rows(with_value):
    """Rows of one request share its cache yet see only their own prefix."""
    # 5118 + ndt crosses into a new 128-token block, so the bound must count the drafts.
    case = _Case([1000, 5118], with_value)
    backend = _backend(case.req_to_token, case.pool, with_value)
    batch = case.batch()
    backend.init_forward_metadata_out_graph(batch)
    assert backend._max_seqlen_k == 5118 + NDT
    got = _verify(
        backend,
        batch,
        case.q,
        case.k,
        case.v,
        case.idx_q,
        case.idx_k,
        case.idx_v,
    )
    _assert_same(got, case.reference(), exact=True)
    # Verify leaves the batch's per-request metadata to the dense layers.
    assert batch.seq_lens.tolist() == [1000, 5118]
    assert batch.extend_seq_lens is None


@pytest.mark.parametrize("with_value", [False, True])
def test_graph_replay_at_longer_context(with_value):
    """A graph captured on 1-token dummies replays at the end of the context."""
    # Request 0's last two rows run past MAX_CONTEXT: they are clamped, never committed.
    case = _Case([MAX_CONTEXT - 2, 33000], with_value)
    backend = _backend(case.req_to_token, case.pool, with_value)

    static = case.batch()
    static.seq_lens.fill_(1)  # get_cuda_graph_seq_len_fill_value()
    static.seq_lens_cpu = static.seq_lens.cpu()
    # Capture-time KV writes land in the free tail of each row.
    static.out_cache_loc.copy_(
        case.req_to_token[:, WIDTH - NDT :].reshape(-1).to(torch.int64)
    )
    inputs = [t.clone() for t in (case.q, case.k, case.v, case.idx_q, case.idx_k)]
    inputs.append(None if case.idx_v is None else case.idx_v.clone())

    backend.init_forward_metadata_out_graph(static, in_capture=True)
    assert backend._max_seqlen_k == MAX_CONTEXT
    _verify(backend, static, *inputs)  # warm-up: autotune and JIT compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _verify(backend, static, *inputs)

    static.seq_lens.copy_(case.prefix)
    static.out_cache_loc.copy_(case.out_cache_loc)
    graph.replay()
    torch.cuda.synchronize()
    # No row may read past the context-length bound the score buffer was sized for.
    assert int(backend._linear_verify_meta.seq_lens.max()) == MAX_CONTEXT
    inside = [
        r
        for r in range(case.bs * NDT)
        if int(case.prefix[r // NDT]) + r % NDT + 1 <= MAX_CONTEXT
    ]
    # The captured and reference score buffers differ in width; compare to tolerance.
    _assert_same(out, case.reference(), exact=False, rows=inside)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
