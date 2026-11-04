"""DSA under decode context parallelism (DCP), one GPU simulating every rank.

- Storage: every rank receives the same widened write locs and must store
  exactly the tokens it owns (slot % N == rank) at row slot // N, byte-equal to
  the DCP1 rows, leaving every other row untouched.
- Decode top-k: each rank scores its own index-K shard through the serving
  page-table, logits and top-k paths, and the merge over all ranks must select
  the stable global top-k (score descending, position ascending) of the DCP1
  logits.
- Attention: every rank runs FlashMLA on the gathered q (96 native heads, or
  padded to 128 on older kernels) over its own KV shard with top-k compacted; the
  merged partials must be about as accurate as DCP1 against an fp32 reference.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dcp_kernels import CPTritonContext, correct_attn_out
from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
from sglang.kernels.ops.attention.dsa.indexer_k import fused_k_indexer_norm_rope
from sglang.kernels.ops.attention.dsa.metadata import fused_dsa_decode_metadata
from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    deepgemm_paged_mqa_logits_native,
)
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import (
    _scatter_candidates_kernel,
    dcp_all_visible,
    dcp_global_topk,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _flashmla_kv_padded_heads,
)
from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_mla_dcp_lse_base_on_e,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cuda_ci(est_time=190, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def setUpModule():
    # Kimi-K3.1's sharded index-K DCP, which its DSA layers opt into
    dcp_layout._dsa_shards_index_k = True


def tearDownModule():
    dcp_layout._dsa_shards_index_k = False


PAGE = 64
TOPK = 2048
KV_LORA, ROPE, INDEX_DIM = 512, 64, 128
FP8_DS_MLA_BYTES = 656


def _make_pool(rows: int) -> DSATokenToKVPool:
    return DSATokenToKVPool(
        size=rows,
        page_size=PAGE,
        kv_lora_rank=KV_LORA,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=ROPE,
        layer_num=1,
        device="cuda",
        index_head_dim=INDEX_DIM,
        enable_memory_saver=False,
        kv_cache_dim=FP8_DS_MLA_BYTES,
    )


# --- storage

MAX_POS = 8192


class _IndexerStub:
    """The Indexer state its two index-K writers read."""

    _store_index_k_cache = Indexer._store_index_k_cache
    _fused_k_prepare_and_store = Indexer._fused_k_prepare_and_store

    def __init__(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        self.block_size, self.scale_fmt = 128, "ue8m0"
        self.k_norm = SimpleNamespace(
            weight=torch.rand(INDEX_DIM, device="cuda", generator=gen) + 0.5,
            bias=torch.randn(INDEX_DIM, device="cuda", generator=gen) * 0.1,
            variance_epsilon=1e-6,
        )
        self._indexer_cos_sin_cache = torch.rand(
            MAX_POS, ROPE, device="cuda", generator=gen
        )


class _Batch:
    def __init__(self, loc, positions):
        self.loc, self.positions = loc, positions
        n = loc.numel()
        self.k_nope = torch.randn(n, 1, KV_LORA, device="cuda", dtype=torch.bfloat16)
        self.k_rope = torch.randn(n, 1, ROPE, device="cuda", dtype=torch.bfloat16)
        self.index_key = torch.randn(n, INDEX_DIM, device="cuda", dtype=torch.bfloat16)


def _write(pool, indexer, loc, batch, writer) -> None:
    """Write through the serving paths: MLA set_mla_kv_buffer, then one of the
    indexer writers. `reference` replays the fallback branch's math explicitly."""
    pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, batch.k_nope, batch.k_rope)
    fb = SimpleNamespace(out_cache_loc=loc)
    with forward_context(
        ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=pool))
    ):
        if writer == "store":
            indexer._store_index_k_cache(
                forward_batch=fb, layer_id=0, key=batch.index_key, out_cache_loc=loc
            )
        elif writer == "prepare":
            indexer._fused_k_prepare_and_store(
                batch.index_key, batch.positions, fb, 0, None, out_cache_loc=loc
            )
        else:  # "prepare_reference": norm + rope, then the plain store
            key = fused_k_indexer_norm_rope(
                batch.index_key,
                indexer.k_norm.weight,
                indexer.k_norm.bias,
                indexer.k_norm.variance_epsilon,
                indexer._indexer_cos_sin_cache,
                batch.positions,
            )
            indexer._store_index_k_cache(
                forward_batch=fb, layer_id=0, key=key, out_cache_loc=loc
            )
    torch.cuda.synchronize()


def _mla_rows(pool, rows) -> torch.Tensor:
    return pool.kv_buffer[0].view(torch.uint8).view(-1, FP8_DS_MLA_BYTES)[rows]


def _index_rows(pool, rows, buf=None) -> torch.Tensor:
    """[len(rows), 132]: 128 fp8 bytes + 4 scale bytes of each index-K row."""
    if buf is None:
        buf = pool.get_index_k_with_scale_buffer(0)
    buf = buf.view(-1, PAGE * (INDEX_DIM + 4))
    page, off = rows // PAGE, rows % PAGE
    values = buf[:, : PAGE * INDEX_DIM].reshape(-1, PAGE, INDEX_DIM)[page, off]
    scales = buf[:, PAGE * INDEX_DIM :].reshape(-1, PAGE, 4)[page, off]
    return torch.cat([values, scales], dim=-1)


def _snapshot(pool):
    return pool.kv_buffer[0].clone(), pool.get_index_k_with_scale_buffer(0).clone()


class TestDSADCPKVPool(CustomTestCase):
    ROWS = 4096  # physical rows per rank, identical for DCP1 and DCP8

    def _run(self, dcp_size: int, steps, writer: str) -> None:
        """`steps`: per forward, a list of (request, num_new_tokens); requests
        grow from their own widened-page-aligned start."""
        torch.manual_seed(0)
        widened_page = PAGE * dcp_size
        starts = [widened_page, widened_page * 40, widened_page * 50]  # disjoint
        dcp1_starts = [PAGE, PAGE * 40, PAGE * 50]
        lens = [0] * len(starts)
        indexer = _IndexerStub()
        ref_writer = "prepare_reference" if writer == "prepare" else writer
        all_rows = torch.arange(self.ROWS, device="cuda")

        with published_topology():
            ref = _make_pool(self.ROWS)
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            pools = []
            for rank in range(dcp_size):
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    pools.append(_make_pool(self.ROWS))
                    self.assertEqual(pools[-1].size, self.ROWS)

        for step in steps:
            pos = torch.cat(
                [torch.arange(lens[r], lens[r] + n, device="cuda") for r, n in step]
            )
            loc = torch.cat(
                [
                    starts[r] + torch.arange(lens[r], lens[r] + n, device="cuda")
                    for r, n in step
                ]
            )
            dcp1_loc = torch.cat(
                [
                    dcp1_starts[r] + torch.arange(lens[r], lens[r] + n, device="cuda")
                    for r, n in step
                ]
            )
            for r, n in step:
                lens[r] += n
            batch = _Batch(loc, pos)
            with published_topology():
                _write(ref, indexer, dcp1_loc, batch, ref_writer)
            with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
                for rank, pool in enumerate(pools):
                    before_kv, before_index = _snapshot(pool)
                    with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                        _write(pool, indexer, loc, batch, writer)
                    owned = loc % dcp_size == rank
                    rows = loc[owned] // dcp_size
                    self.assertTrue(
                        torch.equal(
                            _mla_rows(pool, rows), _mla_rows(ref, dcp1_loc[owned])
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            _index_rows(pool, rows), _index_rows(ref, dcp1_loc[owned])
                        )
                    )
                    # Everything else is unchanged, also when this rank owns no new token.
                    others = all_rows[~torch.isin(all_rows, rows)]
                    before_mla = before_kv.view(torch.uint8).view(-1, FP8_DS_MLA_BYTES)
                    self.assertTrue(
                        torch.equal(_mla_rows(pool, others), before_mla[others])
                    )
                    self.assertTrue(
                        torch.equal(
                            _index_rows(pool, others),
                            _index_rows(pool, others, buf=before_index),
                        )
                    )

    # prefill ending mid-page, three single-token decodes (each owned by one
    # rank only), then a 3-request decode step
    STEPS = [
        [(0, 1531)],
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
        [(1, 700)],
        [(0, 1), (1, 1), (2, 1)],
    ]

    def test_dcp8_store(self):
        self._run(8, self.STEPS, "store")

    def test_dcp2_store(self):
        self._run(2, self.STEPS, "store")

    def test_dcp8_fused_prepare_fallback(self):
        self._run(8, self.STEPS, "prepare")


class TestDSADCPRetractionCopy(CustomTestCase):
    """Retraction offloads a request's KV and index-K to CPU by its LOGICAL
    (widened) slots and restores them later. Under DCP each rank keeps only
    the slots it owns, as rows slot // dcp_size, for the index-K exactly as for
    the MLA KV: the copy must use those rows (the widened slots index past the
    rank's index-K buffer)."""

    ROWS = 4096  # physical rows per rank

    def _run(self, dcp_size: int) -> None:
        gen = torch.Generator().manual_seed(dcp_size)
        # One request's logical slots: three whole widened pages, not from 0.
        widened = PAGE * dcp_size
        slots = (torch.arange(3 * widened) + 2 * widened).cuda()
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            for rank in range(dcp_size):
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    pool = _make_pool(self.ROWS)
                    for buf in (
                        pool.kv_buffer[0].view(torch.uint8),
                        pool.get_index_k_with_scale_buffer(0).view(torch.uint8),
                    ):
                        buf.copy_(
                            torch.randint(0, 256, buf.shape, generator=gen).to(buf)
                        )
                    rows = slots[slots % dcp_size == rank] // dcp_size
                    want_kv = _mla_rows(pool, rows).clone()
                    want_index = _index_rows(pool, rows).clone()
                    saved = pool.get_cpu_copy(slots)
                    pool.kv_buffer[0].zero_()
                    pool.get_index_k_with_scale_buffer(0).zero_()
                    pool.load_cpu_copy(saved, slots)
                    self.assertTrue(torch.equal(_mla_rows(pool, rows), want_kv))
                    self.assertTrue(torch.equal(_index_rows(pool, rows), want_index))

    def test_dcp8(self):
        self._run(8)

    def test_dcp2(self):
        self._run(2)


# --- decode top-k

TOPK_ROWS = 384 * 1024  # physical rows per rank


# Index keys per request: ("random", n); ("tied", n, d): keys 2^j * B for
# random j < d, so the d scores tie exactly yet sit in separate histogram bins
# (B and 2B quantize to the same fp8 values with doubled scales);
# ("concentrated", n): rank 0 holds 2040 keys 2B and 20 keys B, another rank
# 20 keys B, the rest are zero. The k-th place falls among the B keys, which
# interleave in the first 480 positions, and rank 0 must keep its 8 first.
def _index_keys(spec, gen: torch.Generator) -> torch.Tensor:
    kind, n = spec[0], spec[1]
    if kind == "random":
        keys = torch.randn(n, INDEX_DIM, generator=gen)
    elif kind == "tied":
        b = torch.randn(INDEX_DIM, generator=gen)
        scale = 2.0 ** torch.randint(0, spec[2], (n, 1), generator=gen)
        keys = scale * b
    else:
        b = torch.randn(INDEX_DIM, generator=gen)
        keys = torch.zeros(n, INDEX_DIM)
        # positions 8j are on rank 0, 8j + 1 on rank 1, for dcp_size 2 and 8
        keys[torch.arange(0, 8 * 2060, 8)] = 2 * b
        keys[8 * torch.randperm(60, generator=gen)[:20]] = b
        keys[8 * torch.randperm(60, generator=gen)[:20] + 1] = b
    return keys.to("cuda", torch.bfloat16)


# (spec, strict): strict rows must match the stable order exactly.
DECODE = [
    (("random", 1), True),
    (("random", 3), True),  # most DCP8 ranks empty
    (("random", 63), True),
    (("random", 64), True),
    (("random", 65), True),
    (("random", 513), True),
    (("random", 2047), True),
    (("random", 2048), True),
    (("random", 2049), True),
    (("random", 9000), True),
    (("random", 131072), True),  # long enough for the cluster path
    (("tied", 5000, 3), True),
    (("concentrated", 20000), True),
    (("tied", 4096, 1), False),
    (("tied", 20000, 1), False),
]


def _req_to_token(span: int, lens, gen: torch.Generator) -> torch.Tensor:
    """[bs, max_len] slots; each request takes shuffled, non-adjacent pages
    (page 0, the padding page, fills the tail)."""
    max_pages = -(-max(lens) // span)
    used = [-(-n // span) for n in lens]
    free = torch.randperm(TOPK_ROWS // PAGE - 1, generator=gen)[: sum(used)] + 1
    table = torch.zeros(len(lens), max_pages, dtype=torch.int64)
    for b, part in enumerate(free.split(used)):
        table[b, : part.numel()] = part
    pos = torch.arange(max_pages * span, device="cuda")
    return (table.cuda()[:, pos // span] * span + pos % span).to(torch.int32)


def _store(pool, slots: torch.Tensor, keys: torch.Tensor) -> None:
    slots = slots.to(torch.int64)
    ctx = ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=pool))
    with forward_context(ctx):
        _IndexerStub()._store_index_k_cache(
            forward_batch=SimpleNamespace(out_cache_loc=slots),
            layer_id=0,
            key=keys,
            out_cache_loc=slots,
        )


def _paged_logits(pool, q, weights, lens, block_table) -> torch.Tensor:
    import deep_gemm

    kv = pool.get_index_k_with_scale_buffer(0)
    kv = kv.view(kv.shape[0], PAGE, 1, INDEX_DIM + 4)
    lens_2d = lens.view(-1, 1).to(torch.int32).contiguous()
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens_2d, PAGE, num_sms)
    return deep_gemm.fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        lens_2d,
        block_table,
        schedule,
        block_table.shape[1] * PAGE,
        clean_logits=False,
    )


def _native_verify_logits(pool, q, weights, lens, block_table) -> torch.Tensor:
    """The serving DG-native verify call: q as [bs, NEXT_N], a [bs, NEXT_N]
    length per row and one page-table row per request."""
    import deep_gemm

    kv = pool.get_index_k_with_scale_buffer(0)
    kv = kv.view(kv.shape[0], PAGE, 1, INDEX_DIM + 4)
    rows = lens.numel()
    lens_2d = lens.view(-1, NEXT_N).to(torch.int32).contiguous()
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens_2d, PAGE, num_sms)
    return deepgemm_paged_mqa_logits_native(
        deep_gemm.fp8_paged_mqa_logits,
        q.squeeze(1),
        kv,
        weights,
        lens_2d,
        block_table,
        schedule,
        block_table.shape[1] * PAGE,
        q_offset=rows,
        B=rows // NEXT_N,
        next_n=NEXT_N,
    )


def _real_table(req_to_token: torch.Tensor, dcp_size: int) -> torch.Tensor:
    backend = SimpleNamespace(real_page_span=PAGE * dcp_size)
    return DeepseekSparseAttnBackend._transform_table_1_to_real(backend, req_to_token)


def _fused_real_table(req_to_token, lens, dcp_size: int) -> torch.Tensor:
    """The table the fused decode-metadata kernel writes on CUDA graph replay."""
    bs, max_len = req_to_token.shape
    span = PAGE * dcp_size
    table = torch.zeros(bs, -(-max_len // span), dtype=torch.int32, device="cuda")
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device="cuda")
    fused_dsa_decode_metadata(
        seq_lens=lens.to(torch.int64),
        req_pool_indices=torch.arange(bs, device="cuda"),
        req_to_token=req_to_token,
        cache_seqlens=i32(bs),
        cu_seqlens_k=i32(bs + 1),
        page_table_1=None,
        dsa_cache_seqlens=i32(bs),
        dsa_cu_seqlens_k=i32(bs + 1),
        real_page_table=table,
        bs=bs,
        max_len=max_len,
        dsa_index_topk=TOPK,
        physical_page_size=span,
    )
    return table


def _merge_all_ranks(dcp_size: int, run_rank):
    """Run every rank twice: the first pass records each rank's candidates,
    the second feeds their concatenation as the all-gather result."""
    calls = []
    for rank in range(dcp_size):
        rank_calls = []

        def record(t, rank_calls=rank_calls):
            rank_calls.append(t.clone())
            return t.repeat(dcp_size, *([1] * (t.dim() - 1)))

        run_rank(rank, record)
        calls.append(rank_calls)
    outs = []
    for rank in range(dcp_size):
        it = iter(
            torch.cat([calls[r][i] for r in range(dcp_size)])
            for i in range(len(calls[0]))
        )
        outs.append(run_rank(rank, lambda t, it=it: next(it)))
    return calls, outs


class TestDSADCPTopK(CustomTestCase):
    """The v2 top-k kernel resolves ties exactly while the candidates near the
    k-th score fit its tie buffer (2048). Rows with more ties than that (DCP1
    has the same limit) are checked as in the acceptance definition: same
    count, bitwise equal score multiset, and equal sets apart from keys tied at
    the k-th score."""

    def _check_row(self, msg, slots, inv_slot, scores, visible, strict):
        """`scores`: DCP1 fp32 scores of the row's global positions 0..visible-1."""
        k = min(visible, TOPK)
        valid = slots[slots >= 0].long()
        self.assertEqual(valid.numel(), k, msg)
        got = inv_slot[valid]
        self.assertTrue(bool((got >= 0).all()) and got.unique().numel() == k, msg)
        order = torch.sort(-scores, stable=True).indices[:k]  # stable reference
        if strict:
            self.assertTrue(torch.equal(got.sort().values, order.sort().values), msg)
            return
        self.assertTrue(
            torch.equal(scores[got].sort().values, scores[order].sort().values), msg
        )
        kth = scores[order[-1]]
        self.assertTrue(
            torch.equal(
                got[scores[got] != kth].sort().values,
                order[scores[order] != kth].sort().values,
            ),
            msg,
        )

    def _layouts(self, specs, dcp_size, gen):
        lens = [s[1] for s in specs]
        keys = [_index_keys(s, gen) for s in specs]
        ref_r2t = _req_to_token(PAGE, lens, gen)
        with published_topology():
            ref_pool = _make_pool(TOPK_ROWS)
            for b, n in enumerate(lens):
                _store(ref_pool, ref_r2t[b, :n], keys[b])
        r2t = _req_to_token(PAGE * dcp_size, lens, gen)
        inv_slot = torch.full(
            (TOPK_ROWS * dcp_size,), -1, dtype=torch.int64, device="cuda"
        )
        pools = []
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            for rank in range(dcp_size):
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    pools.append(_make_pool(TOPK_ROWS))
                    for b, n in enumerate(lens):
                        _store(pools[-1], r2t[b, :n], keys[b])
        for b, n in enumerate(lens):
            inv_slot[r2t[b, :n].long()] = torch.arange(n, device="cuda")
        return lens, ref_r2t, ref_pool, r2t, pools, inv_slot

    def _run_decode(self, dcp_size: int) -> None:
        gen = torch.Generator().manual_seed(dcp_size)
        specs = [s for s, _ in DECODE]
        lens_list, ref_r2t, ref_pool, r2t, pools, inv_slot = self._layouts(
            specs, dcp_size, gen
        )
        bs = len(lens_list)
        lens = torch.tensor(lens_list, dtype=torch.int32, device="cuda")
        q = torch.randn(bs, 1, 64, INDEX_DIM, generator=gen).cuda()
        q = q.to(torch.float8_e4m3fn)
        weights = torch.rand(bs, 64, generator=gen).cuda()
        ref_logits = _paged_logits(
            ref_pool, q, weights, lens, ref_r2t[:, ::PAGE] // PAGE
        )

        table = _real_table(r2t, dcp_size)
        fused = _fused_real_table(r2t, lens, dcp_size)
        for b, n in enumerate(lens_list):
            pages = -(-n // (PAGE * dcp_size))
            self.assertTrue(torch.equal(table[b, :pages], fused[b, :pages]))

        rank_logits = []
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            for rank in range(dcp_size):
                local_lens = get_dcp_lens(lens, dcp_size, rank)
                logits = _paged_logits(pools[rank], q, weights, local_lens, table)
                for b in range(bs):
                    n = int(local_lens[b])
                    cols = torch.arange(n, device="cuda") * dcp_size + rank
                    self.assertTrue(
                        torch.equal(logits[b, :n], ref_logits[b, cols]),
                        f"rank {rank} request {b}: logits differ from DCP1",
                    )
                rank_logits.append((logits, local_lens))

            # decode merges with the per-forward v2 plan (cluster path)
            plan = plan_topk_v2(lens)
            torch.cuda.synchronize()

            def run_rank(rank, all_gather):
                logits, local_lens = rank_logits[rank]
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    return dcp_global_topk(
                        logits.clone(),
                        local_lens,
                        lens,
                        table,
                        TOPK,
                        PAGE,
                        plan,
                        all_gather=all_gather,
                    )

            _, outs = _merge_all_ranks(dcp_size, run_rank)
        for rank, out in enumerate(outs):
            for b, (spec, strict) in enumerate(DECODE):
                n = lens_list[b]
                self._check_row(
                    f"rank {rank} request {b} {spec}",
                    out[b],
                    inv_slot,
                    ref_logits[b, :n],
                    n,
                    strict,
                )

    def test_dcp8_decode(self):
        self._run_decode(8)

    def test_dcp2_decode(self):
        self._run_decode(2)


# Target verify: every request of length L (its draft tokens included) is
# NEXT_N query rows; row j sees the causal prefix L - NEXT_N + 1 + j.
NEXT_N = 8
VERIFY = [
    (("random", 8), True),  # rows see 1..8 keys: most DCP8 ranks empty
    (("random", 9), True),
    (("random", 64), True),
    (("random", 71), True),  # rows cross the 64-key page
    (("random", 513), True),
    (("random", 2047), True),
    (("random", 2052), True),  # rows on both sides of the k-only bound 2048
    (("random", 2056), True),
    (("random", 9000), True),
    (("random", 131072), True),  # long enough for the cluster path
    (("tied", 5000, 3), True),
    (("concentrated", 20000), True),
    (("tied", 4096, 1), False),
]


def _verify_rows(lens_list):
    """Per-row global lengths and the request of each row."""
    rows = [
        (b, n - NEXT_N + 1 + j) for b, n in enumerate(lens_list) for j in range(NEXT_N)
    ]
    req = torch.tensor([b for b, _ in rows], device="cuda")
    glob = torch.tensor([g for _, g in rows], dtype=torch.int32, device="cuda")
    return req, glob


class TestDSADCPVerifyTopK(TestDSADCPTopK):
    """Target verify under DCP: the decode merge applied to the bs x NEXT_N
    flattened rows, each with its own local / global causal length and the
    request's page table repeated per row (as DSAMetadata lays them out)."""

    def _run_verify(self, dcp_size: int) -> None:
        gen = torch.Generator().manual_seed(100 + dcp_size)
        specs = [s for s, _ in VERIFY]
        lens_list, ref_r2t, ref_pool, r2t, pools, inv_slot = self._layouts(
            specs, dcp_size, gen
        )
        req, glob = _verify_rows(lens_list)
        rows = glob.numel()
        q = torch.randn(rows, 1, 64, INDEX_DIM, generator=gen).cuda()
        q = q.to(torch.float8_e4m3fn)
        weights = torch.rand(rows, 64, generator=gen).cuda()
        ref_table = (ref_r2t[:, ::PAGE] // PAGE)[req]
        ref_logits = _paged_logits(ref_pool, q, weights, glob, ref_table)
        table = _real_table(r2t, dcp_size)[req]

        rank_logits = []
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            for rank in range(dcp_size):
                local = get_dcp_lens(glob, dcp_size, rank)
                logits = _paged_logits(pools[rank], q, weights, local, table)
                native = _native_verify_logits(pools[rank], q, weights, local, table)
                for r in range(rows):
                    n = int(local[r])
                    cols = torch.arange(n, device="cuda") * dcp_size + rank
                    self.assertTrue(
                        torch.equal(logits[r, :n], ref_logits[r, cols]),
                        f"rank {rank} row {r}: logits differ from DCP1",
                    )
                    self.assertTrue(
                        torch.equal(native[r, :n], logits[r, :n]),
                        f"rank {rank} row {r}: DG-native logits differ from per-row",
                    )
                rank_logits.append((logits, local))

            plan = plan_topk_v2(glob)
            torch.cuda.synchronize()

            def run_rank(rank, all_gather):
                logits, local = rank_logits[rank]
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    return dcp_global_topk(
                        logits.clone(),
                        local,
                        glob,
                        table,
                        TOPK,
                        PAGE,
                        plan,
                        all_gather=all_gather,
                    )

            _, outs = _merge_all_ranks(dcp_size, run_rank)
        strict_of = [strict for _, strict in VERIFY]
        for rank, out in enumerate(outs):
            for r in range(rows):
                b, n = int(req[r]), int(glob[r])
                self._check_row(
                    f"rank {rank} row {r} (request {b} {VERIFY[b][0]}, {n} keys)",
                    out[r],
                    inv_slot,
                    ref_logits[r, :n],
                    n,
                    strict_of[b],
                )

    def _run_verify_all_visible(self, dcp_size: int) -> None:
        """The indexer's k-only path: rows that see at most TOPK keys get every
        visible position as a widened slot."""
        gen = torch.Generator().manual_seed(200 + dcp_size)
        specs = [s for s, _ in VERIFY if s[1] <= TOPK + NEXT_N - 1]
        lens_list, _, _, r2t, _, inv_slot = self._layouts(specs, dcp_size, gen)
        req, glob = _verify_rows(lens_list)
        keep = glob <= TOPK
        req, glob = req[keep], glob[keep]
        table = _real_table(r2t, dcp_size)[req]
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            out = dcp_all_visible(glob, table, TOPK, PAGE)
        for r in range(glob.numel()):
            n = int(glob[r])
            slots = out[r]
            valid = slots[slots >= 0].long()
            self.assertEqual(valid.numel(), n, f"row {r}")
            got = inv_slot[valid].sort().values
            self.assertTrue(
                torch.equal(got, torch.arange(n, device="cuda")),
                f"row {r} (request {int(req[r])}, {n} keys): slots are not 0..{n - 1}",
            )

    def test_dcp8_verify(self):
        self._run_verify(8)

    def test_dcp2_verify(self):
        self._run_verify(2)

    def test_dcp8_verify_all_visible(self):
        self._run_verify_all_visible(8)

    def test_dcp2_verify_all_visible(self):
        self._run_verify_all_visible(2)

    def test_verify_ctx_lens_take_dg_native_layout(self):
        """The backend hands the indexer this rank's verify lengths as
        [bs, NEXT_N], the shape its DG-native gate requires."""
        dcp_size, rank, bs = 8, 3, 5
        backend = SimpleNamespace(
            speculative_num_draft_tokens=NEXT_N,
            attn_dcp_size=dcp_size,
            attn_dcp_rank=rank,
        )
        backend._dcp_local_ctx_lens = lambda lens: (
            DeepseekSparseAttnBackend._dcp_local_ctx_lens(backend, lens)
        )
        _, glob = _verify_rows([9, 71, 513, 2048, 70000][:bs])
        lens = DeepseekSparseAttnBackend._build_paged_mqa_schedule_2d_ctx_lens(
            backend, ForwardMode.TARGET_VERIFY, None, glob, bs
        )
        self.assertEqual(tuple(lens.shape), (bs, NEXT_N))
        self.assertTrue(
            torch.equal(lens.view(-1), get_dcp_lens(glob, dcp_size, rank).int())
        )

    # the decode cases already run in TestDSADCPTopK
    test_dcp8_decode = None
    test_dcp2_decode = None


# --- attention

HEADS, DIM, DV = 96, KV_LORA + ROPE, KV_LORA
ATTN_ROWS = 128 * 1024  # physical rows per rank
ATTN_TOKENS = 30000
SM_SCALE = 0.1


def _slots(span: int, gen: torch.Generator) -> torch.Tensor:
    """Slots of positions 0..ATTN_TOKENS-1 on shuffled pages."""
    pages = -(-ATTN_TOKENS // span)
    table = (torch.randperm(ATTN_ROWS // PAGE - 1, generator=gen)[:pages] + 1).cuda()
    pos = torch.arange(ATTN_TOKENS, device="cuda")
    return table[pos // span] * span + pos % span


def _row_positions(dcp_size: int, gen: torch.Generator) -> torch.Tensor:
    """[rows, TOPK] global positions, -1 padded."""
    rows = []

    def pick(candidates, n):
        p = candidates[torch.randperm(candidates.numel(), generator=gen)[:n]]
        return torch.cat([p, torch.full((TOPK - n,), -1)])

    every = torch.arange(ATTN_TOKENS)
    rows.append(pick(every, TOPK))
    rows.append(pick(every, 500))
    rows.append(pick(every[every % dcp_size == 0], TOPK))  # all keys on rank 0
    rows.append(pick(every, 1))  # one rank holds the only key
    rows.append(pick(every[: dcp_size // 2 + 1], dcp_size // 2 + 1))
    for _ in range(3):
        rows.append(pick(every, TOPK))
    return torch.stack(rows).cuda()


def _attention(backend, pool, q, indices):
    valid = (indices >= 0).sum(-1).to(torch.int32)
    metadata = SimpleNamespace(
        dsa_cache_seqlens_int32=valid,
        flashmla_metadata=DeepseekSparseAttnBackend._compute_flashmla_metadata(
            backend, valid, 1
        ),
    )
    return DeepseekSparseAttnBackend._forward_flashmla_kv(
        backend,
        q_all=q,
        kv_cache=pool.get_key_buffer(0),
        v_head_dim=DV,
        sm_scale=SM_SCALE,
        layer=SimpleNamespace(tp_q_head_num=HEADS, head_dim=DIM),
        metadata=metadata,
        page_table_1=indices.to(torch.int32),
        dcp_partial=backend.attn_dcp_size > 1,
    )


class _BackendStub(SimpleNamespace):
    _dcp_flashmla_sched_meta = staticmethod(
        DeepseekSparseAttnBackend._dcp_flashmla_sched_meta
    )


def _backend(dcp_size: int):
    return _BackendStub(
        flashmla_kv_num_q_heads=_flashmla_kv_padded_heads(HEADS),
        physical_page_size=PAGE,
        kv_cache_dim=FP8_DS_MLA_BYTES,
        dsa_kv_cache_store_fp8=True,
        dsa_index_topk=TOPK,
        flashmla_kv_format="V32",
        attn_dcp_size=dcp_size,
    )


def _reference(pool, q, slots_of_row) -> torch.Tensor:
    """fp32 attention over the dequantized cache rows."""
    cache = dequantize_k_cache(
        pool.get_key_buffer(0).view(-1, 1, FP8_DS_MLA_BYTES)
    ).view(-1, DIM)
    out = []
    for i in range(q.shape[0]):
        rows = slots_of_row[i][slots_of_row[i] >= 0]
        k = cache[rows].float()
        p = torch.softmax(q[i].float() @ k.T * SM_SCALE, dim=-1)
        out.append(p @ k[:, :DV])
    return torch.stack(out)


def _errors(out, ref):
    out = out.float().reshape(ref.shape)
    rel = ((out - ref).norm() / ref.norm()).item()
    return rel, (out - ref).abs().max().item()


class TestDSADCPAttention(CustomTestCase):
    """Acceptance: relative L2 error <= 1.25x DCP1, max absolute error <= 2x
    DCP1. Rows whose keys all sit on one rank, rows with a single key, and short
    rows leave ranks without local keys."""

    def _run(self, dcp_size: int) -> None:
        gen = torch.Generator().manual_seed(dcp_size)
        k_nope = torch.randn(ATTN_TOKENS, 1, DV, generator=gen).cuda().bfloat16()
        k_rope = torch.randn(ATTN_TOKENS, 1, DIM - DV, generator=gen).cuda().bfloat16()
        positions = _row_positions(dcp_size, gen)
        q = torch.randn(positions.shape[0], HEADS, DIM, generator=gen).cuda().bfloat16()
        layer = SimpleNamespace(layer_id=0)

        ref_slots = _slots(PAGE, gen)
        with published_topology():
            ref_pool = _make_pool(ATTN_ROWS)
            ref_pool.set_mla_kv_buffer(layer, ref_slots, k_nope, k_rope)
            ref_idx = torch.where(positions >= 0, ref_slots[positions.clamp(min=0)], -1)
            dcp1 = _attention(_backend(1), ref_pool, q, ref_idx)
        reference = _reference(ref_pool, q, ref_idx)

        slots = _slots(PAGE * dcp_size, gen)
        idx = torch.where(positions >= 0, slots[positions.clamp(min=0)], -1)
        outs, lses = [], []
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            for rank in range(dcp_size):
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    pool = _make_pool(ATTN_ROWS)
                    pool.set_mla_kv_buffer(layer, slots, k_nope, k_rope)
                    o, lse = _attention(_backend(dcp_size), pool, q, idx)
                self.assertFalse(o.isnan().any() or lse.isnan().any())
                outs.append(o)
                lses.append(lse)
        lses = torch.stack(lses).float()

        # FlashMLA's LSE is natural-log; the K3.1 DSA merge must treat it so
        with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
            self.assertEqual(is_mla_dcp_lse_base_on_e("dsa"), dcp_size > 1)
        ctx = CPTritonContext()
        parts = [
            correct_attn_out(
                o,
                lses,
                r,
                ctx,
                o.new_empty(o.transpose(0, 1).shape, dtype=torch.float32),
                is_lse_base_on_e=True,
            )[0]
            for r, o in enumerate(outs)
        ]
        merged = torch.stack(parts).sum(0).transpose(0, 1)  # [B, H, D]
        rel1, max1 = _errors(dcp1, reference)
        rel_n, max_n = _errors(merged.bfloat16(), reference)
        self.assertLessEqual(rel_n, 1.25 * rel1)
        self.assertLessEqual(max_n, 2 * max1)

    def test_dcp8(self):
        self._run(8)

    def test_dcp2(self):
        self._run(2)


class TestDCPCandidatePrefix(CustomTestCase):
    def test_scatter_ignores_stale_padding_on_replay(self):
        rows, width = 5, 32768
        lens = torch.full((rows,), 8199, device="cuda", dtype=torch.int32)
        cols = torch.arange(TOPK, device="cuda")
        for world in (2, 8):
            with self.subTest(world=world):
                gathered = torch.empty(
                    world * rows, TOPK, 2, device="cuda", dtype=torch.int32
                )
                dense = torch.empty(rows, width, device="cuda")

                def scatter():
                    _scatter_candidates_kernel[(world * rows, 2)](
                        gathered,
                        rows,
                        lens,
                        dense,
                        width,
                        WORLD=world,
                        TOPK=TOPK,
                        BLOCK=1024,
                    )

                gathered.zero_()
                scatter()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    scatter()
                for lengths in (
                    [32768, 8199, 9, 1, 0],
                    [0, 1, 8193, 32768, 9],
                    [8199, 9, 0, 1, 32768],
                ):
                    lens.copy_(torch.tensor(lengths, device="cuda"))
                    # Stale padding points at a visible slot with a high score.
                    gathered[..., 0] = 0x7F7FFFFF
                    gathered[..., 1] = 0
                    expected = torch.full_like(dense, -float("inf"))
                    for rank in range(world):
                        for row, length in enumerate(lengths):
                            n = min((length + world - 1 - rank) // world, TOPK)
                            positions = cols[:n] * world + rank
                            scores = (positions + 1).float()
                            src = gathered[rank * rows + row, :n]
                            src[:, 0] = scores.view(torch.int32)
                            src[:, 1] = positions
                            expected[row, positions] = scores
                    dense.fill_(-float("inf"))
                    graph.replay()
                    torch.testing.assert_close(dense, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
