"""DSA page-table metadata on the unified memory pool, on GPU.

`req_to_token` holds virtual token ids there. The fused metadata builders, and
the eager path's gather, translate them through the full sub-pool's
virtual->physical page table as they fill the page tables:

    physical(t) = v2p[t // 64] * 64 + t % 64

in the same launches a static pool's metadata takes.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsa.metadata import (
    fused_dsa_decode_metadata,
    fused_dsa_draft_extend_metadata,
    fused_dsa_target_verify_metadata,
    gather_physical_token_table,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64
REQ_ROWS, WIDTH = 8, 4096
VIRTUAL_PAGES = REQ_ROWS * WIDTH // PAGE + 8
TOPK = 2048


def _virtual_pool(seed=0):
    """`req_to_token` over shuffled virtual pages, and a v2p scattering them
    over physical pages; virtual page 0 is the sink, the last row the -1
    sentinel."""
    g = torch.Generator().manual_seed(seed)
    pages = torch.randperm(VIRTUAL_PAGES - 1, generator=g)[: REQ_ROWS * WIDTH // PAGE]
    pages = (pages + 1).view(REQ_ROWS, WIDTH // PAGE)
    req_to_token = (pages[:, :, None] * PAGE + torch.arange(PAGE)).view(REQ_ROWS, WIDTH)
    v2p = torch.full((VIRTUAL_PAGES + 1,), -1, dtype=torch.int64)
    v2p[0] = 0
    v2p[1:VIRTUAL_PAGES] = torch.randperm(VIRTUAL_PAGES - 1, generator=g) + 3
    return req_to_token.to(torch.int32).cuda(), v2p.cuda()


def _physical(ids, v2p):
    ids = ids.to(torch.int64)
    return (v2p[ids // PAGE] * PAGE + ids % PAGE).to(torch.int32)


def _runner_ids(v2p):
    space = IdSpace(key=("full", 0), read_v2p=v2p, table=v2p)
    return SimpleNamespace(
        reads_are_translated=True,
        is_translating=True,
        page_size=PAGE,
        space=lambda kind: space if kind == IdSpaceKind.FULL else None,
    )


class _Tables:
    """Graph-shaped metadata buffers for ``rows`` query rows."""

    def __init__(self, bs, rows, *, wide):
        dev = "cuda"
        self.cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=dev)
        self.cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
        self.seqlens_expanded = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.dsa_cache_seqlens = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.dsa_cu_seqlens_k = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
        self.page_table_1 = (
            torch.full((rows, WIDTH), -7, dtype=torch.int32, device=dev)
            if wide
            else None
        )
        self.real_page_table = torch.full(
            (rows, WIDTH // PAGE), -7, dtype=torch.int32, device=dev
        )


class TestFusedMetadataTranslates(CustomTestCase):
    def _check(self, tables, expected, lengths):
        """Live prefixes hold `expected` (the reference translation, one row
        per query row); the tails are never read."""
        for row, length in enumerate(lengths):
            if tables.page_table_1 is not None:
                torch.testing.assert_close(
                    tables.page_table_1[row, :length], expected[row, :length]
                )
            pages = -(-length // PAGE)
            torch.testing.assert_close(
                tables.real_page_table[row, :pages],
                expected[row, :length:PAGE] // PAGE,
            )

    def test_decode(self):
        req_to_token, v2p = _virtual_pool()
        req = torch.tensor([5, 0, 7, 2], device="cuda")
        seq = torch.tensor([1, 64, 2049, 4000], device="cuda")
        expected = _physical(req_to_token[req], v2p)
        for wide in (True, False):
            with self.subTest(wide=wide):
                t = _Tables(4, 4, wide=wide)
                fused_dsa_decode_metadata(
                    seq_lens=seq,
                    req_pool_indices=req,
                    req_to_token=req_to_token,
                    cache_seqlens=t.cache_seqlens,
                    cu_seqlens_k=t.cu_seqlens_k,
                    page_table_1=t.page_table_1,
                    dsa_cache_seqlens=t.dsa_cache_seqlens,
                    dsa_cu_seqlens_k=t.dsa_cu_seqlens_k,
                    real_page_table=t.real_page_table,
                    bs=4,
                    max_len=WIDTH,
                    dsa_index_topk=TOPK,
                    physical_page_size=PAGE,
                    v2p=v2p,
                )
                self._check(t, expected, seq.tolist())

    def test_decode_without_v2p_keeps_ids(self):
        req_to_token, _ = _virtual_pool()
        req = torch.tensor([3, 1], device="cuda")
        seq = torch.tensor([100, 700], device="cuda")
        t = _Tables(2, 2, wide=True)
        fused_dsa_decode_metadata(
            seq_lens=seq,
            req_pool_indices=req,
            req_to_token=req_to_token,
            cache_seqlens=t.cache_seqlens,
            cu_seqlens_k=t.cu_seqlens_k,
            page_table_1=t.page_table_1,
            dsa_cache_seqlens=t.dsa_cache_seqlens,
            dsa_cu_seqlens_k=t.dsa_cu_seqlens_k,
            real_page_table=t.real_page_table,
            bs=2,
            max_len=WIDTH,
            dsa_index_topk=TOPK,
            physical_page_size=PAGE,
        )
        self._check(t, req_to_token[req], seq.tolist())

    def test_target_verify(self):
        req_to_token, v2p = _virtual_pool(1)
        next_n = 3
        req = torch.tensor([6, 1, 4], device="cuda")
        seq = torch.tensor([60, 129, 3000], device="cuda")
        expected = _physical(req_to_token[req], v2p).repeat_interleave(next_n, 0)
        lengths = (seq + next_n).repeat_interleave(next_n).tolist()
        for wide in (True, False):
            with self.subTest(wide=wide):
                t = _Tables(3, 3 * next_n, wide=wide)
                fused_dsa_target_verify_metadata(
                    seq_lens=seq,
                    req_pool_indices=req,
                    req_to_token=req_to_token,
                    cache_seqlens=t.cache_seqlens,
                    cu_seqlens_k=t.cu_seqlens_k,
                    page_table_1=t.page_table_1,
                    seqlens_expanded=t.seqlens_expanded,
                    dsa_cache_seqlens=t.dsa_cache_seqlens,
                    dsa_cu_seqlens_k=t.dsa_cu_seqlens_k,
                    real_page_table=t.real_page_table,
                    bs=3,
                    max_seqlen_k=WIDTH,
                    dsa_index_topk=TOPK,
                    physical_page_size=PAGE,
                    next_n=next_n,
                    v2p=v2p,
                )
                self._check(t, expected, lengths)

    def test_draft_extend(self):
        req_to_token, v2p = _virtual_pool(2)
        width = 4
        req = torch.tensor([2, 7], device="cuda")
        seq = torch.tensor([70, 2500], device="cuda")
        expected = _physical(req_to_token[req], v2p).repeat_interleave(width, 0)
        lengths = seq.repeat_interleave(width).tolist()
        for wide in (True, False):
            with self.subTest(wide=wide):
                t = _Tables(2, 2 * width, wide=wide)
                fused_dsa_draft_extend_metadata(
                    seq_lens=seq,
                    extend_seq_lens=torch.full(
                        (2,), width, dtype=torch.int32, device="cuda"
                    ),
                    req_pool_indices=req,
                    req_to_token=req_to_token,
                    cache_seqlens=t.cache_seqlens,
                    cu_seqlens_k=t.cu_seqlens_k,
                    page_table_1=t.page_table_1,
                    seqlens_expanded=t.seqlens_expanded,
                    dsa_cache_seqlens=t.dsa_cache_seqlens,
                    dsa_cu_seqlens_k=t.dsa_cu_seqlens_k,
                    real_page_table=t.real_page_table,
                    bs=2,
                    total_len=2 * width,
                    max_seqlen_k=WIDTH,
                    dsa_index_topk=TOPK,
                    physical_page_size=PAGE,
                    max_extend_len=width,
                    max_total_len=2 * width,
                    static_extend_len=True,
                    v2p=v2p,
                )
                self._check(t, expected, lengths)


def _graph_backend(req_to_token, runner_ids, *, drop_wide, bs):
    """A decode-graph DSA backend over real metadata kernels."""
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.__dict__.update(
        device=torch.device("cuda"),
        device_sm_major=torch.cuda.get_device_capability()[0],
        num_q_heads=64,
        physical_page_size=PAGE,
        real_page_span=PAGE,
        dsa_shards_index_k=False,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        dsa_index_topk=TOPK,
        dsa_index_kpool=1,
        speculative_num_draft_tokens=0,
        dsa_drop_wide_page_table=drop_wide,
        dsa_decode_impl="fa3",
        dsa_prefill_impl="fa3",
        enable_auto_select_prefill_impl=False,
        token_to_kv_pool=SimpleNamespace(index_page_size=PAGE),
        dsa_topk_backend=DSATopKBackend.TORCH,
        req_to_token=req_to_token,
        runner_ids=runner_ids,
        _arange_buf=torch.arange(bs + 1, device="cuda", dtype=torch.int32),
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: backend.device_sm_major * 10,
        _is_blackwell=lambda: backend.device_sm_major == 10,
    )
    backend.decode_cuda_graph_metadata = {
        "page_table": (
            None
            if drop_wide
            else torch.zeros(bs, WIDTH, device="cuda", dtype=torch.int32)
        ),
        "real_page_table": torch.zeros(
            bs, WIDTH // PAGE, device="cuda", dtype=torch.int32
        ),
        "cu_seqlens_q": backend._arange_buf,
    }
    backend._init_kpool_metadata_fusion()
    return backend


class TestDecodeGraphReplayTranslates(CustomTestCase):
    def test_replayed_tables_hold_physical_ids(self):
        """Capture builds the graph's buffers; each replay refills them with
        the batch's physical ids, out of graph, through the live v2p."""
        req_to_token, v2p = _virtual_pool(3)
        bs = 3
        for drop_wide in (False, True):
            with self.subTest(drop_wide=drop_wide):
                backend = _graph_backend(
                    req_to_token, _runner_ids(v2p), drop_wide=drop_wide, bs=bs
                )
                for req, seq in (([1, 4, 6], [5, 700, 4000]), ([0, 7, 2], [64, 1, 65])):
                    req = torch.tensor(req, device="cuda")
                    seq = torch.tensor(seq, device="cuda")
                    for _ in range(2):  # the first call captures
                        backend._apply_cuda_graph_metadata(
                            bs=bs,
                            req_pool_indices=req,
                            seq_lens=seq,
                            seq_lens_cpu=seq.cpu(),
                            forward_mode=ForwardMode.DECODE,
                            spec_info=None,
                        )
                    metadata = backend.forward_metadata
                    expected = _physical(req_to_token[req], v2p)
                    for row, length in enumerate(seq.tolist()):
                        if metadata.page_table_1 is not None:
                            torch.testing.assert_close(
                                metadata.page_table_1[row, :length],
                                expected[row, :length],
                            )
                        torch.testing.assert_close(
                            metadata.real_page_table[row, : -(-length // PAGE)],
                            expected[row, :length:PAGE] // PAGE,
                        )
                # The live mapping is read at every replay: move a page.
                page = int(req_to_token[2, 0]) // PAGE
                v2p[page] = 1
                req = torch.tensor([2, 2, 2], device="cuda")
                seq = torch.tensor([3, 3, 3], device="cuda")
                backend._apply_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=req,
                    seq_lens=seq,
                    seq_lens_cpu=seq.cpu(),
                    forward_mode=ForwardMode.DECODE,
                    spec_info=None,
                )
                self.assertEqual(
                    backend.forward_metadata.real_page_table[:, 0].tolist(), [1] * bs
                )
                v2p.copy_(_virtual_pool(3)[1])


class TestEagerGatherTranslates(CustomTestCase):
    def test_gather_matches_the_reference(self):
        req_to_token, v2p = _virtual_pool(4)
        req = torch.tensor([7, 0, 3], device="cuda")
        for width in (1, 64, 1000, WIDTH, WIDTH + 100):
            with self.subTest(width=width):
                table = gather_physical_token_table(req_to_token, req, width, v2p, PAGE)
                self.assertEqual(table.dtype, torch.int32)
                expected = _physical(req_to_token[req, :width], v2p)
                torch.testing.assert_close(table, expected, rtol=0, atol=0)

    def test_unmapped_pages_read_the_sink(self):
        req_to_token, v2p = _virtual_pool(5)
        v2p[int(req_to_token[2, 70]) // PAGE] = -1
        table = gather_physical_token_table(
            req_to_token, torch.tensor([2], device="cuda"), 192, v2p, PAGE
        )
        self.assertEqual(table[0, 64:128].tolist(), [0] * 64)


if __name__ == "__main__":
    unittest.main()
