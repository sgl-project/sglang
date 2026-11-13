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

from sglang.kernels.ops.attention.dsa.metadata import gather_physical_token_table
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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
        for width in (1000, WIDTH + 100):
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
