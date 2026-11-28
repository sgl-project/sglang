"""gfx950 k-pool indexer TP split: the ranks' gathered top-k equals the unsplit
top-k row by row as sets, at TP4 and TP8, for decode and target verify."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
    IndexerKPool,
    _kpool_indexer_tp_split,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.kpool_fp8_index import build_kpool_topk_v2_plan
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")

# GLM-5.3-Flash's indexer
HEADS, D, BLOCK_KV, KPOOL, TOPK = 32, 128, 64, 4, 2048


class _GatherGroup:
    """Rank ``rank`` of a ``world_size`` group in one process: the all-gather
    records this rank's block and returns every block recorded so far."""

    def __init__(self, world_size: int, blocks: dict):
        self.world_size = world_size
        self.rank_in_group = 0
        self.blocks = blocks

    def all_gather_into_tensor(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        self.blocks[self.rank_in_group] = inp.clone()
        for r, part in enumerate(out.chunk(self.world_size)):
            if r in self.blocks:
                part.copy_(self.blocks[r])


def _make_indexer() -> IndexerKPool:
    indexer = IndexerKPool.__new__(IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_topk = TOPK
    indexer.index_kpool = KPOOL
    return indexer


def _make_inputs(reqs: int, next_n: int, pad: int):
    """Requests past 2048 history tokens, so the top-k drops pools."""
    n_real = reqs * next_n
    # tokens of each request's last row; row j of a request sees next_n - 1 - j fewer
    last_tokens = torch.randint(2600, 6000, (reqs,))
    row_tokens = torch.cat(
        [n - torch.arange(next_n - 1, -1, -1) for n in last_tokens.tolist()]
    )
    row_pools = row_tokens // KPOOL
    width = (int(row_pools.max()) + next_n + BLOCK_KV - 1) // BLOCK_KV + 1
    num_blocks = reqs * width + 1

    # one page: BLOCK_KV x D fp8 keys, then BLOCK_KV fp32 scales
    page = torch.empty(num_blocks, BLOCK_KV * (D + 4), dtype=torch.uint8)
    page[:, : BLOCK_KV * D] = (
        torch.randn(num_blocks, BLOCK_KV * D).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    page[:, BLOCK_KV * D :] = (
        torch.rand(num_blocks, BLOCK_KV).add(0.5).view(torch.uint8)
    )
    kv_cache = page.cuda().view(num_blocks, BLOCK_KV, 1, D + 4)

    req_tables = (torch.randperm(num_blocks - 1)[: reqs * width] + 1).view(reqs, width)
    row_tables = req_tables.repeat_interleave(next_n, dim=0).int().cuda().contiguous()
    pool_seqlens = row_pools.int().cuda()
    if next_n > 1:
        mqa_seqlens = pool_seqlens.view(reqs, next_n)[:, -1] + (next_n - 1)
        mqa_block_tables = row_tables.view(reqs, -1)
    else:
        mqa_seqlens, mqa_block_tables = pool_seqlens, row_tables

    max_tokens = int(row_tokens.max())
    page_table_1 = (
        torch.stack([torch.randperm(1 << 20)[:max_tokens] for _ in range(n_real + pad)])
        .int()
        .cuda()
    )
    return SimpleNamespace(
        n_real=n_real,
        num_q_padded=n_real + pad,
        q_fp8=torch.randn(n_real, HEADS, D, device="cuda").to(torch.float8_e4m3fn),
        kv_cache_fp8=kv_cache,
        weights=torch.randn(n_real, HEADS, device="cuda"),
        mqa_seqlens=mqa_seqlens,
        mqa_block_tables=mqa_block_tables,
        pool_max_seq_len=width * BLOCK_KV,
        pool_seqlens=pool_seqlens,
        seqlens_32=row_tokens.int().cuda(),
        page_table_1=page_table_1,
    )


def _metadata(x, *, fused: bool):
    return SimpleNamespace(
        force_unfused_topk=not fused,
        topk_transform_method=TopkTransformMethod.PAGED,
        attn_metadata=SimpleNamespace(page_table_1=x.page_table_1),
    )


def _unsplit_topk(indexer, x, next_n, metadata):
    logits = aiter_paged_mqa_logits(
        x.q_fp8,
        x.kv_cache_fp8,
        x.weights,
        x.mqa_seqlens,
        x.mqa_block_tables,
        x.pool_max_seq_len,
        preshuffle=True,
        kv_block_size=BLOCK_KV,
        next_n=next_n,
    )
    page_table_1, topk_offsets, _ = indexer._kpool_fused_topk_mapping(metadata)
    return indexer._topk_from_kpool_logits(
        logits=logits,
        pool_lens=x.pool_seqlens,
        topk_v2_plan=build_kpool_topk_v2_plan(x.pool_seqlens),
        seq_lens=x.seqlens_32,
        page_table=page_table_1,
        topk_offsets=topk_offsets,
        out_rows=x.num_q_padded if x.num_q_padded != x.n_real else None,
    )


def _split_topk(indexer, x, next_n, metadata, world):
    """Every rank's output; the second round sees all ranks' blocks."""
    mode = ForwardMode.TARGET_VERIFY if next_n > 1 else ForwardMode.DECODE
    group = _GatherGroup(world, blocks={})
    parallel = SimpleNamespace(attn_tp_group=group)
    outs = []
    with patch.object(dsa_indexer_kpool, "get_parallel", return_value=parallel):
        for round_ in range(2):
            for rank in range(world):
                group.rank_in_group = rank
                split = _kpool_indexer_tp_split(
                    SimpleNamespace(forward_mode=mode), x.n_real, next_n
                )
                if split is None:
                    return None
                out = indexer._get_topk_paged_tp_split(
                    split,
                    q_fp8=x.q_fp8,
                    kv_cache_fp8=x.kv_cache_fp8,
                    weights=x.weights,
                    mqa_seqlens=x.mqa_seqlens,
                    mqa_block_tables=x.mqa_block_tables,
                    pool_max_seq_len=x.pool_max_seq_len,
                    pool_seqlens=x.pool_seqlens,
                    seqlens_32=x.seqlens_32,
                    metadata=metadata,
                    num_q_padded=x.num_q_padded,
                    kv_block_size=BLOCK_KV,
                )
                if round_ == 1:
                    outs.append(out)
    return outs


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_gfx95_supported()
    and aiter_can_use_preshuffle_paged_mqa(),
    "requires gfx950 with AITER's preshuffle paged-MQA kernel",
)
class TestKpoolIndexerTpSplit(CustomTestCase):
    # (world, requests, next_n, padded rows): even, uneven with fewer or no
    # requests on the last ranks, and CUDA-graph padding
    CASES = (
        (4, 4, 1, 0),
        (4, 6, 6, 0),
        (4, 9, 1, 3),
        (8, 8, 1, 0),
        (8, 13, 6, 0),
        (8, 20, 1, 4),
        (8, 16, 6, 6),
    )

    @torch.no_grad()
    def test_split_matches_unsplit_as_sets(self):
        torch.manual_seed(0)
        indexer = _make_indexer()
        for world, reqs, next_n, pad in self.CASES:
            x = _make_inputs(reqs, next_n, pad)
            for fused in (True, False):
                with self.subTest(
                    world=world, reqs=reqs, next_n=next_n, pad=pad, fused=fused
                ):
                    metadata = _metadata(x, fused=fused)
                    want = _unsplit_topk(indexer, x, next_n, metadata)
                    outs = _split_topk(indexer, x, next_n, metadata, world)
                    self.assertIsNotNone(outs)
                    self.assertEqual(len(outs), world)
                    for rank, got in enumerate(outs):
                        self.assertEqual(got.shape, want.shape, f"rank {rank}")
                        self.assertTrue(
                            torch.equal(
                                got[: x.n_real].sort(dim=1).values,
                                want[: x.n_real].sort(dim=1).values,
                            ),
                            f"rank {rank}",
                        )
                        self.assertTrue(bool((got[x.n_real :] == -1).all()))

    def test_no_split_below_the_request_floor(self):
        x = SimpleNamespace(n_real=0)
        for world, reqs, next_n in ((4, 3, 1), (8, 7, 6), (8, 4, 1)):
            with self.subTest(world=world, reqs=reqs, next_n=next_n):
                x.n_real = reqs * next_n
                group = _GatherGroup(world, blocks={})
                mode = ForwardMode.TARGET_VERIFY if next_n > 1 else ForwardMode.DECODE
                with patch.object(
                    dsa_indexer_kpool,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_group=group),
                ):
                    self.assertIsNone(
                        _kpool_indexer_tp_split(
                            SimpleNamespace(forward_mode=mode), x.n_real, next_n
                        )
                    )


if __name__ == "__main__":
    unittest.main()
