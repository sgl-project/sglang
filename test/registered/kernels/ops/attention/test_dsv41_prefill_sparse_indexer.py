"""The DeepGEMM two-level indexer on the dense prefill path against the torch
block selection and the dense implementation of the same protocol."""

import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    DeepGEMMPrefillData,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS, DIM = 32, 128
TOPK_BLOCKS, BLOCK, TOPK = 2048, 8, 512
PAGE = 128  # index-K pool page size under DeepGEMM's paged sparse logits
CASES = [(256, 40000), (64, 3000)]  # (query rows, context): above and below 2048 blocks

# The sparse kernel scores with bf16 weights and accumulation, the dense one in
# fp32; picks a few bf16 ulps from the selection floor can go either way.
MIN_OVERLAP = 0.95
FLOOR_TOLERANCE = 2**-5
# A tail table is a different DeepGEMM row set: its row pairing moves boundary
# scores as well.
MIN_TAIL_OVERLAP = 0.98


class Case(NamedTuple):
    dense: torch.Tensor  # fp32 [rows, width] scores of every row against its context
    lens: torch.Tensor  # [rows] int32 causal lengths
    data: DeepGEMMPrefillData
    kv: tuple  # the flattened index K (int8 [n, 64], int32 [n])
    k_cache: torch.Tensor  # [pages, PAGE, 1, 68] uint8, the index-K pool
    page_table: torch.Tensor  # [rows, pages] int32 at PAGE slots


def make_case(rows, ctx, seed) -> Case:
    """One request: `rows` query tokens whose causal lengths end at `ctx`, its
    index K in the pool page layout [pages, PAGE * 64 payload | PAGE * 4 scale]."""
    from deep_gemm import fp8_fp4_mqa_logits

    torch.manual_seed(seed)
    dev = "cuda"
    n_slots = (ctx + PAGE - 1) // PAGE * PAGE
    k_fp4, k_sf = quantize_fp4_indexer_tensor(
        torch.randn(n_slots, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    n_pages = n_slots // PAGE
    k_cache = torch.cat(
        [
            k_fp4.view(torch.uint8).reshape(n_pages, PAGE * 64),
            k_sf.view(torch.uint8).reshape(n_pages, PAGE * 4),
        ],
        1,
    ).view(n_pages, PAGE, 1, 68)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(
        torch.randn(rows * HEADS, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    q_fp4, q_sf = q_fp4.view(rows, HEADS, 64), q_sf.view(rows, HEADS)
    weights = torch.rand(rows, HEADS, device=dev)
    lens = torch.linspace(ctx - rows + 1, ctx, rows, device=dev).to(torch.int32)
    lens[0] = 0  # a query that sees no compressed position yet
    width = (ctx + 7) // 8 * 8
    starts = torch.zeros(rows, device=dev, dtype=torch.int32)
    dense = fp8_fp4_mqa_logits(
        (q_fp4, q_sf),
        (k_fp4[:width], k_sf[:width]),
        weights,
        starts,
        lens,
        False,
        width,
    )
    data = DeepGEMMPrefillData(
        k_slots=torch.arange(width, device=dev),
        request_starts=starts,
        lens_per_request=[ctx],
        rows_per_request=[rows],
        compress_lens=lens,
        q_fp4=q_fp4,
        q_sf=q_sf,
        weights=weights,
    )
    # KV pages of PAGE tokens at ratio 1: the index page table is the KV one
    page_table = (
        torch.arange(n_pages, device=dev, dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    return Case(dense, lens, data, (k_fp4[:width], k_sf[:width]), k_cache, page_table)


def rows_of(data: DeepGEMMPrefillData, idx: torch.Tensor) -> DeepGEMMPrefillData:
    """The operands of a subset of rows (one request)."""
    return msgspec.structs.replace(
        data,
        q_fp4=data.q_fp4[idx],
        q_sf=data.q_sf[idx],
        weights=data.weights[idx],
        compress_lens=data.compress_lens[idx],
        request_starts=data.request_starts[idx],
        rows_per_request=[idx.numel()],
    )


def reference_blocks(case: Case) -> torch.Tensor:
    """[rows, blocks] bool: the torch block selection of the dense scores."""
    j = torch.arange(case.dense.shape[1], device=case.dense.device)
    scores = case.dense.masked_fill(j[None, :] >= case.lens[:, None], -torch.inf)
    ids = select_candidate_block_ids(
        scores, case.lens[:, None], topk_blocks=TOPK_BLOCKS, block_size=BLOCK
    ).to(torch.int64)
    num_blocks = -(-scores.shape[1] // BLOCK)
    keep = torch.zeros(
        ids.shape[0], num_blocks + 1, dtype=torch.bool, device=ids.device
    )
    keep.scatter_(1, ids.masked_fill(ids < 0, num_blocks), True)
    return keep[:, :num_blocks]


def publish_sparse(case: Case):
    """(the DeepGEMM sparse table, the source layer's own top-k)."""
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
        publish_prefill_table,
    )

    own = case.data.empty_selection(TOPK)
    table = publish_prefill_table(
        data=case.data,
        kv=case.kv,
        index_page_table=case.page_table,
        index_page_size=PAGE,
        topk_blocks=TOPK_BLOCKS,
        out_positions=own,
    )
    return table, own


def select_sparse(table, data: DeepGEMMPrefillData, k_cache: torch.Tensor):
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
        select_prefill_table,
    )

    positions = data.empty_selection(TOPK)
    select_prefill_table(
        table=table, data=data, k_cache=k_cache, out_positions=positions
    )
    return positions


def publish_dense(case: Case):
    """(the CP implementation's block ids, the source layer's own top-k)."""
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        _publish_prefill_blocks,
    )

    own, blocks = _publish_prefill_blocks(
        data=case.data,
        kv=case.kv,
        topk=TOPK,
        topk_blocks=TOPK_BLOCKS,
        block_size=BLOCK,
    )
    return blocks, own


def select_dense(blocks, case: Case):
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        _consume_prefill_blocks,
    )

    return _consume_prefill_blocks(
        data=case.data,
        kv=case.kv,
        topk=TOPK,
        blocks=blocks,
        block_size=BLOCK,
    )


def picks(positions: torch.Tensor, row: int) -> set:
    return set(positions[row].tolist()) - {-1}


def assert_same_sparse_selection(tables, outputs, data, k_cache):
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import sparse_logits

    rows, heads = data.q_sf.shape
    logits = [
        sparse_logits(
            data.q_fp4.view(rows, 1, heads, 64),
            data.q_sf.view(rows, 1, heads),
            k_cache,
            data.weights.to(torch.bfloat16),
            table.schedule,
            TOPK_BLOCKS,
        )
        for table in tables
    ]
    for scores, table in zip(logits, tables):
        scores.masked_fill_(
            torch.arange(scores.shape[1], device="cuda")[None, :]
            >= table.valid_lens[:, None],
            -torch.inf,
        )
    torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
    expected = logits[0].topk(TOPK, dim=1).values.sort().values
    for table, output in zip(tables, outputs):
        relative = output - data.request_starts[:, None]
        active = output >= 0
        ordinal = torch.searchsorted(table.blocks, relative // BLOCK).long()
        block = table.blocks.gather(1, ordinal.clamp_max(TOPK_BLOCKS - 1))
        assert torch.all((block == relative // BLOCK) | ~active)
        assert torch.all((relative < data.compress_lens[:, None]) | ~active)
        torch.testing.assert_close(
            active.sum(1), data.compress_lens.clamp_max(TOPK).long()
        )
        ordered = output.sort().values
        assert not torch.any(
            (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
        )
        columns = (ordinal * BLOCK + relative % BLOCK).clamp(0, logits[0].shape[1] - 1)
        values = logits[0].gather(1, columns).masked_fill(~active, -torch.inf)
        torch.testing.assert_close(values.sort().values, expected, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestPrefillSparseIndexer(CustomTestCase):
    @torch.inference_mode()
    def test_fused_source_ragged_tiled_replay(self):
        """Multiple requests preserve both selections through tiling and tail reuse."""
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax8_varlen
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring, sparse_table

        def separate(scores, lens, *, out_offsets, out_indices, out_block_keys):
            topk_transform_ragged_v2(
                scores, lens, out_offsets=out_offsets, out_indices=out_indices
            )
            amax8_varlen(scores, lens, out=out_block_keys)

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                torch.manual_seed(900 + ratio)
                lengths, row_counts = [513, 20003], [8, 12]
                rows = sum(row_counts)
                n_pages = (max(lengths) + PAGE - 1) // PAGE
                pages = torch.randperm(2 * n_pages, device="cuda").view(2, n_pages)
                slots = (
                    pages[:, :, None] * PAGE + torch.arange(PAGE, device="cuda")
                ).view(2, -1)
                req_to_token = (
                    slots[:, :, None] * ratio + torch.arange(ratio, device="cuda")
                ).view(2, -1)
                positions = torch.cat(
                    [
                        torch.arange(n * ratio - count, n * ratio, device="cuda")
                        for n, count in zip(lengths, row_counts)
                    ]
                )
                positions[0] = -1
                q = torch.randn(rows, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
                weights = torch.rand(rows, HEADS, device="cuda")
                inputs = SimpleNamespace(
                    compress_ratio=ratio,
                    indexer=SimpleNamespace(
                        queries=lambda *_: q, head_weights=lambda _: weights
                    ),
                    positions=positions,
                    seq_lens_cpu=[n * ratio for n in lengths],
                    rows_per_request=row_counts,
                    rows_per_request_device=torch.tensor(row_counts, device="cuda"),
                    req_pool_indices=torch.arange(2, device="cuda"),
                    q_lora=None,
                    freqs_cis=torch.empty(max(lengths) * ratio, device="cuda"),
                    x=None,
                )
                data = scoring.get_deep_gemm_prefill_data(inputs, req_to_token)
                k, sf = quantize_fp4_indexer_tensor(
                    torch.randn(
                        2 * n_pages * PAGE, DIM, device="cuda", dtype=torch.bfloat16
                    ),
                    rne=True,
                )
                k_cache = torch.cat(
                    [
                        k.view(torch.uint8).view(-1, PAGE * 64),
                        sf.view(torch.uint8).view(-1, PAGE * 4),
                    ],
                    dim=1,
                ).view(-1, PAGE, 1, 68)
                kv = (k[data.k_slots], sf[data.k_slots])
                page_table = torch.repeat_interleave(
                    pages.to(torch.int32), inputs.rows_per_request_device, dim=0
                )
                own = [data.empty_selection(TOPK) for _ in range(2)]
                consumed = [data.empty_selection(TOPK) for _ in range(2)]

                def run(index):
                    table = sparse_table.publish_prefill_table(
                        data=data,
                        kv=kv,
                        index_page_table=page_table,
                        index_page_size=PAGE,
                        topk_blocks=TOPK_BLOCKS,
                        out_positions=own[index],
                    )
                    sparse_table.select_prefill_table(
                        table=table,
                        data=data,
                        k_cache=k_cache,
                        out_positions=consumed[index],
                    )
                    return table

                budget = 4 * ((max(lengths) + 7) // 8 * 8) * 4
                with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget):
                    with patch.object(
                        sparse_table, "topk_transform_ragged_amax8", separate
                    ):
                        expected = run(0)
                    actual = run(1)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        sparse_table.select_prefill_table(
                            table=actual,
                            data=data,
                            k_cache=k_cache,
                            out_positions=consumed[1],
                        )
                    graph.replay()
                    for field in ("blocks", "phys_blocks", "valid_lens"):
                        torch.testing.assert_close(
                            getattr(actual, field),
                            getattr(expected, field),
                            rtol=0,
                            atol=0,
                        )
                    torch.testing.assert_close(
                        own[0].sort().values, own[1].sort().values, rtol=0, atol=0
                    )
                    assert_same_sparse_selection(
                        [expected, actual], consumed, data, k_cache
                    )
                    tails = [table.tail([2, 3]) for table in (expected, actual)]
                    idx = torch.tensor([6, 7, 17, 18, 19], device="cuda")
                    tail_data = msgspec.structs.replace(
                        rows_of(data, idx), rows_per_request=[2, 3]
                    )
                    tail_outputs = [select_sparse(t, tail_data, k_cache) for t in tails]
                    assert_same_sparse_selection(
                        tails, tail_outputs, tail_data, k_cache
                    )

    @torch.inference_mode()
    def test_publish_prefill_is_the_torch_block_selection(self):
        """The published blocks equal `select_candidate_block_ids` block for block
        (ragged lengths, an empty row, block counts above and below 2048), and the
        source's own top-k from the same tiled pass is the dense one."""
        for rows, ctx in CASES:
            with self.subTest(rows=rows, ctx=ctx):
                case = make_case(rows, ctx, seed=rows + ctx)
                table, own = publish_sparse(case)
                expected = reference_blocks(case)
                nb = expected.shape[1]
                # INT32_MAX padding lands in a spare column instead of a block
                got = torch.zeros(rows, nb + 1, dtype=torch.bool, device="cuda")
                got.scatter_(1, table.blocks.clamp_max(nb).long(), True)
                self.assertTrue(torch.equal(got[:, :nb], expected))
                self.assertFalse(got[0, :nb].any(), "an empty row keeps no block")
                _, own_dense = publish_dense(case)
                for r in range(rows):
                    self.assertEqual(picks(own, r), picks(own_dense, r), r)

    @torch.inference_mode()
    def test_select_prefill_matches_the_dense_implementation(self):
        """A consumer scoring its blocks sparsely picks what the dense
        implementation picks from the same inputs, up to bf16 rounding at the
        selection floor, and never a position outside its blocks."""
        for rows, ctx in CASES:
            with self.subTest(rows=rows, ctx=ctx):
                case = make_case(rows, ctx, seed=rows * 3 + ctx)
                got = select_sparse(publish_sparse(case)[0], case.data, case.k_cache)
                want = select_dense(publish_dense(case)[0], case)
                keep = reference_blocks(case).repeat_interleave(BLOCK, dim=1)
                self.assertEqual(picks(got, 0), set(), "an empty row selects nothing")
                for r in range(1, rows):
                    g, w = picks(got, r), picks(want, r)
                    self.assertEqual(len(g), len(w), r)
                    self.assertTrue(
                        keep[r, list(g)].all(), f"row {r}: outside its blocks"
                    )
                    self.assertGreaterEqual(len(g & w), MIN_OVERLAP * len(w), r)
                    floor = case.dense[r, list(w)].min()
                    self.assertTrue(
                        (
                            case.dense[r, list(g)]
                            >= floor - floor.abs() * FLOOR_TOLERANCE
                        ).all(),
                        r,
                    )

    @torch.inference_mode()
    def test_prefill_tail_rebuilds_the_last_rows(self):
        """The tail table carries the last rows' blocks and lengths and selects
        what the full table selects for those rows."""
        rows, tail = 128, 16
        case = make_case(rows, 20000, seed=7)
        table, _ = publish_sparse(case)
        idx = torch.arange(rows - tail, rows, device="cuda")
        sub = table.tail([tail])
        self.assertTrue(torch.equal(sub.blocks, table.blocks[idx]))
        self.assertTrue(torch.equal(sub.valid_lens, table.valid_lens[idx]))
        self.assertTrue(torch.equal(sub.compress_lens, table.compress_lens[idx]))

        full = select_sparse(table, case.data, case.k_cache)
        part = select_sparse(sub, rows_of(case.data, idx), case.k_cache)
        for r in range(tail):
            a, b = picks(full, rows - tail + r), picks(part, r)
            self.assertGreaterEqual(len(a & b), MIN_TAIL_OVERLAP * len(a), r)

    def test_block_ids_tail_does_not_sync_the_host(self):
        """The late-layer tail of published block ids is cut on the device
        without a host sync while earlier work is still queued."""
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            BlockIds,
        )

        busy = torch.randn(4096, 4096, device="cuda")
        for _ in range(8):
            busy = busy @ busy
        blocks = torch.arange(40, dtype=torch.int32, device="cuda").view(20, 2)
        ids = BlockIds(blocks=blocks, rows_per_request=[5, 0, 10, 5])
        torch.cuda.set_sync_debug_mode("error")
        try:
            tail = ids.tail([2, 0, 3, 5])
        finally:
            torch.cuda.set_sync_debug_mode("default")
        rows = [3, 4, 12, 13, 14, 15, 16, 17, 18, 19]
        torch.testing.assert_close(tail.blocks, blocks[rows])


if __name__ == "__main__":
    unittest.main()
