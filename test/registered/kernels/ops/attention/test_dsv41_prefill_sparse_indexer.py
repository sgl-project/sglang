"""The DeepGEMM two-level indexer on the dense prefill path against the torch
block selection and the dense implementation of the same protocol."""

import unittest
from types import SimpleNamespace
from typing import NamedTuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMPrefillData
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


def make_multi_case(rows_per_request, ctx, seed) -> Case:
    """Several requests over one index-K pool, each owning its own slot range, so
    their rows carry different request ids. `dense` is left empty: this fixture is
    for comparing two schedules of the same rows, not against a scalar reference."""
    torch.manual_seed(seed)
    dev = "cuda"
    rows = sum(rows_per_request)
    slots = (ctx + PAGE - 1) // PAGE * PAGE
    pages = slots // PAGE
    n_req = len(rows_per_request)
    k_fp4, k_sf = quantize_fp4_indexer_tensor(
        torch.randn(slots * n_req, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    k_cache = torch.cat(
        [
            k_fp4.view(torch.uint8).reshape(pages * n_req, PAGE * 64),
            k_sf.view(torch.uint8).reshape(pages * n_req, PAGE * 4),
        ],
        1,
    ).view(pages * n_req, PAGE, 1, 68)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(
        torch.randn(rows * HEADS, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    lens, starts, table = [], [], []
    for r, n in enumerate(rows_per_request):
        lens.append(torch.linspace(ctx - n + 1, ctx, n, device=dev))
        starts.append(torch.full((n,), r * slots, device=dev))
        table.append(
            (torch.arange(pages, device=dev) + r * pages).expand(n, -1).contiguous()
        )
    cat = lambda xs, d: (
        torch.cat(xs).to(d) if xs else torch.empty(0, device=dev, dtype=d)
    )
    data = DeepGEMMPrefillData(
        k_slots=torch.arange(slots * n_req, device=dev),
        request_starts=cat(starts, torch.int32),
        lens_per_request=[ctx] * n_req,
        rows_per_request=list(rows_per_request),
        compress_lens=cat(lens, torch.int32),
        q_fp4=q_fp4.view(rows, HEADS, 64),
        q_sf=q_sf.view(rows, HEADS),
        weights=torch.rand(rows, HEADS, device=dev),
    )
    page_table = (
        torch.cat(table).to(torch.int32)
        if table
        else torch.empty(0, pages, device=dev, dtype=torch.int32)
    )
    return Case(
        torch.empty(0), data.compress_lens, data, (k_fp4, k_sf), k_cache, page_table
    )


def retable(table, request_ids: torch.Tensor):
    """The same rows scheduled with a different request-id vector."""
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
        _build_prefill_table,
    )

    return _build_prefill_table(
        # sort_candidate_blocks works in place; keep the source table intact.
        blocks=table.blocks.clone(),
        compress_lens=table.compress_lens,
        page_table=table.page_table,
        page_size=table.page_size,
        request_ids=request_ids,
        rows_per_request=table.rows_per_request,
        q_dtype=table.q_dtype,
        valid_lens=table.valid_lens,
    )


def per_request_ids(rows_per_request, device) -> torch.Tensor:
    """The ids the schedule carried before the row-pair ones: one per request."""
    counts = torch.tensor(rows_per_request, device=device)
    return torch.repeat_interleave(
        torch.arange(len(rows_per_request), device=device), counts
    ).to(torch.int32)


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


def case_rows(case: Case, rows: list[int], counts: list[int]) -> Case:
    """Subset query rows while retaining the requests' global index-K pool."""
    idx = torch.tensor(rows, device=case.lens.device, dtype=torch.int64)
    data = msgspec.structs.replace(rows_of(case.data, idx), rows_per_request=counts)
    return case._replace(
        dense=case.dense[idx] if case.dense.numel() else case.dense,
        lens=data.compress_lens,
        data=data,
        page_table=case.page_table[idx].contiguous(),
    )


def cp_rows(rows_per_request, rank, cp_size, tail=None):
    """Global round-robin CP rows, optionally restricted to each request's tail."""
    rows, counts, start = [], [], 0
    for count in rows_per_request:
        end = start + count
        first = start if tail is None else max(start, end - tail)
        selected = [row for row in range(first, end) if row % cp_size == rank]
        rows.extend(selected)
        counts.append(len(selected))
        start = end
    return rows, counts


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
    """(the dense fallback's block ids, the source layer's own top-k)."""
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


def logits_of(table, data: DeepGEMMPrefillData, k_cache: torch.Tensor):
    """The paged sparse logits this table's schedule produces."""
    from sglang.kernels.ops.attention.dsv4.index_logits import sparse_logits

    rows, heads = data.q_sf.shape
    return sparse_logits(
        data.q_fp4.view(rows, 1, heads, 64),
        data.q_sf.view(rows, 1, heads),
        k_cache,
        data.weights.to(torch.bfloat16),
        table.schedule,
        table.blocks.shape[1],
    )


def picks(positions: torch.Tensor, row: int) -> set:
    return set(positions[row].tolist()) - {-1}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestPrefillSparseIndexer(CustomTestCase):
    def assert_cp_selection(self, table, case, got, dense_blocks):
        from deep_gemm import fp8_fp4_mqa_logits

        want = select_dense(dense_blocks, case)
        want = torch.where(want >= 0, want - case.data.request_starts[:, None], -1)
        dense = fp8_fp4_mqa_logits(
            (case.data.q_fp4, case.data.q_sf),
            case.kv,
            case.data.weights,
            case.data.request_starts,
            case.data.request_starts + case.lens,
            False,
            -(-max(case.data.lens_per_request) // BLOCK) * BLOCK,
        )
        for row in range(case.data.num_rows):
            selected, expected = picks(got, row), picks(want, row)
            self.assertEqual(len(selected), len(expected), row)
            self.assertGreaterEqual(
                len(selected & expected), MIN_OVERLAP * len(expected), row
            )
            kept_blocks = set(table.blocks[row].tolist()) - {2147483647}
            self.assertTrue(all(p // BLOCK in kept_blocks for p in selected), row)
            self.assertTrue(all(0 <= p < int(case.lens[row]) for p in selected), row)
            if expected:
                floor = dense[row, sorted(expected)].min()
                self.assertTrue(
                    (
                        dense[row, sorted(selected)]
                        >= floor - floor.abs() * FLOOR_TOLERANCE
                    ).all(),
                    row,
                )

        # Attention's output rows may include CP padding. Map logical positions
        # through the actual pool pages, not through request ids or pair ids.
        raw = torch.full((got.shape[0] + 3, TOPK), -1, dtype=torch.int32, device="cuda")
        raw[: got.shape[0]].copy_(got)
        pages = torch.full_like(raw, -1)
        case.data.write_page_indices(
            SimpleNamespace(out_raw_indices=raw, out_page_indices=pages)
        )
        indices = got.clamp_min(0).long()
        physical = case.page_table.gather(1, indices // PAGE) * PAGE + indices % PAGE
        torch.testing.assert_close(
            pages[: got.shape[0]], physical.masked_fill(got < 0, -1).to(torch.int32)
        )
        self.assertTrue((pages[got.shape[0] :] == -1).all())

    @torch.inference_mode()
    def test_cp4_interleaved_requests_and_tail(self):
        # Odd request boundaries, a zero-row request and ranks that receive no
        # row from the one-token requests exercise both row-pair and CP layouts.
        counts = [129, 0, 141, 3, 1]
        for ctx in (3001, 20003):
            full = make_multi_case(counts, ctx, seed=ctx)
            full.lens[0] = 0
            full.lens[counts[0]] = 3  # partial block, fewer candidates than top-k
            for rank in range(4):
                with self.subTest(ctx=ctx, rank=rank):
                    rows, local_counts = cp_rows(counts, rank, 4)
                    case = case_rows(full, rows, local_counts)
                    table, own = publish_sparse(case)
                    blocks, dense_own = publish_dense(case)
                    dense_own = torch.where(
                        dense_own >= 0,
                        dense_own - case.data.request_starts[:, None],
                        -1,
                    )
                    for row in range(case.data.num_rows):
                        self.assertEqual(picks(own, row), picks(dense_own, row), row)
                        self.assertEqual(
                            set(table.blocks[row].tolist()) - {2147483647},
                            set(blocks[row].tolist()) - {-1},
                            row,
                        )
                    got = select_sparse(table, case.data, case.k_cache)
                    self.assert_cp_selection(table, case, got, blocks)

                    tail_rows, tail_counts = cp_rows(counts, rank, 4, tail=128)
                    local_rows = {global_row: i for i, global_row in enumerate(rows)}
                    idx = [local_rows[row] for row in tail_rows]
                    subcase = case_rows(case, idx, tail_counts)
                    sub = table.tail(tail_counts)
                    torch.testing.assert_close(sub.blocks, table.blocks[idx])
                    torch.testing.assert_close(sub.page_table, case.page_table[idx])
                    torch.testing.assert_close(sub.request_ids, table.request_ids[idx])
                    torch.testing.assert_close(sub.compress_lens, subcase.lens)
                    self.assertNotEqual(
                        sub.schedule.data_ptr(), table.schedule.data_ptr()
                    )
                    part = select_sparse(sub, subcase.data, subcase.k_cache)
                    self.assert_cp_selection(sub, subcase, part, blocks[idx])
                    # A multi-request tail can change BLOCK_Q pairing at each
                    # request boundary. Check against the same tail's fp32
                    # reference above, not a bitwise full-table selection.

    @torch.inference_mode()
    def test_cp_tail_consumer_graph_replay_reads_live_queries(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
            select_prefill_table,
        )

        counts = [133, 0, 141, 7, 1]
        full = make_multi_case(counts, 20003, seed=19)
        rows, local_counts = cp_rows(counts, 2, 4)
        case = case_rows(full, rows, local_counts)
        table, _ = publish_sparse(case)
        tail_rows, tail_counts = cp_rows(counts, 2, 4, tail=128)
        local_rows = {global_row: i for i, global_row in enumerate(rows)}
        tail = case_rows(case, [local_rows[row] for row in tail_rows], tail_counts)
        sub = table.tail(tail_counts)
        out = tail.data.empty_selection(TOPK)

        def consume():
            select_prefill_table(
                table=sub, data=tail.data, k_cache=tail.k_cache, out_positions=out
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                consume()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            consume()
        initial = None
        for shift in (0, 1):
            if shift:
                # Same addresses and shapes, different runtime query values.
                tail.data.q_fp4.copy_(tail.data.q_fp4.roll(1, dims=0))
                tail.data.q_sf.copy_(tail.data.q_sf.roll(1, dims=0))
                tail.data.weights.copy_(tail.data.weights.roll(1, dims=0))
            logits = logits_of(sub, tail.data, tail.k_cache)
            col = torch.arange(logits.shape[1], device="cuda")
            logits = logits.masked_fill(
                col[None, :] >= sub.valid_lens[:, None], -torch.inf
            )
            expected_scores = logits.topk(TOPK, dim=1).values.sort(1).values
            out.fill_(-2)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(((out >= 0) & (out < tail.lens[:, None])).all())
            self.assertTrue((out.sort(1).values.diff(dim=1) > 0).all())
            block_columns = torch.searchsorted(sub.blocks, out // BLOCK)
            torch.testing.assert_close(
                sub.blocks.gather(1, block_columns), out // BLOCK
            )
            columns = block_columns * BLOCK + out % BLOCK
            self.assertTrue((columns < sub.valid_lens[:, None]).all())
            # Unordered top-k may choose different indices tied at the bf16
            # floor, even between two eager calls. Compare score multisets.
            torch.testing.assert_close(
                logits.gather(1, columns).sort(1).values,
                expected_scores,
                rtol=0,
                atol=0,
            )
            if initial is None:
                initial = out.clone()
            else:
                overlap = sum(
                    len(picks(out, row) & picks(initial, row))
                    for row in range(out.shape[0])
                )
                self.assertLess(overlap, out.numel() // 2, "replay used stale queries")

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

    @torch.inference_mode()
    def test_row_pair_ids_schedule_the_same_logits_as_per_request_ids(self):
        """Row-pair ids only split what DeepGEMM's BLOCK_Q == 2 pairing already
        groups, so the sparse logits are the ones per-request ids produced. Odd
        row counts put a request boundary inside a pair; the single-row and
        zero-row requests are the degenerate groups."""
        for rows_per_request in ([5, 1, 4, 3], [2, 0, 7, 1]):
            with self.subTest(rows_per_request=rows_per_request):
                case = make_multi_case(
                    rows_per_request, 6000, seed=len(rows_per_request)
                )
                table, _ = publish_sparse(case)
                flat = retable(table, per_request_ids(rows_per_request, "cuda"))
                a = logits_of(table, case.data, case.k_cache)
                b = logits_of(flat, case.data, case.k_cache)
                # Past a row's valid length the logits are an unread tail.
                j = torch.arange(a.shape[-1], device=a.device)
                valid = j[None, :] < table.valid_lens[:, None]
                self.assertTrue(torch.equal(a[valid], b[valid]))

    @torch.inference_mode()
    def test_prefill_tail_starting_inside_a_pair(self):
        """`tail()` slices the id vector instead of rebuilding it, so a tail whose
        first row is the second of a pair leaves that row's id alone in the tail,
        a group of one. The schedule must still carry those rows' blocks and
        select inside them. Asserted on the selection's shape rather than its
        overlap with the full table: the two row sets pair differently, which
        moves boundary scores by a pick or two."""
        rows, tail = 128, 15  # starts at row 113, the odd half of a pair
        case = make_case(rows, 20000, seed=11)
        table, _ = publish_sparse(case)
        idx = torch.arange(rows - tail, rows, device="cuda")
        sub = table.tail([tail])
        self.assertNotEqual(
            int(sub.request_ids[0]), int(sub.request_ids[1]), "tail starts mid-pair"
        )
        self.assertTrue(torch.equal(sub.blocks, table.blocks[idx]))
        self.assertTrue(torch.equal(sub.valid_lens, table.valid_lens[idx]))

        part = select_sparse(sub, rows_of(case.data, idx), case.k_cache)
        keep = reference_blocks(case).repeat_interleave(BLOCK, dim=1)
        for r in range(tail):
            row = rows - tail + r
            picked = sorted(picks(part, r))
            self.assertEqual(len(picked), min(TOPK, int(case.lens[row])), r)
            self.assertTrue(keep[row, picked].all(), f"row {r}: outside its blocks")
            self.assertLess(
                picked[-1], int(case.lens[row]), f"row {r}: past its length"
            )

    def test_block_ids_tail_does_not_sync_the_host(self):
        """The late-layer tail of published block ids is cut on the device
        without a host sync while earlier work is still queued."""
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds

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
