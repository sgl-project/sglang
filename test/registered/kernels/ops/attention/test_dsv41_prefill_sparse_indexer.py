"""The DeepGEMM two-level indexer on the dense prefill path against the torch
block selection and the dense implementation of the same protocol."""

import unittest
from typing import NamedTuple

import msgspec
import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillIndexerInputs,
    make_candidate_indexer,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
    DenseCandidateIndexer,
)
from sglang.srt.runtime_context import get_parallel
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
    inputs: PrefillIndexerInputs


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
    inputs = PrefillIndexerInputs(
        q_fp4=q_fp4,
        q_sf=q_sf,
        weights=weights,
        compress_lens=lens,
        request_starts=starts,
        lens_per_request=[ctx],
        rows_per_request=[rows],
        kv=(k_fp4[:width], k_sf[:width]),
        k_cache=k_cache,
        page_size=PAGE,
        # KV pages of PAGE tokens at ratio 1: the index page table is the KV one
        kv_page_table=torch.arange(n_pages, device=dev, dtype=torch.int32)
        .expand(rows, -1)
        .contiguous(),
        kv_page_size=PAGE,
        compress_ratio=1,
    )
    return Case(dense, lens, inputs)


def rows_of(inputs: PrefillIndexerInputs, idx: torch.Tensor) -> PrefillIndexerInputs:
    """The inputs of a subset of rows (one request)."""
    return msgspec.structs.replace(
        inputs,
        q_fp4=inputs.q_fp4[idx],
        q_sf=inputs.q_sf[idx],
        weights=inputs.weights[idx],
        compress_lens=inputs.compress_lens[idx],
        request_starts=inputs.request_starts[idx],
        rows_per_request=[idx.numel()],
        kv_page_table=inputs.kv_page_table[idx],
    )


def reference_blocks(case: Case) -> torch.Tensor:
    """[rows, blocks] bool: the torch block selection of the dense scores."""
    j = torch.arange(case.dense.shape[1], device=case.dense.device)
    scores = case.dense.masked_fill(j[None, :] >= case.lens[:, None], -torch.inf)
    keep = select_candidate_blocks(
        scores, case.lens[:, None], topk_blocks=TOPK_BLOCKS, block_size=BLOCK
    )
    return keep.unflatten(1, (-1, BLOCK)).any(-1)


def publish(indexer, inputs):
    """(published metadata, the source layer's own top-k)."""
    own = torch.full((inputs.num_rows, TOPK), -1, dtype=torch.int32, device="cuda")
    return indexer.publish_prefill(inputs, own), own


def select(indexer, published, inputs):
    positions = torch.full(
        (inputs.num_rows, TOPK), -1, dtype=torch.int32, device="cuda"
    )
    indexer.select_prefill(published, inputs, positions)
    return positions


def picks(positions: torch.Tensor, row: int) -> set:
    return set(positions[row].tolist()) - {-1}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestPrefillSparseIndexer(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
            DeepGemmCandidateIndexer,
        )

        self.sparse = DeepGemmCandidateIndexer(TOPK_BLOCKS, BLOCK)
        self.dense = DenseCandidateIndexer(TOPK_BLOCKS, BLOCK)

    @torch.inference_mode()
    def test_publish_prefill_is_the_torch_block_selection(self):
        """The published blocks equal `select_candidate_blocks` block for block
        (ragged lengths, an empty row, block counts above and below 2048), and the
        source's own top-k from the same tiled pass is the dense one."""
        for rows, ctx in CASES:
            with self.subTest(rows=rows, ctx=ctx):
                case = make_case(rows, ctx, seed=rows + ctx)
                table, own = publish(self.sparse, case.inputs)
                expected = reference_blocks(case)
                nb = expected.shape[1]
                # INT32_MAX padding lands in a spare column instead of a block
                got = torch.zeros(rows, nb + 1, dtype=torch.bool, device="cuda")
                got.scatter_(1, table.blocks.clamp_max(nb).long(), True)
                self.assertTrue(torch.equal(got[:, :nb], expected))
                self.assertFalse(got[0, :nb].any(), "an empty row keeps no block")
                _, own_dense = publish(self.dense, case.inputs)
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
                got = select(
                    self.sparse, publish(self.sparse, case.inputs)[0], case.inputs
                )
                want = select(
                    self.dense, publish(self.dense, case.inputs)[0], case.inputs
                )
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
        table, _ = publish(self.sparse, case.inputs)
        idx = torch.arange(rows - tail, rows, device="cuda")
        sub = self.sparse.prefill_tail(table, [tail])
        self.assertTrue(torch.equal(sub.blocks, table.blocks[idx]))
        self.assertTrue(torch.equal(sub.valid_lens, table.valid_lens[idx]))
        self.assertTrue(torch.equal(sub.compress_lens, table.compress_lens[idx]))

        full = select(self.sparse, table, case.inputs)
        part = select(self.sparse, sub, rows_of(case.inputs, idx))
        for r in range(tail):
            a, b = picks(full, rows - tail + r), picks(part, r)
            self.assertGreaterEqual(len(a & b), MIN_TAIL_OVERLAP * len(a), r)


def _cp_indexer(cp_size=1):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("DeepGEMM paged sparse MQA logits need SM100")
    with get_parallel().override(attn_cp_size=cp_size):
        return make_candidate_indexer(TOPK_BLOCKS, BLOCK)


def _take_prefill_rows(inputs, rows, counts):
    return msgspec.structs.replace(
        inputs,
        q_fp4=inputs.q_fp4[rows].contiguous(),
        q_sf=inputs.q_sf[rows].contiguous(),
        weights=inputs.weights[rows].contiguous(),
        compress_lens=inputs.compress_lens[rows].contiguous(),
        request_starts=inputs.request_starts[rows].contiguous(),
        kv_page_table=inputs.kv_page_table[rows].contiguous(),
        rows_per_request=counts,
    )


def _cp_case(ratio):
    counts, contexts = [1, 47, 48], [256, 20224, 17920]
    cases = [
        make_case((n + 3) // 4 * 4, ctx, seed=ctx).inputs
        for n, ctx in zip(counts, contexts)
    ]
    kv_page = 2 * PAGE
    pages_per_request = [ctx * ratio // kv_page for ctx in contexts]
    physical_pages = torch.randperm(sum(pages_per_request), device="cuda")
    packed = torch.cat([c.k_cache for c in cases])
    cache = torch.empty_like(packed)
    cache.view(-1, 2 // ratio, PAGE, 1, 68)[physical_pages] = packed.view(
        -1, 2 // ratio, PAGE, 1, 68
    )
    req_to_token = torch.zeros(
        3, max(contexts) * ratio, dtype=torch.int64, device="cuda"
    )
    page_table = torch.zeros(
        3, max(pages_per_request), dtype=torch.int32, device="cuda"
    )
    for request, pages in enumerate(physical_pages.split(pages_per_request)):
        page_table[request, : pages.numel()] = pages
        slots = pages[:, None] * kv_page + torch.arange(kv_page, device="cuda")
        req_to_token[request, : slots.numel()] = slots.flatten()
    request_ids = torch.repeat_interleave(
        torch.arange(3, device="cuda"), torch.tensor(counts, device="cuda")
    )
    seq_lens = torch.cat(
        [
            torch.arange(ctx * ratio - n + 1, ctx * ratio + 1, device="cuda")
            for n, ctx in zip(counts, contexts)
        ]
    ).int()
    inputs = PrefillIndexerInputs(
        q_fp4=torch.cat([c.q_fp4[:n] for c, n in zip(cases, counts)]),
        q_sf=torch.cat([c.q_sf[:n] for c, n in zip(cases, counts)]),
        weights=torch.cat([c.weights[:n] for c, n in zip(cases, counts)]),
        compress_lens=seq_lens // ratio,
        request_starts=torch.tensor(
            [0, contexts[0], sum(contexts[:2])], device="cuda", dtype=torch.int32
        )[request_ids],
        lens_per_request=contexts,
        rows_per_request=counts,
        kv=tuple(torch.cat([c.kv[i] for c in cases]) for i in range(2)),
        k_cache=cache,
        page_size=PAGE,
        kv_page_table=page_table[request_ids],
        kv_page_size=kv_page,
        compress_ratio=ratio,
    )
    return inputs, req_to_token, request_ids


@pytest.mark.parametrize("ratio,cp_size", [(1, 2), (2, 4)])
@torch.inference_mode()
def test_cp_prefill_matches_dense_with_shuffled_pages(ratio, cp_size):
    """Sparse selection on CP-local rows must match dense scores on one GPU."""
    indexer, dense = _cp_indexer(cp_size), DenseCandidateIndexer(TOPK_BLOCKS, BLOCK)
    inputs, req_to_token, request_ids = _cp_case(ratio)
    for rank in range(cp_size):
        rows = torch.arange(rank, inputs.num_rows, cp_size, device="cuda")
        ids = request_ids[rows]
        counts = torch.bincount(ids, minlength=3).tolist()
        local = _take_prefill_rows(inputs, rows, counts)
        table, own = publish(indexer, local)
        reference, own_dense = publish(dense, local)
        assert torch.equal(own.sort().values, own_dense.sort().values)
        consumer = msgspec.structs.replace(
            local,
            q_fp4=local.q_fp4.roll(1, 0),
            q_sf=local.q_sf.roll(1, 0),
            weights=local.weights.roll(1, 0),
        )
        scores = torch.empty(0, device="cuda")
        tail_counts = [0, min(4, counts[1]), min(4, counts[2])]
        ends = torch.tensor(counts, device="cuda").cumsum(0).tolist()
        tail_rows = torch.cat(
            [
                torch.arange(end - n, end, device="cuda")
                for end, n in zip(ends, tail_counts)
                if n
            ]
        )
        for tail, selected_rows, lengths in (
            (False, torch.arange(local.num_rows, device="cuda"), counts),
            (True, tail_rows, tail_counts),
        ):
            sparse_table = indexer.prefill_tail(table, lengths) if tail else table
            dense_table = dense.prefill_tail(reference, lengths) if tail else reference
            selected = _take_prefill_rows(consumer, selected_rows, lengths)
            positions = select(indexer, sparse_table, selected)
            expected = select(dense, dense_table, selected)
            for row, source_row in enumerate(selected_rows.tolist()):
                start = selected.request_starts[row]
                got = positions[row][positions[row] >= 0].long() - start
                want = expected[row][expected[row] >= 0].long() - start
                assert (
                    got.numel()
                    == want.numel()
                    == min(TOPK, selected.compress_lens[row].item())
                )
                assert ((got >= 0) & (got < selected.compress_lens[row])).all()
                assert got.unique().numel() == got.numel()
                assert (
                    len(set(got.tolist()) & set(want.tolist()))
                    >= 0.95 * want.numel()
                )
                blocks = sparse_table.blocks[row]
                columns = torch.searchsorted(blocks, got // BLOCK)
                assert torch.equal(blocks[columns].long(), got // BLOCK)
                slots = (
                    sparse_table.phys_blocks[row, columns].long() * BLOCK + got % BLOCK
                )
                assert torch.equal(
                    slots, req_to_token[ids[source_row], got * ratio] // ratio
                )


@pytest.mark.parametrize("rank", [0, 1])
@torch.inference_mode()
def test_cp_signed_prefill_selects_topk_of_consumed_logits(rank, monkeypatch):
    """Check signed BF16 selection and CP page mapping using consumed logits."""
    from sglang.srt.layers.attention.dsv4 import candidate_indexer_deep_gemm as sparse

    indexer = _cp_indexer(4)
    inputs, req_to_token, request_ids = _cp_case(2)
    rows = torch.arange(rank, inputs.num_rows, 4, device="cuda")
    ids = request_ids[rows]
    counts = torch.bincount(ids, minlength=3).tolist()
    local = _take_prefill_rows(inputs, rows, counts)
    local = msgspec.structs.replace(
        local, weights=(2 * local.weights - 1).bfloat16().float()
    )
    table, own = publish(indexer, local)
    reference, own_dense = publish(DenseCandidateIndexer(TOPK_BLOCKS, BLOCK), local)
    assert torch.equal(own.sort().values, own_dense.sort().values)
    source_blocks = [row for blocks in reference.request_blocks for row in blocks]
    consumer = msgspec.structs.replace(
        local,
        q_fp4=local.q_fp4.roll(1, 0),
        q_sf=local.q_sf.roll(1, 0),
        weights=local.weights.roll(1, 0),
    )
    captured = []
    original = sparse.sparse_logits

    def capture_logits(*args, **kwargs):
        logits = original(*args, **kwargs)
        captured.append(logits)
        return logits

    monkeypatch.setattr(sparse, "sparse_logits", capture_logits)
    positions = select(indexer, table, consumer)
    assert len(captured) == 1 and captured[0].dtype == torch.bfloat16
    for row, length in enumerate(local.compress_lens.tolist()):
        blocks = table.blocks[row]
        want_blocks = source_blocks[row]
        assert torch.equal(
            blocks[blocks < (length + BLOCK - 1) // BLOCK],
            want_blocks[want_blocks >= 0].sort().values,
        )
        valid = positions[row] >= 0
        assert (positions[row, ~valid] == -1).all()
        got = positions[row, valid].long() - local.request_starts[row]
        assert got.numel() == got.unique().numel() == min(TOPK, length)
        assert ((got >= 0) & (got < length)).all()
        columns = torch.searchsorted(blocks, got // BLOCK)
        assert (columns < blocks.numel()).all()
        assert torch.equal(blocks[columns].long(), got // BLOCK)
        sparse_columns = columns * BLOCK + got % BLOCK
        valid_length = int(table.valid_lens[row])
        assert (sparse_columns < valid_length).all()
        scores = captured[0][row, :valid_length]
        # Equal scores may choose different indices; compare their multisets.
        torch.testing.assert_close(
            scores[sparse_columns].sort().values,
            torch.topk(scores, got.numel()).values.sort().values,
            rtol=0,
            atol=0,
        )
        slots = table.phys_blocks[row, columns].long() * BLOCK + got % BLOCK
        assert torch.equal(slots, req_to_token[ids[row], got * 2] // 2)


if __name__ == "__main__":
    unittest.main()
