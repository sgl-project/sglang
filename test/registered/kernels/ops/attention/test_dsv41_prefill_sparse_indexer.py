"""The DeepGEMM two-level indexer on the dense prefill path against the torch
block selection and the dense implementation of the same protocol."""

import unittest
from typing import NamedTuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillIndexerInputs,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
    DenseCandidateIndexer,
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

    @torch.inference_mode()
    def test_consumer_pairwise_with_torch_oracle(self):
        """Identical candidate tables isolate scorer rounding from publication."""
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            PrefillCandidateBlocks,
        )
        from sglang.srt.layers.attention.dsv4.triton_candidate_indexer import (
            TritonCandidateIndexer,
        )

        for rows, ctx in [(32, 20000), (16, 513)]:
            with self.subTest(rows=rows, ctx=ctx):
                case = make_case(rows, ctx, seed=ctx + 19)
                table, _ = publish(self.sparse, case.inputs)
                blocks = table.blocks.masked_fill(
                    table.blocks == torch.iinfo(torch.int32).max, -1
                )
                published = PrefillCandidateBlocks(request_blocks=[blocks])
                inputs = _torch_inputs(case.inputs)
                triton = TritonCandidateIndexer(
                    TOPK_BLOCKS, BLOCK, budget_bytes=32 << 20
                )
                results = [
                    select(self.sparse, table, case.inputs),
                    select(self.dense, published, case.inputs),
                    select(triton, published, inputs),
                    torch_reference_selection(inputs, blocks),
                ]
                keep = reference_blocks(case).repeat_interleave(BLOCK, dim=1)
                for result in results:
                    self.assertTrue((result[0] == -1).all())
                    for row in range(1, rows):
                        got = result[row][result[row] >= 0].long()
                        self.assertEqual(got.unique().numel(), got.numel())
                        self.assertTrue((got < case.lens[row]).all())
                        self.assertTrue(keep[row, got].all())
                for a in range(len(results)):
                    for b in range(a + 1, len(results)):
                        for row in range(1, rows):
                            got, want = picks(results[a], row), picks(results[b], row)
                            self.assertEqual(len(got), len(want))
                            self.assertGreaterEqual(
                                len(got & want), MIN_OVERLAP * len(want)
                            )
                            floor = case.dense[row, list(want)].min()
                            self.assertTrue(
                                (
                                    case.dense[row, list(got)]
                                    >= floor - floor.abs() * FLOOR_TOLERANCE
                                ).all()
                            )


def _torch_inputs(inputs):
    """Decode the exact packed operands used by the kernel cases, on device.

    This is test-only decoding, independent of the engine's cache accessors.
    No FP8 requantization or FP32 head accumulation is introduced.
    """
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        TritonPrefillInputs,
    )

    def unpack(payload, sf):
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=payload.device,
        )
        codes = payload.view(torch.uint8)
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
        exponents = sf.contiguous().view(torch.uint8).reshape(*sf.shape, 4).float()
        scales = torch.exp2(exponents - 127).repeat_interleave(32, dim=-1)
        return (lut[codes.long()] * scales).to(torch.bfloat16)

    keys = unpack(*inputs.kv)
    return TritonPrefillInputs(
        q=unpack(inputs.q_fp4, inputs.q_sf),
        weights=inputs.weights.to(torch.bfloat16),
        compress_lens=inputs.compress_lens,
        request_starts=inputs.request_starts,
        lens_per_request=inputs.lens_per_request,
        rows_per_request=inputs.rows_per_request,
        get_keys=lambda b: keys[: inputs.lens_per_request[b]],
    )


def torch_reference_selection(inputs, blocks):
    """Independent dense-and-mask Torch oracle; never calls a production scorer."""
    scores = torch.einsum("rhd,nd->rhn", inputs.q, inputs.get_keys(0))
    scores = (scores.relu() * inputs.weights[:, :, None]).sum(1).float()
    keep = torch.zeros_like(scores, dtype=torch.bool)
    for row, candidates in enumerate(blocks):
        for block in candidates[candidates >= 0].tolist():
            keep[row, block * BLOCK : (block + 1) * BLOCK] = True
    columns = torch.arange(scores.shape[1], device=scores.device)
    scores.masked_fill_(
        ~keep | (columns[None] >= inputs.compress_lens[:, None]), -torch.inf
    )
    values, indices = scores.topk(min(TOPK, scores.shape[1]), dim=1)
    out = torch.full(
        (inputs.num_rows, TOPK), -1, dtype=torch.int32, device=scores.device
    )
    out[:, : indices.shape[1]] = indices.masked_fill(~torch.isfinite(values), -1).int()
    return out


if __name__ == "__main__":
    unittest.main()
