"""The DeepSeek V4.1 index-layer top-k selections under every --dsa-topk-backend:
each selects the reference picks on tie-free scores, the unfused ones also across row
chunks, FlashInfer's index tie-break follows logical positions, and the FlashInfer
paths replay under CUDA graphs."""

import itertools
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4.v41_indexer import selector as selector_module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEV = "cuda"
K = 512
BLOCK = 8
INT32_MAX = torch.iinfo(torch.int32).max

# (backend, SGLANG_DSA_FUSE_TOPK, SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK, deterministic)
SGL_KERNEL = ("sgl-kernel", True, None, False)
NON_DEFAULT = [
    ("torch", True, None, False),
    ("flashinfer", True, None, False),
    ("flashinfer", True, "small", True),
    ("flashinfer", False, None, False),
    ("flashinfer", False, "large", False),
]
FLASHINFER_TIES = [
    ("flashinfer", fused, tie_break, False)
    for fused in (True, False)
    for tie_break in ("small", "large")
]


@contextmanager
def make_selector(backend, fused, tie_break, deterministic):
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(fused))
        stack.enter_context(
            envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.override(tie_break)
        )
        stack.enter_context(
            envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.override(deterministic)
        )
        yield selector_module.TopKSelector(DSATopKBackend(backend))


def distinct_scores(lens, width, dtype, *, seed, stride=None):
    """``[rows, width]`` scores, pairwise distinct within each row's first ``lens``
    columns and larger than all of them past it, so reading past a row's length
    changes the picks. ``stride`` lays the rows out as a padded view."""
    rows = lens.shape[0]
    if dtype == torch.bfloat16:
        # Consecutive positive bf16 bit patterns are distinct, increasing values.
        values = torch.arange(
            0x3000, 0x3000 + width, device=DEV, dtype=torch.int16
        ).view(torch.bfloat16)
    else:
        values = torch.arange(width, device=DEV, dtype=torch.float32) - width / 2
    g = torch.Generator(device=DEV).manual_seed(seed)
    order = torch.rand(rows, width, generator=g, device=DEV).argsort(dim=-1)
    past = torch.arange(width, device=DEV)[None, :] >= lens[:, None]
    scores = values[order].masked_fill(past, torch.finfo(dtype).max)
    storage = torch.zeros(rows, stride or width, dtype=dtype, device=DEV)
    storage[:, :width] = scores
    return storage[:, :width]


def peaked_scores(lens, width, *, seed):
    """bf16 rows too wide for distinct values whose top-K is still unique: up to 2K
    distinct peaks at random positions among each row's first ``lens``, -1 elsewhere
    in that range, and larger than all of them past it."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.full(
        (lens.shape[0], width),
        torch.finfo(torch.bfloat16).max,
        dtype=torch.bfloat16,
        device=DEV,
    )
    for row, length in enumerate(lens.tolist()):
        peaks = min(length, 2 * K)
        values = torch.arange(
            0x3000, 0x3000 + peaks, device=DEV, dtype=torch.int16
        ).view(torch.bfloat16)
        columns = torch.randperm(length, generator=g, device=DEV)[:peaks]
        scores[row, :length] = -1.0
        scores[row, columns] = values[torch.randperm(peaks, generator=g, device=DEV)]
    return scores


def reference_columns(scores, lens, k):
    """Each row's top-k columns among its first ``lens``; unique on distinct scores."""
    width = scores.shape[1]
    past = torch.arange(width, device=DEV)[None, :] >= lens[:, None]
    top = scores.float().masked_fill(past, -torch.inf).topk(min(k, width), dim=-1)
    return [
        {c for c, v in zip(cols, vals) if v != float("-inf")}
        for cols, vals in zip(top.indices.tolist(), top.values.tolist())
    ]


def through_table(columns, table, page_size):
    """``table[b, c // page_size] * page_size + c % page_size`` for each row's set."""
    table = table.tolist()
    return [
        {table[b][c // page_size] * page_size + c % page_size for c in row}
        for b, row in enumerate(columns)
    ]


def random_page_table(rows, pages, *, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.stack(
        [
            torch.randperm(4 * pages, generator=g, device=DEV)[:pages]
            for _ in range(rows)
        ]
    ).to(torch.int32)


def sparse_tables(valid_lens, topk_blocks, *, seed):
    """(logical, physical) ``[rows, topk_blocks]`` block tables of a published sparse
    table: ascending logical ids, distinct physical ids, INT32_MAX past each row's
    ``ceil(valid_lens / 8)`` kept blocks."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = valid_lens.shape[0]
    logical = torch.stack(
        [
            torch.randperm(4 * topk_blocks, generator=g, device=DEV)[:topk_blocks]
            .sort()
            .values
            for _ in range(rows)
        ]
    ).to(torch.int32)
    physical = random_page_table(rows, topk_blocks, seed=seed + 1)
    kept = (valid_lens + BLOCK - 1) // BLOCK
    past = torch.arange(topk_blocks, device=DEV)[None, :] >= kept[:, None]
    return logical.masked_fill(past, INT32_MAX), physical.masked_fill(past, INT32_MAX)


def block_scores_with_distinct_maxima(lens, width, *, seed):
    """bf16 scores whose block maxima are pairwise distinct: each block holds one
    distinct value above a common low floor."""
    rows = lens.shape[0]
    num_blocks = width // BLOCK
    g = torch.Generator(device=DEV).manual_seed(seed)
    maxima = torch.arange(
        0x3000, 0x3000 + num_blocks, device=DEV, dtype=torch.int16
    ).view(torch.bfloat16)
    order = torch.rand(rows, num_blocks, generator=g, device=DEV).argsort(dim=-1)
    scores = torch.full((rows, num_blocks, BLOCK), 2.0**-60, device=DEV)
    slot = torch.randint(0, BLOCK, (rows, num_blocks), generator=g, device=DEV)
    scores.scatter_(2, slot[:, :, None], maxima[order].float()[:, :, None])
    scores = scores.flatten(1).to(torch.bfloat16)
    past = torch.arange(width, device=DEV)[None, :] >= lens[:, None]
    return scores.masked_fill(past, torch.finfo(torch.bfloat16).max)


def reference_blocks(scores, lens, topk_blocks):
    """The source's block selection: block maxima over each row's first ``lens``
    positions, its newest block forced in."""
    width = scores.shape[1]
    past = torch.arange(width, device=DEV)[None, :] >= lens[:, None]
    keys = (
        scores.float().masked_fill(past, -torch.inf).unflatten(1, (-1, BLOCK)).amax(-1)
    )
    newest = (lens.long() + BLOCK - 1) // BLOCK - 1
    blocks = torch.arange(keys.shape[1], device=DEV)
    keys = keys.masked_fill((blocks[None, :] == newest[:, None]), torch.inf)
    return reference_columns(keys, (lens + BLOCK - 1) // BLOCK, topk_blocks)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "the V4.1 sgl-kernel top-k kernels and FlashInfer's DSA transforms run on SM100",
)
class TestDSV41TopKSelector(CustomTestCase):
    def assert_picks(self, out, expected):
        """Every slot written, ``min(k, len)`` distinct picks, ``-1`` past them."""
        rows = out.tolist()
        for b, (row, want) in enumerate(zip(rows, expected)):
            picked = [v for v in row if v != -1]
            self.assertNotIn(-2, row, f"row {b}: unwritten slot")
            self.assertEqual(len(picked), len(set(picked)), f"row {b}: duplicates")
            self.assertEqual(set(picked), want, f"row {b}")

    @torch.inference_mode()
    def test_paged_matches_reference(self):
        """Decode pool slots through the index page table (full top-k layers and
        the candidate source's own top-k), bf16 and fp32 scores in padded rows."""
        for dtype, width, page_size in (
            (torch.bfloat16, 8192, 64),
            (torch.float32, 8192, 64),
            (torch.float32, 4096, 1),
        ):
            lens = torch.tensor(
                [0, 1, K - 1, K, K + 1, width], dtype=torch.int32, device=DEV
            )
            page_table = random_page_table(6, width // page_size, seed=width)
            metadata = SimpleNamespace(
                compressed_seq_lens=lens,
                page_table=page_table,
                compressed_page_size=page_size,
                use_topk_v2=True,
                topk_metadata=plan_topk_v2(lens),
            )
            scores = distinct_scores(lens, width, dtype, seed=3, stride=width + 64)
            want = through_table(
                reference_columns(scores, lens, K), page_table, page_size
            )
            for variant in [SGL_KERNEL] + NON_DEFAULT:
                with self.subTest(dtype=dtype, page_size=page_size, variant=variant):
                    with make_selector(*variant) as selector:
                        out = torch.full((6, K), -2, dtype=torch.int32, device=DEV)
                        selector.paged_from_metadata(scores, metadata, out)
                    self.assert_picks(out, want)

    @torch.inference_mode()
    def test_sparse_matches_reference(self):
        """A consumer's picks among its published blocks as positions and as pool
        slots, from one selection; full 16384-column rows and short ones."""
        for topk_blocks in (2048, 96):
            width = topk_blocks * BLOCK
            valid_lens = torch.tensor(
                [0, 1, 300, K, width - 3, width], dtype=torch.int32, device=DEV
            )
            logical, physical = sparse_tables(valid_lens, topk_blocks, seed=topk_blocks)
            scores = distinct_scores(valid_lens, width, torch.bfloat16, seed=5)
            columns = reference_columns(scores, valid_lens, K)
            want_positions = through_table(columns, logical, BLOCK)
            want_slots = through_table(columns, physical, BLOCK)
            for variant in [SGL_KERNEL] + NON_DEFAULT:
                with self.subTest(topk_blocks=topk_blocks, variant=variant):
                    positions = torch.full((6, K), -2, dtype=torch.int32, device=DEV)
                    slots = torch.full_like(positions, -2)
                    with make_selector(*variant) as selector:
                        selector.sparse(
                            scores,
                            valid_lens,
                            [(logical, positions), (physical, slots)],
                        )
                    self.assert_picks(positions, want_positions)
                    self.assert_picks(slots, want_slots)

    @torch.inference_mode()
    def test_ragged_matches_reference(self):
        """Prefill picks plus per-row offsets from padded score rows."""
        width = 4096
        lens = torch.tensor(
            [0, 1, K - 1, K, K + 1, width], dtype=torch.int32, device=DEV
        )
        offsets = (torch.arange(6, device=DEV, dtype=torch.int32) * 10007) % 65536
        for dtype in (torch.bfloat16, torch.float32):
            reference = distinct_scores(lens, width, dtype, seed=7, stride=width + 64)
            want = [
                {c + int(o) for c in row}
                for row, o in zip(reference_columns(reference, lens, K), offsets)
            ]
            for variant in [SGL_KERNEL] + NON_DEFAULT:
                with self.subTest(dtype=dtype, variant=variant):
                    # top-k v2 may mask columns in place, so each run gets its own rows
                    scores = distinct_scores(
                        lens, width, dtype, seed=7, stride=width + 64
                    )
                    out = torch.full((6, K), -2, dtype=torch.int32, device=DEV)
                    with make_selector(*variant) as selector:
                        selector.ragged(scores, lens, out=out, offsets=offsets)
                    self.assert_picks(out, want)

    @torch.inference_mode()
    def test_amax_blocks_match_reference(self):
        """The candidate source's top 2048 blocks by block maximum, the newest block
        always kept, for rows below, at and above 2048 blocks."""
        topk_blocks, width = 2048, 4096 * BLOCK
        lens = torch.tensor(
            [1, 9, BLOCK * topk_blocks, BLOCK * topk_blocks + 1, width - 5, width],
            dtype=torch.int32,
            device=DEV,
        )
        scores = block_scores_with_distinct_maxima(lens, width, seed=11)
        want = reference_blocks(scores, lens, topk_blocks)
        nblocks = (lens + BLOCK - 1) // BLOCK
        for variant in [SGL_KERNEL] + NON_DEFAULT:
            with self.subTest(variant=variant):
                with make_selector(*variant) as selector:
                    blocks = selector.amax_blocks(scores, lens, nblocks, topk_blocks)
                self.assert_picks(blocks, want)

    @torch.inference_mode()
    def test_dense_matches_reference(self):
        """The torch-scored paths' plain top-k, as (values, indices)."""
        width = 3000
        lens = torch.full((4,), width, dtype=torch.int32, device=DEV)
        for dtype in (torch.bfloat16, torch.float32):
            scores = distinct_scores(lens, width, dtype, seed=13)
            want = reference_columns(scores, lens, K)
            for variant in [SGL_KERNEL] + NON_DEFAULT:
                with self.subTest(dtype=dtype, variant=variant):
                    with make_selector(*variant) as selector:
                        values, indices = selector.dense(scores, K)
                    self.assertEqual([set(r) for r in indices.tolist()], want)
                    torch.testing.assert_close(values, scores.gather(1, indices))

    @torch.inference_mode()
    def test_unfused_selections_split_rows_into_chunks(self):
        """Capped at four rows of scratch, each unfused selection runs two chunks, the
        second short, and still picks the reference: pool slots, positions plus
        offsets, and both sparse outputs of one selection."""
        rows, width, page_size = 6, 4096, 64
        lens = torch.tensor(
            [0, 1, K - 1, K, K + 1, width], dtype=torch.int32, device=DEV
        )
        scores = distinct_scores(lens, width, torch.float32, seed=43, stride=width + 64)
        columns = reference_columns(scores, lens, K)
        page_table = random_page_table(rows, width // page_size, seed=47)
        metadata = SimpleNamespace(
            compressed_seq_lens=lens,
            page_table=page_table,
            compressed_page_size=page_size,
            use_topk_v2=False,
            topk_metadata=None,
        )
        offsets = torch.arange(rows, dtype=torch.int32, device=DEV) * 4099
        logical, physical = sparse_tables(lens, width // BLOCK, seed=53)
        topk_func = DSATopKBackend.topk_func
        for variant in [
            ("torch", True, None, False),
            ("flashinfer", False, None, False),
            ("flashinfer", False, "small", True),
        ]:
            with (
                self.subTest(variant=variant),
                make_selector(*variant) as selector,
                patch.object(selector_module, "_UNFUSED_CHUNK_BYTES", 4 * width * 4),
                patch.object(
                    DSATopKBackend, "topk_func", autospec=True, side_effect=topk_func
                ) as spy,
            ):
                out = torch.full((rows, K), -2, dtype=torch.int32, device=DEV)
                selector.paged_from_metadata(scores, metadata, out)
                self.assert_picks(out, through_table(columns, page_table, page_size))

                selector.ragged(scores, lens, out=out, offsets=offsets)
                self.assert_picks(
                    out,
                    [{c + int(o) for c in row} for row, o in zip(columns, offsets)],
                )

                positions = torch.full_like(out, -2)
                selector.sparse(scores, lens, [(logical, positions), (physical, out)])
                self.assert_picks(positions, through_table(columns, logical, BLOCK))
                self.assert_picks(out, through_table(columns, physical, BLOCK))
                self.assertEqual(
                    [call.args[1].shape[0] for call in spy.call_args_list], [4, 2] * 3
                )

    @torch.inference_mode()
    def test_flashinfer_ties_follow_logical_positions(self):
        """Among equal scores FlashInfer keeps the smallest (or largest) logical
        positions on every V4.1 selection: positions, pool slots through a table,
        candidate columns in ascending block order, block ids, and dense-block
        candidates whose published order is unsorted."""
        rows, valid, width = 2, 1000, 2048
        lens = torch.full((rows,), valid, dtype=torch.int32, device=DEV)
        ties = torch.zeros(rows, width, dtype=torch.bfloat16, device=DEV)
        page_table = random_page_table(rows, width // 64, seed=17)
        metadata = SimpleNamespace(
            compressed_seq_lens=lens,
            page_table=page_table,
            compressed_page_size=64,
            use_topk_v2=False,
            topk_metadata=None,
        )
        logical, physical = sparse_tables(lens, width // BLOCK, seed=19)
        # Unsorted dense-block candidates: block 5's columns come before block 1's.
        published = torch.tensor([[5, 1, 3, -1]] * rows, dtype=torch.int32, device=DEV)
        candidate_lens = torch.full((rows,), 48, dtype=torch.int32, device=DEV)
        for variant in FLASHINFER_TIES:
            small = variant[2] == "small"
            first = list(range(K)) if small else list(range(valid - K, valid))
            with self.subTest(variant=variant), make_selector(*variant) as selector:
                out = torch.full((rows, K), -2, dtype=torch.int32, device=DEV)
                selector.ragged(ties, lens, out=out, offsets=torch.zeros_like(lens))
                self.assert_picks(out, [set(first)] * rows)

                selector.paged_from_metadata(ties, metadata, out)
                self.assert_picks(
                    out, through_table([set(first)] * rows, page_table, 64)
                )

                positions = torch.full((rows, K), -2, dtype=torch.int32, device=DEV)
                selector.sparse(ties, lens, [(logical, positions), (physical, out)])
                self.assert_picks(
                    positions, through_table([set(first)] * rows, logical, BLOCK)
                )
                self.assert_picks(
                    out, through_table([set(first)] * rows, physical, BLOCK)
                )

                # 125 blocks, the newest (124) forced in, 7 more among 124 tied ones.
                blocks = selector.amax_blocks(ties, lens, (lens + 7) // 8, 8)
                rest = range(7) if small else range(117, 124)
                self.assert_picks(blocks, [{124, *rest}] * rows)

                ids = select_candidate_block_ids(
                    ties[:, :valid].float(),
                    lens[:, None],
                    8,
                    BLOCK,
                    topk_op=selector.dense,
                )
                self.assert_picks(ids, [{124, *rest}] * rows)

                picks = topk_among_blocks(
                    ties[:, :48].float(),
                    candidate_lens,
                    selector.order_candidates(published),
                    12,
                    block_size=BLOCK,
                    topk_op=selector.dense,
                )
                want = (
                    {*range(8, 16), *range(24, 28)}
                    if small
                    else {
                        *range(28, 32),
                        *range(40, 48),
                    }
                )
                self.assert_picks(picks.to(torch.int32), [want] * rows)

    def test_flashinfer_paths_replay_under_cuda_graphs(self):
        """A captured decode step (full top-k, consumer sparse top-k, source block
        top-k) replays with fresh scores and lengths and rewrites every output slot,
        fused and unfused, under the default modes, the small tie-break, and the small
        tie-break with determinism. Off deterministic mode CUB serves the 128K-column
        full top-k and every unfused call and native FilteredTopK the sparse top-k, so
        both graph-safe backends replay."""
        rows, width, block_width, topk_blocks = 4, 131072, 8192, 128
        sparse_width = topk_blocks * BLOCK
        lens = torch.zeros(rows, dtype=torch.int32, device=DEV)
        valid_lens = torch.zeros_like(lens)
        page_table = random_page_table(rows, width // 64, seed=23)
        metadata = SimpleNamespace(
            compressed_seq_lens=lens,
            page_table=page_table,
            compressed_page_size=64,
            use_topk_v2=False,
            topk_metadata=None,
        )
        scores = torch.zeros(rows, width + 64, dtype=torch.bfloat16, device=DEV)[
            :, :width
        ]
        sparse_scores = torch.zeros(
            rows, sparse_width, dtype=torch.bfloat16, device=DEV
        )
        logical = torch.zeros(rows, topk_blocks, dtype=torch.int32, device=DEV)
        physical = torch.zeros_like(logical)
        block_lens = torch.zeros_like(lens)
        block_input = torch.zeros(rows, block_width, dtype=torch.bfloat16, device=DEV)
        outputs = [
            torch.zeros(rows, K, dtype=torch.int32, device=DEV) for _ in range(3)
        ]
        out_blocks = torch.zeros(rows, 64, dtype=torch.int32, device=DEV)

        def load(step):
            lens.copy_(torch.tensor([1, 404 - step, 5000, width - step], device=DEV))
            valid_lens.copy_(torch.tensor([0, 9, 700, sparse_width - step], device=DEV))
            block_lens.copy_(
                torch.tensor([8, 300, 4000 + step, block_width], device=DEV)
            )
            scores.copy_(peaked_scores(lens, width, seed=29 + step))
            sparse_scores.copy_(
                distinct_scores(
                    valid_lens, sparse_width, torch.bfloat16, seed=31 + step
                )
            )
            tables = sparse_tables(valid_lens, topk_blocks, seed=37 + step)
            logical.copy_(tables[0])
            physical.copy_(tables[1])
            block_input.copy_(
                block_scores_with_distinct_maxima(
                    block_lens, block_width, seed=41 + step
                )
            )

        def expected():
            columns = reference_columns(sparse_scores, valid_lens, K)
            return [
                through_table(reference_columns(scores, lens, K), page_table, 64),
                through_table(columns, logical, BLOCK),
                through_table(columns, physical, BLOCK),
                reference_blocks(block_input, block_lens, 64),
            ]

        modes = [(None, False), ("small", False), ("small", True)]
        for fused, (tie_break, deterministic) in itertools.product(
            (True, False), modes
        ):
            with self.subTest(
                fused=fused, tie_break=tie_break, deterministic=deterministic
            ):
                with make_selector(
                    "flashinfer", fused, tie_break, deterministic
                ) as selector:

                    def run():
                        selector.paged_from_metadata(scores, metadata, outputs[0])
                        selector.sparse(
                            sparse_scores,
                            valid_lens,
                            [(logical, outputs[1]), (physical, outputs[2])],
                        )
                        out_blocks.copy_(
                            selector.amax_blocks(
                                block_input,
                                block_lens,
                                (block_lens + 7) // 8,
                                out_blocks.shape[1],
                            )
                        )

                    load(0)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    for step in (1, 2, 3):
                        load(step)
                        for out in (*outputs, out_blocks):
                            out.fill_(-2)
                        graph.replay()
                        torch.cuda.synchronize()
                        for got, want in zip([*outputs, out_blocks], expected()):
                            self.assert_picks(got, want)


if __name__ == "__main__":
    unittest.main()
