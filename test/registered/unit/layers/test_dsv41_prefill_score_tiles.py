"""CPU checks for prefill tile budgets and selection equivalence."""

import contextlib
import sys
import types
import unittest
import weakref
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.dsv4 as dsv4_ops
from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.index_logits import (
    FLAT_LOGITS_WIDTH_ALIGN,
    flat_index_logits_rows_per_tile,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import (
    dense_blocks,
    scoring,
    sparse_table,
)
from sglang.srt.layers.attention.mqa_logits_utils import mqa_logits_row_bytes
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HEADS = 32
ROW_ALIGN = 128 // HEADS
TOPK = 16
TOPK_BLOCKS = 4
BLOCK = 8
GARBAGE = 1e30


def ceil_align(x: int, a: int) -> int:
    return -(-x // a) * a


def exact_scores(q, weights, k):
    """``sum_h weights[i, h] * relu(q[i, h] . k[j])`` in exact integer arithmetic,
    so a score never depends on which rows are scored together."""
    dots = torch.einsum("ihd,jd->ihj", q.long(), k.long()).clamp_min(0)
    return torch.einsum("ih,ihj->ij", weights.long(), dots).float()


class FakeDeepGEMM:
    """``fp8_fp4_mqa_logits`` on CPU: DeepGEMM's allocation (rows aligned to the
    row group, stride to 256 fp32), the row's window scored, garbage past it."""

    def __init__(self):
        self.allocations = []  # (rows, bytes)
        self.max_live_tiles = 0
        self._live = []

    def fp8_fp4_mqa_logits(
        self, q, kv, weights, ks, ke, clean_logits, max_k, logits_dtype=torch.float32
    ):
        assert not clean_logits and logits_dtype == torch.float32
        rows = q[0].shape[0]
        stride = ceil_align(ceil_align(max_k, 128), 256)
        buffer = torch.full((ceil_align(rows, ROW_ALIGN), stride), GARBAGE)
        self._live = [r for r in self._live if r() is not None]
        self.max_live_tiles = max(self.max_live_tiles, len(self._live) + 1)
        self._live.append(weakref.ref(buffer))
        self.allocations.append((rows, buffer.numel() * buffer.element_size()))
        scores = exact_scores(q[0], weights, kv[0])
        for i in range(rows):
            start, end = int(ks[i]), int(ke[i])
            buffer[i, : end - start] = scores[i, start:end]
        return buffer[:rows, :max_k]

    def module(self):
        return types.SimpleNamespace(fp8_fp4_mqa_logits=self.fp8_fp4_mqa_logits)


def fake_topk_transform_ragged(
    scores, seq_lens, *, out_offsets, out_indices, row_starts=None
):
    """The ragged top-k: each row's top ``k`` of ``scores[i, :seq_lens[i]]`` in
    ascending column order plus its offset, ``-1`` padded."""
    assert row_starts is None
    k = out_indices.shape[1]
    out_indices.fill_(-1)
    for i in range(scores.shape[0]):
        n = int(seq_lens[i])
        if n == 0:
            continue
        picked = scores[i, :n].topk(min(k, n)).indices.sort().values
        out_indices[i, : picked.numel()] = (picked + out_offsets[i]).to(
            out_indices.dtype
        )


def make_case(request_lengths, ratio, seed=0):
    """``request_lengths`` is ``[(query rows, compressed context)]``; the rows of
    a request are its newest positions, as an extend chunk's are."""
    gen = torch.Generator().manual_seed(seed)
    rows = sum(q for q, _ in request_lengths)
    columns = sum(n for _, n in request_lengths)
    q = torch.randint(-3, 4, (rows, HEADS, 64), generator=gen, dtype=torch.int8)
    k = torch.randint(-3, 4, (columns, 64), generator=gen, dtype=torch.int8)
    weights = torch.randint(1, 6, (rows, HEADS), generator=gen).float()
    starts, lengths, start = [], [], 0
    for queries, context in request_lengths:
        assert queries <= context * ratio or context == 0
        starts += [start] * queries
        lengths += [
            (p + 1) // ratio for p in range(context * ratio - queries, context * ratio)
        ]
        start += context
    data = scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(columns),
        request_starts=torch.tensor(starts, dtype=torch.int32),
        lens_per_request=[n for _, n in request_lengths],
        rows_per_request=[q for q, _ in request_lengths],
        compress_lens=torch.tensor(lengths, dtype=torch.int32),
        q_fp4=q,
        q_sf=torch.zeros((rows, HEADS), dtype=torch.int32),
        weights=weights,
    )
    kv = (k, torch.zeros(columns, dtype=torch.int32))
    width = max([n for _, n in request_lengths], default=0)
    clean = torch.full((rows, width), -torch.inf)
    full = exact_scores(q, weights, k)
    for i in range(rows):
        s, n = starts[i], lengths[i]
        clean[i, :n] = full[i, s : s + n]
    return data, kv, clean


def request_rows(data):
    row = 0
    for queries, context in zip(data.rows_per_request, data.lens_per_request):
        yield slice(row, row + queries), context
        row += queries


def reference_publish(data, clean):
    """Untiled reference with pad-based block selection."""
    rows = data.num_rows
    width = min(TOPK_BLOCKS, -(-max(data.lens_per_request, default=0) // BLOCK))
    blocks = torch.full((rows, width), -1, dtype=torch.int32)
    for sl, context in request_rows(data):
        if sl.stop > sl.start and context:
            ids = select_candidate_block_ids(
                logits=clean[sl, :context],
                compress_lens=data.compress_lens[sl, None],
                topk_blocks=TOPK_BLOCKS,
                block_size=BLOCK,
            )
            blocks[sl, : ids.shape[1]] = ids
    selected = torch.full((rows, TOPK), -1, dtype=torch.int32)
    if clean.shape[1]:
        fake_topk_transform_ragged(
            clean,
            data.compress_lens,
            out_offsets=data.request_starts,
            out_indices=selected,
        )
    return selected, blocks


def reference_consume(data, clean, blocks):
    selected = torch.full((data.num_rows, TOPK), -1, dtype=torch.int32)
    for sl, context in request_rows(data):
        if sl.stop > sl.start and context:
            padded = torch.nn.functional.pad(
                clean[sl, :context], (0, -context % BLOCK), value=-torch.inf
            )
            picks = topk_among_blocks(
                padded, data.compress_lens[sl], blocks[sl], TOPK, block_size=BLOCK
            )
            selected[sl] = torch.where(
                picks >= 0, picks + data.request_starts[sl, None], -1
            ).to(torch.int32)
    return selected


CASES = [
    # unaligned contexts, empty requests and requests split across tiles
    [(0, 0), (1, 1), (5, 17), (13, 64), (0, 9), (23, 301)],
    [(7, 7), (2, 255), (33, 129)],
    [(1, 0), (1, 1), (0, 7)],
]


class TestRowsPerTile(CustomTestCase):
    def test_tile_and_scratch_stay_within_the_budget(self):
        for width in (1, 255, 256, 4097, 65535, 1 << 20, 485_123):
            for rows in (1, 3, 4, 509, 8192):
                for scratch, fixed in ((0, 0), (width, 8 * width), (200_000, 0)):
                    for budget in (1 << 20, 64 << 20, 2 << 30):
                        n = flat_index_logits_rows_per_tile(
                            rows,
                            width,
                            heads=HEADS,
                            budget_bytes=budget,
                            scratch_row_bytes=scratch,
                            scratch_tile_bytes=fixed,
                        )
                        row_bytes = mqa_logits_row_bytes(width) + scratch
                        if ceil_align(rows, ROW_ALIGN) * row_bytes + fixed <= budget:
                            self.assertEqual(n, rows)
                            continue
                        self.assertEqual(n % ROW_ALIGN, 0)
                        self.assertGreaterEqual(n, ROW_ALIGN)
                        self.assertGreater((n + ROW_ALIGN) * row_bytes + fixed, budget)
                        if n > ROW_ALIGN:
                            self.assertLess(n, rows)
                            self.assertLessEqual(n * row_bytes + fixed, budget)


class TestScoreTiles(CustomTestCase):
    def test_tiles_cover_rows_once_with_identical_logits(self):
        data, kv, clean = make_case(CASES[0], ratio=1)
        fake = FakeDeepGEMM()
        width = ceil_align(max(data.lens_per_request), FLAT_LOGITS_WIDTH_ALIGN)
        row_bytes = mqa_logits_row_bytes(width)
        with patch.dict(sys.modules, {"deep_gemm": fake.module()}):
            for groups in (1, 2, 3, 100):
                with patch.object(
                    scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", groups * 4 * row_bytes
                ):
                    covered = []
                    for tile, logits in scoring.score_tiles(data, kv):
                        self.assertEqual(tile.start % ROW_ALIGN, 0)
                        covered.extend(range(tile.start, tile.stop))
                        n = clean.shape[1]
                        lens = data.compress_lens[tile]
                        past = torch.arange(n)[None, :] >= lens[:, None]
                        self.assertTrue(
                            torch.equal(
                                logits[:, :n].masked_fill(past, 0),
                                clean[tile].masked_fill(past, 0),
                            )
                        )
                        del logits
                    self.assertEqual(covered, list(range(data.num_rows)))


class TestSelectionIgnoresTiling(CustomTestCase):
    def run_all(self, data, kv):
        selected, blocks = dense_blocks._publish_prefill_blocks(
            data=data, kv=kv, topk=TOPK, topk_blocks=TOPK_BLOCKS, block_size=BLOCK
        )
        consumed = dense_blocks._consume_prefill_blocks(
            data=data, kv=kv, topk=TOPK, blocks=blocks, block_size=BLOCK
        )
        local = torch.full((data.num_rows, TOPK), -1, dtype=torch.int32)
        scoring.dense_prefill_topk(data, kv, out=local)
        return selected, blocks, consumed, local

    def test_publish_consume_and_full_topk(self):
        for case in CASES:
            for ratio in (1, 2):
                data, kv, clean = make_case(case, ratio=ratio, seed=ratio)
                ref_selected, ref_blocks = reference_publish(data, clean)
                ref_consumed = reference_consume(data, clean, ref_blocks)
                ref_local = torch.where(
                    ref_selected >= 0,
                    ref_selected - data.request_starts[:, None],
                    -1,
                )
                width = ceil_align(
                    max(data.lens_per_request, default=0), FLAT_LOGITS_WIDTH_ALIGN
                )
                group = ROW_ALIGN * mqa_logits_row_bytes(width)
                for budget in (1, group, 2 * group, 3 * group, 1 << 40):
                    with self.subTest(case=case, ratio=ratio, budget=budget):
                        fake = FakeDeepGEMM()
                        with (
                            patch.dict(sys.modules, {"deep_gemm": fake.module()}),
                            patch.object(
                                dsv4_ops,
                                "topk_transform_ragged_v2",
                                fake_topk_transform_ragged,
                            ),
                            patch.object(
                                scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget
                            ),
                        ):
                            selected, blocks, consumed, local = self.run_all(data, kv)
                        self.assertTrue(torch.equal(selected, ref_selected))
                        self.assertTrue(torch.equal(blocks, ref_blocks))
                        self.assertTrue(torch.equal(consumed, ref_consumed))
                        self.assertTrue(torch.equal(local, ref_local))
                        if fake.allocations:
                            self.assertEqual(fake.max_live_tiles, 1)

    def test_tiles_with_scratch_stay_within_the_budget(self):
        data, kv, _ = make_case(CASES[0], ratio=1)
        width = ceil_align(max(data.lens_per_request), FLAT_LOGITS_WIDTH_ALIGN)
        scratch, fixed = dense_blocks._publish_scratch_bytes(width, BLOCK, TOPK_BLOCKS)
        budget = 3 * ROW_ALIGN * (mqa_logits_row_bytes(width) + scratch) + fixed
        fake = FakeDeepGEMM()
        with (
            patch.dict(sys.modules, {"deep_gemm": fake.module()}),
            patch.object(
                dsv4_ops, "topk_transform_ragged_v2", fake_topk_transform_ragged
            ),
            patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget),
        ):
            dense_blocks._publish_prefill_blocks(
                data=data, kv=kv, topk=TOPK, topk_blocks=TOPK_BLOCKS, block_size=BLOCK
            )
        self.assertGreater(len(fake.allocations), 1)
        self.assertEqual(fake.allocations[0][0], 3 * ROW_ALIGN)
        for rows, nbytes in fake.allocations:
            self.assertLessEqual(nbytes + rows * scratch + fixed, budget)


class TestSparseTableTileLifetime(CustomTestCase):
    def test_publish_frees_each_tile_and_its_keys(self):
        data, kv, _ = make_case(CASES[0], ratio=1)
        width = ceil_align(max(data.lens_per_request), FLAT_LOGITS_WIDTH_ALIGN)
        fake = FakeDeepGEMM()
        keys_alive = []
        key_bytes = []
        logits_fn = fake.fp8_fp4_mqa_logits

        def checked_logits(*args, **kwargs):
            self.assertTrue(all(r() is None for r in keys_alive))
            return logits_fn(*args, **kwargs)

        fake.fp8_fp4_mqa_logits = checked_logits

        def fake_amax8(logits, lens, *, out):
            keys_alive.append(weakref.ref(out))
            key_bytes.append(out.untyped_storage().nbytes())
            return out.zero_()

        budget = (
            2
            * ROW_ALIGN
            * (mqa_logits_row_bytes(width) + sparse_table._block_keys_row_bytes(width))
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": fake.module()}),
            patch.object(sparse_table, "amax8_varlen", fake_amax8),
            patch.object(
                sparse_table, "topk_transform_ragged_v2", fake_topk_transform_ragged
            ),
            patch.object(
                dsv4_ops, "topk_transform_ragged_v2", fake_topk_transform_ragged
            ),
            patch.object(
                sparse_table,
                "candidate_row_lens",
                lambda lens, k: ((lens + BLOCK - 1) // BLOCK, lens.clone()),
            ),
            patch.object(sparse_table, "_build_prefill_table", lambda **kw: kw),
            patch.object(sparse_table, "_row_pair_ids", lambda *a, **kw: None),
            patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget),
        ):
            sparse_table.publish_prefill_table(
                data=data,
                kv=kv,
                index_page_table=torch.zeros((1, 1), dtype=torch.int32),
                index_page_size=64,
                topk_blocks=TOPK_BLOCKS,
                out_positions=torch.full((data.num_rows, TOPK), -1, dtype=torch.int32),
            )
        self.assertGreater(len(fake.allocations), 1)
        self.assertEqual(fake.max_live_tiles, 1)
        self.assertEqual(len(key_bytes), len(fake.allocations))
        for (_, nbytes), keys in zip(fake.allocations, key_bytes):
            self.assertLessEqual(nbytes + keys, budget)


CAPTURE_STATES = {
    "model_capture": lambda: patch.object(capture_mode, "is_capture_mode", True),
    "stream": lambda: patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: True
    ),
    "tc_piecewise": lambda: patch.object(
        scoring, "is_in_tc_piecewise_cuda_graph", lambda: True
    ),
}


class TestPrefillScoreBudget(CustomTestCase):
    device = torch.device("cuda", 0)

    def budget(self, *, reads=(1 << 30,), capturing=None, rows=8192, seq_len=485_000):
        """The budget with free-memory reads replaced by ``reads`` (one per query)
        and the allocator's empty_cache recorded."""
        calls = {"reads": 0, "empty_cache": 0}
        values = list(reads)

        def read(*, device_index, allow_sync):
            assert allow_sync and device_index == 0
            calls["reads"] += 1
            return values.pop(0)

        def empty_cache():
            calls["empty_cache"] += 1

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(torch.cuda, "is_current_stream_capturing", lambda: False)
            )
            if capturing:
                stack.enter_context(CAPTURE_STATES[capturing]())
            stack.enter_context(patch.object(scoring, "mqa_logits_budget_bytes", read))
            stack.enter_context(
                patch.object(
                    scoring, "mqa_logits_static_budget_bytes", lambda **_: 3 << 30
                )
            )
            stack.enter_context(patch.object(torch.cuda, "empty_cache", empty_cache))
            value = scoring.prefill_score_budget_bytes(
                num_rows=rows, max_seq_len=seq_len, device=self.device
            )
        return value, calls

    def test_ample_free_memory_takes_the_cap_without_reclaiming(self):
        value, calls = self.budget(reads=(8 << 30,))
        self.assertEqual(value, 2 << 30)
        self.assertEqual(calls, {"reads": 1, "empty_cache": 0})

    def test_low_free_memory_returns_unused_blocks_and_reads_again(self):
        """A running server's allocator holds nearly all free memory: the second
        read, after empty_cache, decides."""
        value, calls = self.budget(reads=(10 << 20, 700 << 20))
        self.assertEqual(value, 700 << 20)
        self.assertEqual(calls, {"reads": 2, "empty_cache": 1})
        value, _ = self.budget(reads=(10 << 20, 8 << 30))
        self.assertEqual(value, 2 << 30)

    def test_capture_uses_the_static_budget_without_querying(self):
        for name in CAPTURE_STATES:
            with self.subTest(capturing=name):
                value, calls = self.budget(reads=(), capturing=name)
                self.assertEqual(value, 2 << 30)
                self.assertEqual(calls, {"reads": 0, "empty_cache": 0})

    def test_breakable_replay_reads_free_memory(self):
        with enable_breakable_cuda_graph():
            value, calls = self.budget(reads=(10 << 20, 400 << 20))
        self.assertEqual(value, 400 << 20)
        self.assertEqual(calls["empty_cache"], 1)

    def test_small_forward_skips_the_query(self):
        value, calls = self.budget(reads=(), rows=16, seq_len=4096)
        self.assertEqual(value, 2 << 30)
        self.assertEqual(calls["reads"], 0)


class TestScoreTilesTakeTheForwardBudget(CustomTestCase):
    def test_tiles_are_sized_by_the_budget_for_their_rows_and_width(self):
        data, kv, _ = make_case(CASES[0], ratio=1)
        width = ceil_align(max(data.lens_per_request), FLAT_LOGITS_WIDTH_ALIGN)
        group = ROW_ALIGN * mqa_logits_row_bytes(width)
        asked = []

        def budget(**kwargs):
            asked.append(kwargs)
            return 2 * group

        fake = FakeDeepGEMM()
        with (
            patch.dict(sys.modules, {"deep_gemm": fake.module()}),
            patch.object(scoring, "prefill_score_budget_bytes", budget),
        ):
            for _ in scoring.score_tiles(data, kv):
                pass
        self.assertEqual(
            asked,
            [
                {
                    "num_rows": data.num_rows,
                    "max_seq_len": max(data.lens_per_request),
                    "device": data.q_fp4.device,
                }
            ],
        )
        self.assertEqual(fake.allocations[0][0], 2 * ROW_ALIGN)


if __name__ == "__main__":
    unittest.main()
