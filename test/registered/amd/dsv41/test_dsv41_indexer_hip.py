"""HIP candidate selection ignores unreachable logits and bounds prefill allocations."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=50, suite="stage-b-test-1-gpu-small-amd-mi35x")

INDEX_PAGE_SIZE = 64
TOPK = 512
# Released V4.1 config; the span 2048 x 8 = 16384 is where level one starts to bind.
TOPK_BLOCKS, BLOCK_SIZE = 2048, 8







def index_slots(page_table, pos):
    """Slot of compressed position `pos` through the expanded indexer page table."""
    return (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )


def reference_position_mask(logits, lens, topk_blocks, block_size):
    """The reference's level one on logits whose tail past the reach is -inf."""
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    col = torch.arange(logits.shape[1], device=logits.device)
    pre = logits.masked_fill(col >= lens[:, None], -torch.inf)
    return select_candidate_blocks(
        pre, lens[:, None], topk_blocks=topk_blocks, block_size=block_size
    )


def candidate_block_ids_to_mask(ids, num_blocks):
    """bool [rows, num_blocks] block mask of `CandidateBlocks.ids`."""
    rows = ids.shape[0]
    # column num_blocks is the sink for the -1 padding; the mask is the view before it
    keep = torch.zeros((rows, num_blocks + 1), dtype=torch.bool, device=ids.device)
    keep.scatter_(1, ids.masked_fill(ids < 0, num_blocks).to(torch.int64), True)
    return keep[:, :num_blocks]


def ids_to_position_mask(ids, block_size, width):
    num_blocks = (width + block_size - 1) // block_size
    keep = candidate_block_ids_to_mask(ids, num_blocks)
    return keep.repeat_interleave(block_size, dim=-1)[:, :width]


def reference_consumer_rows(logits, lens, pos_mask, topk):
    """Per row: the set the reference consumer selects among the reachable candidates
    and its valid count; tie-free logits make it exact."""
    col = torch.arange(logits.shape[1], device=logits.device)
    out = []
    for b in range(logits.shape[0]):
        n = int(lens[b])
        cand = pos_mask[b] & (col < n)
        n_cand = int(cand.sum())
        k = min(topk, n_cand)
        s = logits[b].masked_fill(~cand, -torch.inf)
        out.append((set(s.topk(k).indices.tolist()), k))
    return out






@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "FlyDSL fp4 indexer kernels are gfx950 only"
)
class TestTwoLevelDecodeHip(CustomTestCase):
    """Level one of the two-level top-k: the source keeps TOPK_BLOCKS x BLOCK_SIZE positions, later ratio-1 sources select inside them."""

    def _garbage_tail(self, logits, lens):
        """Kernel garbage past each row's reach (large positives on even rows, NaN on
        odd), so a helper reading the tail fails loudly."""
        col = torch.arange(logits.shape[1], device=logits.device)
        tail = col[None, :] >= lens[:, None]
        odd = (torch.arange(logits.shape[0], device=logits.device) % 2 == 1)[:, None]
        logits = logits.masked_fill(tail & ~odd, 1e4)
        return logits.masked_fill(tail & odd, torch.nan)

    def _assert_consumer_matches_reference(self, logits, seq, cands, page_table, msg):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            topk_within_candidate_blocks_hip,
        )

        rows, width = logits.shape
        page_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        raw_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        topk_within_candidate_blocks_hip(
            logits,
            seq,
            cands,
            page_table=page_table,
            page_size=INDEX_PAGE_SIZE,
            page_indices=page_indices,
            raw_indices=raw_indices,
        )
        pos_mask = ids_to_position_mask(cands.ids, cands.block_size, width)
        for b, (want, k) in enumerate(
            reference_consumer_rows(logits, seq, pos_mask, TOPK)
        ):
            got = raw_indices[b]
            self.assertTrue(bool((got[:k] >= 0).all()), f"{msg}: prefix row {b}")
            self.assertTrue(bool((got[k:] == -1).all()), f"{msg}: padding row {b}")
            self.assertEqual(set(got[:k].tolist()), want, f"{msg}: selection row {b}")
            sel = got[:k].to(torch.int64)
            expect = index_slots(page_table[b : b + 1], sel[None])[0]
            self.assertTrue(
                torch.equal(page_indices[b, :k].to(torch.int64), expect),
                f"{msg}: slots row {b}",
            )
            self.assertTrue(bool((page_indices[b, k:] == -1).all()))

    def test_level_one_matches_reference_under_garbage_tail(self):
        """The HIP block top-k (AOT row-split and torch fallback) must publish the
        reference's blocks and never read past a row's reach."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            select_candidate_blocks_hip,
        )

        torch.manual_seed(11)
        cases = (
            # Released blocks, a rectangle just wider than the longest row.
            (TOPK_BLOCKS, BLOCK_SIZE, 40000, [3, 8, 16384, 16385, 20000, 40000, 1]),
            # Released blocks on a 1M-wide rectangle, the page table's capacity on
            # a 1M-context server: the block top-k takes the AOT row-split path.
            (TOPK_BLOCKS, BLOCK_SIZE, 1 << 20, [16385, 131072, 7, 600]),
        )
        for topk_blocks, block_size, width, lens in cases:
            with self.subTest(topk_blocks=topk_blocks, width=width, lens=lens):
                seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
                raw = torch.randn(len(lens), width, device="cuda")
                raw = self._garbage_tail(raw, seq)
                expected = reference_position_mask(raw, seq, topk_blocks, block_size)

                cands = select_candidate_blocks_hip(
                    raw, seq, topk_blocks=topk_blocks, block_size=block_size
                )
                ids = cands.ids
                self.assertEqual(ids.shape, (len(lens), topk_blocks))
                self.assertTrue(
                    torch.equal(
                        cands.compact_lens.cpu(),
                        torch.tensor(
                            [
                                min((n + block_size - 1) // block_size, topk_blocks)
                                * block_size
                                for n in lens
                            ],
                            dtype=torch.int32,
                        ),
                    )
                )
                self.assertEqual(ids.dtype, torch.int32)
                got = ids_to_position_mask(ids, block_size, width)
                self.assertTrue(torch.equal(got, expected), "published blocks")
                for b, n in enumerate(lens):
                    row = ids[b]
                    n_ids = int((row >= 0).sum())
                    self.assertTrue(bool((row[:n_ids] >= 0).all()), "padding last")
                    self.assertLessEqual(
                        int(got[b, :n].sum()), topk_blocks * block_size
                    )

                n_pages = (width + INDEX_PAGE_SIZE - 1) // INDEX_PAGE_SIZE
                page_table = torch.stack(
                    [torch.randperm(n_pages, device="cuda") for _ in lens]
                ).to(torch.int32)
                self._assert_consumer_matches_reference(
                    raw, seq, cands, page_table, "consumer"
                )












@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "V4.1 low-ratio HIP caller is gfx950 only"
)
class TestOversizedPrefillRequestChunking(CustomTestCase):
    """The V4.1 low-ratio caller must enforce the logits budget within one request."""

    @staticmethod
    def _candidate_blocks(row_ids):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            CandidateBlocks,
        )

        ids = torch.as_tensor(row_ids, dtype=torch.int32).reshape(-1, 1)
        rows = ids.shape[0]
        return CandidateBlocks(
            ids=ids,
            compact_lens=torch.ones(rows, dtype=torch.int32),
            compact_page_table=torch.zeros((rows, 1), dtype=torch.int32),
            compact_page_size=1,
            block_size=1,
        )

    def _run(self, rows, rows_per_chunk, *, publish=False, consume=False):
        from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip as hip

        page_indices = torch.full((rows, 1), -1, dtype=torch.int32)
        raw_indices = torch.full((rows, 1), -1, dtype=torch.int32)
        core = SimpleNamespace(
            sparse_page_indices=lambda _ratio: page_indices,
            sparse_raw_indices=lambda _ratio: raw_indices,
        )
        indexer_metadata = SimpleNamespace(
            page_table=torch.zeros((rows, 1), dtype=torch.int32),
            compressed_seq_lens=torch.arange(1, rows + 1, dtype=torch.int32),
            compressed_page_size=INDEX_PAGE_SIZE,
        )
        metadata = SimpleNamespace(
            core_metadata=core,
            late_layer_tail=None,
            fp4_low_ratio_prefill_workspaces={1: None},
            low_ratio_indexer_metadata=lambda _ratio: indexer_metadata,
        )
        pool = SimpleNamespace(
            get_index_k_fp4_payload_buffer=lambda _layer_id: torch.empty(0),
            get_index_k_fp4_scale_buffer=lambda _layer_id: torch.empty(0),
        )
        candidates = self._candidate_blocks(range(rows)) if consume else None
        backend = SimpleNamespace(
            token_to_kv_pool=pool,
            forward_metadata=metadata,
            low_ratio_identity_skip=False,
            candidate_masks=[candidates] if consume else None,
        )
        indexer = SimpleNamespace(
            index_topk=1,
            is_candidate_source=publish,
            uses_candidates=consume,
            candidate_topk_blocks=1,
            candidate_block_size=1,
        )
        layer = SimpleNamespace(compress_ratio=1, indexer=indexer, layer_id=0)
        forward_batch = SimpleNamespace(
            seq_lens_cpu=[max(rows, 1)],
            extend_seq_lens_cpu=[rows],
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
        )
        pos = torch.arange(rows, dtype=torch.int64)
        score_sizes = []
        consumed_ids = []

        def score(**kwargs):
            chunk_rows = kwargs["q_fp4"].shape[0]
            score_sizes.append(chunk_rows)
            return torch.zeros((chunk_rows, 1))

        def select(**kwargs):
            if kwargs["consume"] is not None:
                consumed_ids.append(kwargs["consume"][0].ids.flatten().tolist())
            if kwargs["publish"] is not None:
                row_ids = (kwargs["compress_lens"] - 1).tolist()
                kwargs["publish"].append(self._candidate_blocks(row_ids))

        indexer_inputs = (
            torch.zeros((rows, 1, 1), dtype=torch.uint8),
            torch.zeros((rows, 1), dtype=torch.uint8),
            torch.zeros((rows, 1), dtype=torch.bfloat16),
        )
        with (
            mock.patch.object(
                hip, "logits_rows_per_chunk", new=lambda *_args: rows_per_chunk
            ),
            mock.patch.object(
                hip, "_indexer_inputs", new=lambda *_args: indexer_inputs
            ),
            mock.patch.object(hip, "aiter_fp4_paged_mqa_logits", new=score),
            mock.patch.object(hip, "_select_topk_extend_hip", new=select),
        ):
            hip.low_ratio_index_topk_hip_extend(
                backend,
                layer,
                torch.empty((rows, 1)),
                torch.empty((rows, 1)),
                pos,
                forward_batch,
            )
        return SimpleNamespace(
            score_sizes=score_sizes,
            consumed_ids=consumed_ids,
            published=backend.candidate_masks,
        )

    def test_measured_oor_shape_is_eight_budgeted_score_calls(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            LOW_RATIO_PAGE_TABLE_BUCKET,
            logits_rows_per_chunk,
        )

        score_width = 237568
        page_table = torch.empty((1, 3712), dtype=torch.int32, device="meta")
        rows_per_chunk = logits_rows_per_chunk(page_table, LOW_RATIO_PAGE_TABLE_BUCKET)
        self.assertEqual(rows_per_chunk, 2259)

        result = self._run(16260, rows_per_chunk)

        self.assertEqual(result.score_sizes, [2259] * 7 + [447])
        self.assertTrue(all(size <= rows_per_chunk for size in result.score_sizes))
        self.assertLessEqual(max(result.score_sizes) * score_width * 4, 2 * 1024**3)

    def test_candidate_source_publishes_one_request_in_row_order(self):
        result = self._run(5, 2, publish=True)

        self.assertEqual(result.score_sizes, [2, 2, 1])
        self.assertEqual(len(result.published), 1)
        self.assertEqual(result.published[0].ids.flatten().tolist(), list(range(5)))

    def test_candidate_consumer_slices_the_request_by_local_row_offset(self):
        result = self._run(5, 2, consume=True)

        self.assertEqual(result.score_sizes, [2, 2, 1])
        self.assertEqual(result.consumed_ids, [[0, 1], [2, 3], [4]])



if __name__ == "__main__":
    unittest.main()
