"""CUDA regressions for the DSA kpool pooled radix top-k transform.

The kernel (``kernels/jit/csrc/dsa/kpool_topk_transform.cuh``) selects the
top ``group_topk`` pool groups of a score row with a two-stage radix select.
Stage-1 threshold bins below a 4096-entry shared-memory capacity use a stash;
bins at or above capacity use an allocation-free exact rescan. These tests
build rows on both sides of that boundary (tight clusters, all-equal rows,
clusters that only separate at a deeper key byte), the stage-1 direct-fill
boundary, and the minimal ``length == K + 1`` row, and compare the selected
groups against ``torch.topk``.

Comparison is by value multiset: when the K-th value is tied, any subset of
the tied groups is a valid selection, so the sorted selected values must equal
the sorted reference values while the group ids must be distinct and in
range.

Scores within a row are finite by contract (the kernel's NaN ordering is
unspecified), so no row here contains NaN; ``+inf`` is used only as padding
beyond ``lengths``, where the kernel must not read.
"""

import unittest
from typing import Optional

import torch

from sglang.kernels.ops.moe.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# kSmem / (2 * sizeof(int)) in kpool_topk_transform.cuh: entries per stash round.
STASH_ENTRIES = 4096
POOL_SIZE = 4
# 512 is the instantiation used by GLM-5.3-Flash (token top-k 2048, pool 4);
# 256 is the module default.
GROUP_TOPKS = (512, 256)


def _floats_from_keys(keys: torch.Tensor) -> torch.Tensor:
    """Inverse of the kernel's monotone key for positive floats.

    The kernel maps a float ``x`` to ``bits | 0x80000000`` when ``x >= +0``,
    so a key with the top bit set corresponds to the float whose raw bits are
    the low 31 bits of the key.
    """
    assert bool(((keys >> 31) & 1).all()), "positive-float keys have bit 31 set"
    bits = (keys & 0x7FFFFFFF).to(torch.int32)
    return bits.view(torch.float32)


def _coarse_keys(values: torch.Tensor) -> torch.Tensor:
    """Mirror the kernel's high-byte ordered-FP16 key on the CPU."""
    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where(
        (bits & 0x8000) != 0,
        (~bits) & 0xFFFF,
        bits | 0x8000,
    )
    return ordered >> 8


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDsaKpoolTopkTransform(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")
        self.generator = torch.Generator(device="cpu").manual_seed(0)

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, dtype=torch.float32)

    def _tight_cluster(self, length: int, *, base: float = 1.0) -> torch.Tensor:
        # Strictly increasing (modulo FP32 rounding), with the best candidates
        # deliberately last. All values share the fp16 coarse bin, so an
        # arrival-order stash prefix is deterministically wrong.
        return base + torch.arange(length, dtype=torch.float32) * (1e-3 / length)

    def _byte_depth_clusters(self, depth: int, per_side: int) -> torch.Tensor:
        """Two clusters of ``per_side`` keys that first differ at key bit ``depth``.

        Both share key byte 24 (0xBF, float 0.5..1.0), so stage 1 puts all
        ``2 * per_side`` entries in one bin. The cluster with bit ``depth`` set
        holds every true top-K member; bits below ``depth`` are random.
        """
        base = 0xBF000000
        noise_bits = depth
        noise = torch.randint(
            0, 1 << noise_bits, (2 * per_side,), generator=self.generator
        )
        keys = torch.full((2 * per_side,), base, dtype=torch.int64) + noise
        keys[:per_side] += 1 << depth
        perm = torch.randperm(2 * per_side, generator=self.generator)
        return _floats_from_keys(keys[perm])

    def _run(
        self,
        rows: list,
        group_topk: int,
        page_table_bias: Optional[int] = None,
        topk_offset: Optional[int] = None,
        row_start: int = 0,
    ):
        batch = len(rows)
        stride = row_start + max(row.numel() for row in rows)
        # Pad with +inf so any read past ``lengths`` would corrupt the selection.
        score = torch.full((batch, stride), float("inf"), dtype=torch.float32)
        lengths = torch.empty(batch, dtype=torch.int32)
        for i, row in enumerate(rows):
            score[i, row_start : row_start + row.numel()] = row
            lengths[i] = row.numel()
        score = score.to(self.device)
        lengths = lengths.to(self.device)
        row_starts = (
            torch.full((batch,), row_start, dtype=torch.int32, device=self.device)
            if row_start
            else None
        )

        page_table = None
        topk_indices_offset = None
        if page_table_bias is not None:
            page_table = (
                torch.arange(stride * POOL_SIZE, dtype=torch.int32)
                .unsqueeze(0)
                .repeat(batch, 1)
                + page_table_bias
            ).to(self.device)
        if topk_offset is not None:
            topk_indices_offset = torch.full(
                (batch,), topk_offset, dtype=torch.int32, device=self.device
            )

        out = fast_kpool_topk_transform_fused(
            score=score,
            lengths=lengths,
            pool_size=POOL_SIZE,
            topk=group_topk * POOL_SIZE,
            page_table=page_table,
            topk_indices_offset=topk_indices_offset,
            row_starts=row_starts,
        )
        self.assertEqual(out.shape, (batch, group_topk * POOL_SIZE))
        return out.cpu()

    def _selected_groups(self, out_row: torch.Tensor, group_topk: int, length: int):
        """Decode expanded token columns back to group ids and check the layout."""
        self.assertTrue(
            bool((out_row >= 0).all()), "no pad columns expected when length > K"
        )
        tokens = out_row.view(group_topk, POOL_SIZE).to(torch.int64)
        group_ids = tokens[:, 0] // POOL_SIZE
        expected_tokens = group_ids.unsqueeze(1) * POOL_SIZE + torch.arange(POOL_SIZE)
        self.assertTrue(
            torch.equal(tokens, expected_tokens),
            "each selected group must expand to its pool_size consecutive tokens",
        )
        self.assertLess(int(group_ids.max()), length)
        self.assertEqual(
            torch.unique(group_ids).numel(), group_topk, "group ids must be distinct"
        )
        return group_ids

    def _check_row(self, out_row: torch.Tensor, row: torch.Tensor, group_topk: int):
        group_ids = self._selected_groups(out_row, group_topk, row.numel())
        selected = row[group_ids].sort().values
        reference = torch.topk(row, group_topk).values.sort().values
        self.assertTrue(
            torch.equal(selected, reference),
            "selected values must match torch.topk as a multiset",
        )

    def _check_rows(self, rows: list, group_topk: int):
        out = self._run(rows, group_topk)
        for i, row in enumerate(rows):
            with self.subTest(row=i, length=row.numel(), group_topk=group_topk):
                self._check_row(out[i], row, group_topk)

    def test_stage1_bin_exceeds_stash(self):
        # 4097 and 17802 candidates in one stage-1 bin, both beyond the 4096-entry
        # stash. Before the fix the stash kept the first 4096 by atomic arrival
        # order and the selection disagreed with torch.topk on such rows.
        for group_topk in GROUP_TOPKS:
            rows = [
                self._tight_cluster(STASH_ENTRIES + 1),
                self._tight_cluster(17802),
            ]
            self._check_rows(rows, group_topk)

    def test_oversized_bin_with_strictly_higher_values(self):
        # Exercise the output offset between values emitted above the coarse
        # threshold and values selected by the exact overflow descent.
        for group_topk in GROUP_TOPKS:
            high_count = 37
            high = 2.0 + torch.arange(high_count, dtype=torch.float32) * 1e-3
            threshold = self._tight_cluster(STASH_ENTRIES + 5311)
            low = torch.zeros(257, dtype=torch.float32)
            self._check_rows([torch.cat([high, threshold, low])], group_topk)

    def test_stride_larger_than_length(self):
        # Row length far below the score stride, as with a padded score buffer.
        for group_topk in GROUP_TOPKS:
            long_row = self._tight_cluster(17802)
            short_row = self._rand(group_topk + 1)
            out = self._run([long_row, short_row], group_topk)
            self._check_row(out[0], long_row, group_topk)
            self._check_row(out[1], short_row, group_topk)

    def test_nonzero_row_start_with_oversized_bin(self):
        for group_topk in GROUP_TOPKS:
            row = self._tight_cluster(STASH_ENTRIES + 5311)
            out = self._run([row], group_topk, row_start=3)
            self._check_row(out[0], row, group_topk)

    def test_all_equal_rows_longer_than_stash(self):
        # Every key is identical, so the final scan must clip the exact ties
        # without duplicating indices or leaving an output slot unfilled.
        for group_topk in GROUP_TOPKS:
            rows = [
                torch.full((STASH_ENTRIES + 1,), 0.75),
                torch.full((STASH_ENTRIES + 1,), 0.0),
                torch.full((STASH_ENTRIES + 1,), -0.0),
            ]
            self._check_rows(rows, group_topk)

    def test_clusters_separating_at_each_key_byte(self):
        # 4097 vs 4097 keys that first differ at key bit 16, 8 or 0: the
        # stage-1 bin overflows and the descent must walk to the byte where the
        # clusters separate before the refine rounds see a bin that fits.
        for group_topk in GROUP_TOPKS:
            rows = [
                self._byte_depth_clusters(depth, STASH_ENTRIES + 1)
                for depth in (16, 8, 0)
            ]
            self._check_rows(rows, group_topk)

    def test_stage1_exact_fill_boundary(self):
        # Exactly K distinct values above a 65536-way tie: stage 1 finds the
        # threshold bin with nothing left to refine and fills all K slots directly.
        for group_topk in GROUP_TOPKS:
            top = torch.linspace(2.0, 3.0, group_topk)
            tie = torch.full((65536,), 1.0)
            row = torch.cat([top, tie])
            row = row[torch.randperm(row.numel(), generator=self.generator)]
            self._check_rows([row], group_topk)

    def test_stash_capacity_boundary(self):
        # The exact-capacity case is part of the fallback boundary. Different
        # input orders protect against an arrival-order subset being selected;
        # the mixed-bin row also covers the output offset after values above the
        # exact-capacity threshold bin have already been emitted.
        row = self._tight_cluster(STASH_ENTRIES)
        self.assertEqual(torch.unique(_coarse_keys(row)).numel(), 1)
        negative_row = self._tight_cluster(STASH_ENTRIES, base=-0.9)
        self.assertEqual(torch.unique(_coarse_keys(negative_row)).numel(), 1)
        high = 2.0 + torch.arange(37, dtype=torch.float32) * 1e-3
        low = torch.full((257,), -2.0)
        mixed_bin_row = torch.cat([high, row, low])
        threshold_key = _coarse_keys(row)[0]
        self.assertEqual(
            int((_coarse_keys(mixed_bin_row) == threshold_key).sum()),
            STASH_ENTRIES,
        )
        rows = [
            row,
            row.flip(0),
            row[torch.randperm(row.numel(), generator=self.generator)],
            negative_row,
            mixed_bin_row,
        ]
        below_capacity = self._tight_cluster(STASH_ENTRIES - 1)
        self.assertEqual(torch.unique(_coarse_keys(below_capacity)).numel(), 1)
        for group_topk in GROUP_TOPKS:
            self._check_rows(rows, group_topk)
            self._check_rows([below_capacity], group_topk)

    def test_minimal_radix_length(self):
        # length == K + 1 is the shortest row that takes the radix path.
        for group_topk in GROUP_TOPKS:
            self._check_rows([self._rand(group_topk + 1)], group_topk)

    def test_random_rows_batch(self):
        for group_topk in GROUP_TOPKS:
            rows = [self._rand(length) for length in (3000, 20000)]
            self._check_rows(rows, group_topk)

    def test_page_table_and_offset_expansion(self):
        group_topk = 512
        row = self._tight_cluster(STASH_ENTRIES + 5311)

        bias = 1000
        out = self._run([row], group_topk, page_table_bias=bias)
        self._check_row(out[0] - bias, row, group_topk)

        offset = 777
        out = self._run([row], group_topk, topk_offset=offset)
        self._check_row(out[0] - offset, row, group_topk)

    def test_tail_append_with_page_table_row_remap(self):
        """Tail layout and absolute page-table row selection work together."""
        group_topk = 512
        rows = [self._tight_cluster(STASH_ENTRIES + 5311), self._rand(500)]
        batch = len(rows)
        stride = max(row.numel() for row in rows)
        score = torch.full((batch, stride), float("inf"), dtype=torch.float32)
        lengths = torch.tensor([row.numel() for row in rows], dtype=torch.int32)
        for i, row in enumerate(rows):
            score[i, : row.numel()] = row

        pool_size = POOL_SIZE
        topk = group_topk * pool_size
        max_tokens = stride * pool_size + pool_size - 1
        page_biases = torch.tensor([10000, 20000, 30000], dtype=torch.int32)
        page_table = torch.arange(max_tokens, dtype=torch.int32).expand(3, -1)
        page_table = page_table + page_biases[:, None]
        page_table_row_index = torch.tensor([2, 0], dtype=torch.int32)
        tail_counts = (3, 1)
        seq_lens = torch.tensor(
            [rows[i].numel() * pool_size + tail_counts[i] for i in range(batch)],
            dtype=torch.int32,
        )

        out = fast_kpool_topk_transform_fused(
            score=score.cuda(),
            lengths=lengths.cuda(),
            pool_size=pool_size,
            topk=topk,
            page_table=page_table.cuda(),
            seq_lens=seq_lens.cuda(),
            page_table_row_index=page_table_row_index.cuda(),
        ).cpu()

        for i, row in enumerate(rows):
            bias = int(page_biases[page_table_row_index[i]])
            history_len = min(row.numel() * pool_size, topk)
            if row.numel() > group_topk:
                self._check_row(out[i, :history_len] - bias, row, group_topk)
            else:
                self.assertTrue(
                    torch.equal(
                        out[i, :history_len] - bias,
                        torch.arange(history_len, dtype=torch.int32),
                    )
                )

            tail = out[i, history_len : history_len + tail_counts[i]] - bias
            expected_tail = torch.arange(
                row.numel() * pool_size,
                row.numel() * pool_size + tail_counts[i],
                dtype=torch.int32,
            )
            self.assertTrue(torch.equal(tail, expected_tail))
            self.assertTrue(bool((out[i, history_len + tail_counts[i] :] == -1).all()))


if __name__ == "__main__":
    unittest.main()
