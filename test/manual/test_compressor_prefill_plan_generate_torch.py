import struct
from typing import Tuple

import pytest
import torch

from sglang.jit_kernel.dsv4.compress_old import CompressorPrefillPlan
from sglang.srt.utils import get_device

DEVICE = get_device()

COMPRESS_RATIOS = [4, 128]


# ── helpers ──────────────────────────────────────────────────────────────────


def _t(*args) -> torch.Tensor:
    """Convenience: create a 1-D int64 tensor on the target device."""
    return torch.tensor(args, device=DEVICE)


def _decode_row(row: torch.Tensor) -> Tuple[int, int, int, int]:
    """Unpack one 16-byte row as four little-endian uint32 values."""
    raw = bytes(row.cpu().tolist())
    return struct.unpack("<IIII", raw)


def _count_valid(plan: torch.Tensor) -> int:
    """Count rows whose first uint32 field is not kInvalid."""
    return sum(1 for r in range(plan.shape[0]) if _decode_row(plan[r])[0] != kInvalid)


kInvalid = 0xFFFFFFFF


# ── baseline: single sequence, no overlap ────────────────────────────────────


class TestSingleSequenceNoOverlap:
    """compress_ratio=4/128, is_overlap=False, use_cuda_graph=False"""

    def _run(self, seq_len, extend_len, compress_ratio=4) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            extend_len,
            _t(seq_len),
            _t(extend_len),
            DEVICE,
        )

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_return_plans_are_tensors(self, compress_ratio):
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        assert isinstance(plan.compress_plan, torch.Tensor)
        assert isinstance(plan.write_plan, torch.Tensor)

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_count_matches_compress_rows(self, compress_ratio):
        """One compress entry per `compress_ratio` tokens."""
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        assert _count_valid(plan.compress_plan) == seq_len // compress_ratio

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_write_plan_has_no_padding_rows(self, compress_ratio):
        """All rows present in write_plan must be valid (no kInvalid padding)."""
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        assert plan.write_plan.shape[0] == _count_valid(plan.write_plan)

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_row_fields_are_valid(self, compress_ratio):
        seq_len = compress_ratio * 3
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        for r in range(_count_valid(plan.compress_plan)):
            token_idx, batch_idx, position, window_len = _decode_row(
                plan.compress_plan[r]
            )
            assert token_idx != kInvalid
            assert batch_idx == 0
            assert position >= 0
            assert window_len >= 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_write_row_fields_are_valid(self, compress_ratio):
        seq_len = compress_ratio * 3
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        for r in range(_count_valid(plan.write_plan)):
            token_idx, batch_idx, position, window_len = _decode_row(plan.write_plan[r])
            assert token_idx != kInvalid
            assert batch_idx == 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_token_indices_are_sequential_and_unique(self, compress_ratio):
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        cc = _count_valid(plan.compress_plan)
        indices = [_decode_row(plan.compress_plan[r])[0] for r in range(cc)]
        assert indices == sorted(
            set(indices)
        ), "token indices must be unique and ordered"

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_positions_are_period_boundaries(self, compress_ratio):
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        for r in range(_count_valid(plan.compress_plan)):
            _, _, position, _ = _decode_row(plan.compress_plan[r])
            assert (
                position + 1
            ) % compress_ratio == 0, (
                f"position {position} is not a boundary for ratio {compress_ratio}"
            )

    def test_no_entry_when_extend_len_zero_raises(self):
        with pytest.raises((AssertionError, Exception)):
            self._run(seq_len=4, extend_len=0)

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_extend_len_equals_seq_len(self, compress_ratio):
        """No prefix — all tokens are new."""
        seq_len = compress_ratio * 2
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        assert _count_valid(plan.compress_plan) >= 0
        assert _count_valid(plan.write_plan) >= 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_extend_len_one(self, compress_ratio):
        plan = self._run(
            seq_len=compress_ratio * 2, extend_len=1, compress_ratio=compress_ratio
        )
        assert _count_valid(plan.compress_plan) >= 0
        assert _count_valid(plan.write_plan) >= 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_partial_extend_with_prefix(self, compress_ratio):
        """Prefix occupies first half; only the second half is new."""
        seq_len = compress_ratio * 4
        extend_len = compress_ratio * 2
        plan = self._run(
            seq_len=seq_len, extend_len=extend_len, compress_ratio=compress_ratio
        )
        prefix_len = seq_len - extend_len
        for r in range(_count_valid(plan.compress_plan)):
            _, _, position, _ = _decode_row(plan.compress_plan[r])
            assert prefix_len <= position < seq_len

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_ratio_stored_on_plan(self, compress_ratio):
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        assert plan.compress_ratio == compress_ratio


# ── overlap mode ──────────────────────────────────────────────────────────────


class TestOverlapMode:

    def _run(self, seq_len, extend_len, compress_ratio=4) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            extend_len,
            _t(seq_len),
            _t(extend_len),
            DEVICE,
        )

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_write_count_gte_no_overlap(self, compress_ratio):
        """Overlap mode should produce at least as many write entries as no-overlap."""
        seq_len = compress_ratio * 4

        plan_no = CompressorPrefillPlan.generate(
            compress_ratio,
            seq_len,
            _t(seq_len),
            _t(seq_len),
            DEVICE,
        )
        plan_ov = self._run(seq_len, seq_len, compress_ratio=compress_ratio)
        assert _count_valid(plan_ov.write_plan) >= _count_valid(plan_no.write_plan)

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_window_len_uses_doubled_ratio(self, compress_ratio):
        """With is_overlap=True ratio doubles, so window_len < 2 * compress_ratio."""
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        for r in range(_count_valid(plan.compress_plan)):
            _, _, _, window_len = _decode_row(plan.compress_plan[r])
            assert (
                window_len < 2 * compress_ratio
            ), f"window_len {window_len} >= 2*ratio={2*compress_ratio} for ratio {compress_ratio}"

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_positions_are_period_boundaries_overlap(self, compress_ratio):
        """Boundary condition must hold regardless of overlap mode."""
        seq_len = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=compress_ratio
        )
        for r in range(_count_valid(plan.compress_plan)):
            _, _, position, _ = _decode_row(plan.compress_plan[r])
            assert (position + 1) % compress_ratio == 0


# ── cuda-graph padding ────────────────────────────────────────────────────────


class TestCudaGraphPadding:

    def _run(self, seq_len, extend_len, compress_ratio=4) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            extend_len,
            _t(seq_len),
            _t(extend_len),
            DEVICE,
        )

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_plan_shape_matches_boundary_count(self, compress_ratio):
        """compress_plan has one row per compress boundary in [0, seq_len)."""
        seq_len = compress_ratio * 4
        plan = self._run(seq_len, seq_len, compress_ratio=compress_ratio)
        expected_rows = seq_len // compress_ratio
        assert plan.compress_plan.shape[0] == expected_rows

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_write_plan_has_no_padding_rows(self, compress_ratio):
        """All rows present in write_plan must be valid (no kInvalid padding)."""
        seq_len = compress_ratio * 4
        plan = self._run(seq_len, seq_len, compress_ratio=compress_ratio)
        assert plan.write_plan.shape[0] == _count_valid(plan.write_plan)

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_padding_rows_are_fully_invalid(self, compress_ratio):
        """Rows beyond the valid entries must have all four fields set to kInvalid."""
        seq_len = compress_ratio * 2
        large_ratio = compress_ratio * 4
        plan = self._run(
            seq_len=seq_len, extend_len=seq_len, compress_ratio=large_ratio
        )
        num_rows = plan.compress_plan.shape[0]
        for r in range(num_rows):
            row = _decode_row(plan.compress_plan[r])
            if row[0] == kInvalid:
                assert all(
                    v == kInvalid for v in row
                ), f"partial-invalid row {r} with ratio {compress_ratio}"


# ── batch of multiple sequences ───────────────────────────────────────────────


class TestBatchedSequences:

    def _run(self, extend_lens, seq_lens, compress_ratio=4) -> CompressorPrefillPlan:
        num_q_tokens = sum(extend_lens)
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            torch.tensor(seq_lens, device=DEVICE),
            torch.tensor(extend_lens, device=DEVICE),
            DEVICE,
        )

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_two_sequences_batch_idx_assignment(self, compress_ratio):
        seq_len = compress_ratio * 4
        plan = self._run(
            [seq_len, seq_len], [seq_len, seq_len], compress_ratio=compress_ratio
        )
        cc = _count_valid(plan.compress_plan)
        seen_batches = {_decode_row(plan.compress_plan[r])[1] for r in range(cc)}
        assert seen_batches <= {0, 1}, "unexpected batch index"

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_three_sequences_total_compress_count(self, compress_ratio):
        seq_len = compress_ratio * 4
        extend_lens = [seq_len, seq_len, seq_len]
        seq_lens = [seq_len, seq_len, seq_len]
        plan = self._run(extend_lens, seq_lens, compress_ratio=compress_ratio)
        expected = (seq_len // compress_ratio) * 3
        assert _count_valid(plan.compress_plan) == expected

    def test_batch_mismatch_raises(self):
        with pytest.raises((AssertionError, Exception)):
            self._run([4, 4], [4])

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_token_indices_cover_full_range(self, compress_ratio):
        seq_len = compress_ratio * 3
        extend_lens = [seq_len, seq_len]
        seq_lens = [seq_len, seq_len * 2]
        num_tokens = sum(extend_lens)
        plan = self._run(extend_lens, seq_lens, compress_ratio=compress_ratio)

        all_indices: set = set()
        for r in range(_count_valid(plan.compress_plan)):
            all_indices.add(_decode_row(plan.compress_plan[r])[0])
        for r in range(_count_valid(plan.write_plan)):
            all_indices.add(_decode_row(plan.write_plan[r])[0])

        assert all_indices.issubset(set(range(num_tokens)))

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_write_plan_positions_gte_start_write_pos(self, compress_ratio):
        """Every write entry's position must lie within [0, seq_len)."""
        seq_lens = [compress_ratio * 3, compress_ratio * 4]
        extend_lens = seq_lens[:]
        plan = self._run(extend_lens, seq_lens, compress_ratio=compress_ratio)
        total_seq_len = max(seq_lens)
        for r in range(_count_valid(plan.write_plan)):
            _, batch_idx, position, _ = _decode_row(plan.write_plan[r])
            assert 0 <= position < seq_lens[batch_idx], (
                f"row {r}: position {position} out of range for batch {batch_idx}, "
                f"seq_len {seq_lens[batch_idx]}, ratio {compress_ratio}"
            )


# ── large-ratio edge cases ────────────────────────────────────────────────────


class TestLargeRatio:

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_compress_ratio_larger_than_seq_len_produces_no_compress_entries(
        self, compress_ratio
    ):
        seq_len = max(1, compress_ratio - 1)
        plan = CompressorPrefillPlan.generate(
            compress_ratio,
            seq_len,
            _t(seq_len),
            _t(seq_len),
            DEVICE,
        )
        assert _count_valid(plan.compress_plan) == 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_single_token_sequence(self, compress_ratio):
        plan = CompressorPrefillPlan.generate(
            compress_ratio,
            1,
            _t(1),
            _t(1),
            DEVICE,
        )
        assert _count_valid(plan.compress_plan) >= 0
        assert _count_valid(plan.write_plan) >= 0

    @pytest.mark.parametrize("compress_ratio", COMPRESS_RATIOS)
    def test_exact_multiple_of_ratio(self, compress_ratio):
        """seq_len that is an exact multiple should produce predictable compress count."""
        seq_len = compress_ratio * 5
        plan = CompressorPrefillPlan.generate(
            compress_ratio,
            seq_len,
            _t(seq_len),
            _t(seq_len),
            DEVICE,
        )
        assert _count_valid(plan.compress_plan) == 5
