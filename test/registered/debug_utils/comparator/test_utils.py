import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.output_types import SummaryRecord
from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    argmax_coord,
    calc_per_token_rel_diff,
    calc_rel_diff,
    compute_exit_code,
    compute_smaller_dtype,
    try_unify_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestCalcRelDiff:
    def test_identical_tensors(self):
        x = torch.randn(10, 10)
        assert calc_rel_diff(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_tensors(self):
        result = calc_rel_diff(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        ).item()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_similar_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.01, 2.01, 3.01])
        result = calc_rel_diff(x, y).item()
        assert 0.0 < result < 0.01

    def test_negated_tensors(self):
        x = torch.tensor([1.0, 2.0])
        result = calc_rel_diff(x, -x).item()
        assert result == pytest.approx(2.0, abs=1e-5)


class TestCalcPerTokenRelDiff:
    def test_identical_tensors(self) -> None:
        """Identical tensors → per-token diff all zero."""
        x: torch.Tensor = torch.randn(8, 16)
        result: torch.Tensor = calc_per_token_rel_diff(x, x, seq_dim=0)

        assert result.shape == (8,)
        assert torch.allclose(result, torch.zeros(8), atol=1e-6)

    def test_different_tensors(self) -> None:
        """Single token position differs → that position has higher diff."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x.clone()
        y[3, :] += 10.0

        result: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=0)

        assert result.shape == (8,)
        assert result[3] > result[0]
        assert result[3] > result[7]
        for i in [0, 1, 2, 4, 5, 6, 7]:
            assert result[i] < 1e-6

    def test_seq_dim_selection(self) -> None:
        """Different seq_dim values produce correct output shapes."""
        x: torch.Tensor = torch.randn(4, 8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        assert calc_per_token_rel_diff(x, y, seq_dim=0).shape == (4,)
        assert calc_per_token_rel_diff(x, y, seq_dim=1).shape == (8,)
        assert calc_per_token_rel_diff(x, y, seq_dim=2).shape == (16,)

    def test_1d_tensor(self) -> None:
        """1D tensor with seq_dim=0 returns per-element diff."""
        x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        y: torch.Tensor = torch.tensor([1.0, 2.0, 4.0])

        result: torch.Tensor = calc_per_token_rel_diff(x, y, seq_dim=0)

        assert result.shape == (3,)
        assert result[0] < 1e-6
        assert result[1] < 1e-6
        assert result[2] > 0.01


class TestArgmaxCoord:
    def test_1d_tensor(self):
        x = torch.tensor([0.0, 0.0, 5.0, 0.0])
        assert argmax_coord(x) == (2,)

    def test_2d_tensor(self):
        x = torch.zeros(3, 4)
        x[1, 2] = 10.0
        assert argmax_coord(x) == (1, 2)

    def test_3d_tensor(self):
        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        assert argmax_coord(x) == (1, 2, 3)


class TestTryUnifyShape:
    def test_squeeze_leading_ones(self):
        target = torch.Size([3, 4])
        assert try_unify_shape(torch.randn(1, 1, 3, 4), target).shape == target

    def test_no_squeeze_when_leading_dim_not_one(self):
        target = torch.Size([3, 4])
        assert try_unify_shape(torch.randn(2, 3, 4), target).shape == (2, 3, 4)

    def test_same_shape_noop(self):
        target = torch.Size([3, 4])
        x = torch.randn(3, 4)
        result = try_unify_shape(x, target)
        assert result.shape == target
        assert result.data_ptr() == x.data_ptr()

    def test_trailing_dims_mismatch(self):
        target = torch.Size([5, 6])
        x = torch.randn(1, 3, 4)
        result = try_unify_shape(x, target)
        assert result.shape == (1, 3, 4)


class TestComputeSmallerDtype:
    def test_float32_bfloat16(self):
        assert (
            compute_smaller_dtype(Pair(x=torch.float32, y=torch.bfloat16))
            == torch.bfloat16
        )

    def test_reverse_order(self):
        assert (
            compute_smaller_dtype(Pair(x=torch.bfloat16, y=torch.float32))
            == torch.bfloat16
        )

    def test_same_dtype_returns_none(self):
        assert compute_smaller_dtype(Pair(x=torch.float32, y=torch.float32)) is None

    def test_unknown_pair_returns_none(self):
        assert compute_smaller_dtype(Pair(x=torch.int32, y=torch.int64)) is None


class TestPairMap:
    def test_map_basic(self):
        pair = Pair(x=[1, 2, 3], y=[4, 5, 6])
        result = pair.map(lambda lst: sum(lst))
        assert result.x == 6
        assert result.y == 15

    def test_map_type_change(self):
        pair = Pair(x=[1, 2, 3], y=[10, 20])
        result = pair.map(len)
        assert result.x == 3
        assert result.y == 2

    def test_map_returns_new_pair(self):
        pair = Pair(x="hello", y="world")
        result = pair.map(str.upper)
        assert result.x == "HELLO"
        assert result.y == "WORLD"
        assert result is not pair


class TestComputeExitCode:
    """Unit tests for compute_exit_code logic."""

    def test_all_passed(self):
        """All passed → exit 0."""
        summary = SummaryRecord(total=3, passed=3, failed=0, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 0
        )

    def test_has_failed_and_passed(self):
        """Has failed and passed → exit 1."""
        summary = SummaryRecord(total=4, passed=2, failed=2, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=None,
                failed_names=["a", "b"],
            )
            == 1
        )

    def test_all_failed(self):
        """All failed (0 passed) → exit 1."""
        summary = SummaryRecord(total=3, passed=0, failed=3, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=None,
                failed_names=["a", "b", "c"],
            )
            == 1
        )

    def test_all_skipped_allow_all(self):
        """All skipped + allow_skipped_pattern='.*' → exit 1 (nothing passed)."""
        summary = SummaryRecord(total=2, passed=0, failed=0, skipped=2)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=["a", "b"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 1
        )

    def test_all_skipped_forbid_all(self):
        """All skipped + allow_skipped_pattern='^$' → exit 1."""
        summary = SummaryRecord(total=2, passed=0, failed=0, skipped=2)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="^$",
                skipped_names=["a", "b"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 1
        )

    def test_passed_and_skipped_allow_all(self):
        """Passed + skipped, allow all → exit 0."""
        summary = SummaryRecord(total=3, passed=2, failed=0, skipped=1)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=["a"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 0
        )

    def test_passed_and_skipped_forbid_all(self):
        """Passed + skipped + forbid all → exit 1."""
        summary = SummaryRecord(total=3, passed=2, failed=0, skipped=1)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="^$",
                skipped_names=["a"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 1
        )

    def test_skip_pattern_matches_specific_name(self):
        """Pattern matching specific name allows that skip, forbids others."""
        summary = SummaryRecord(total=4, passed=2, failed=0, skipped=2)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="positions|seq_lens",
                skipped_names=["positions", "seq_lens"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 0
        )

    def test_skip_pattern_partial_match_forbidden(self):
        """Pattern matches some skips but not all → exit 1."""
        summary = SummaryRecord(total=4, passed=1, failed=0, skipped=3)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="positions|seq_lens",
                skipped_names=["positions", "seq_lens", "hidden_states"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 1
        )

    def test_allow_failed_pattern_matches_all(self):
        """allow_failed_pattern='.*' tolerates all failures → exit 0."""
        summary = SummaryRecord(total=3, passed=1, failed=2, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=".*",
                failed_names=["a", "b"],
            )
            == 0
        )

    def test_allow_failed_pattern_matches_specific(self):
        """Pattern matches all failed names → exit 0."""
        summary = SummaryRecord(total=3, passed=1, failed=2, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern="hidden_states|logits",
                failed_names=["hidden_states", "logits"],
            )
            == 0
        )

    def test_allow_failed_pattern_partial_match(self):
        """Pattern matches some but not all failures → exit 1."""
        summary = SummaryRecord(total=3, passed=0, failed=3, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern="hidden_states",
                failed_names=["hidden_states", "logits", "attn"],
            )
            == 1
        )

    def test_allow_failed_pattern_no_failures(self):
        """Pattern set but no failures → exit 0."""
        summary = SummaryRecord(total=2, passed=2, failed=0, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=".*",
                failed_names=[],
            )
            == 0
        )

    def test_both_failed_and_skipped_patterns(self):
        """Both patterns set, both satisfied → exit 0."""
        summary = SummaryRecord(total=4, passed=1, failed=1, skipped=2)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="positions|seq_lens",
                skipped_names=["positions", "seq_lens"],
                allow_failed_pattern="logits",
                failed_names=["logits"],
            )
            == 0
        )

    def test_failed_pattern_satisfied_but_skipped_not(self):
        """Failed pattern OK but skipped pattern fails → exit 1."""
        summary = SummaryRecord(total=3, passed=1, failed=1, skipped=1)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern="^$",
                skipped_names=["a"],
                allow_failed_pattern=".*",
                failed_names=["b"],
            )
            == 1
        )

    def test_zero_passed_exits_one(self):
        """No tensors passed → exit 1, even when all failures are allowed."""
        summary = SummaryRecord(total=2, passed=0, failed=2, skipped=0)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=[],
                allow_failed_pattern=".*",
                failed_names=["a", "b"],
            )
            == 1
        )

    def test_zero_passed_all_skipped_exits_one(self):
        """All skipped, nothing passed → exit 1."""
        summary = SummaryRecord(total=3, passed=0, failed=0, skipped=3)
        assert (
            compute_exit_code(
                summary,
                allow_skipped_pattern=".*",
                skipped_names=["a", "b", "c"],
                allow_failed_pattern=None,
                failed_names=[],
            )
            == 1
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
