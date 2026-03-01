import subprocess
import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import sglang.srt.debug_utils.dumper as _dumper_module
from sglang.srt.debug_utils.comparator.entrypoint import _compute_exit_code, run
from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    ComparisonRecord,
    ConfigRecord,
    GeneralWarning,
    NonTensorRecord,
    ReplicatedCheckResult,
    SkipRecord,
    SummaryRecord,
    WarningRecord,
    _OutputRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper, _RecomputeStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)

_FIXED_EXP_NAME = "my_exp_name"

# Each test has a one-line docstring describing the scenario it covers.


class TestEntrypointGroupingRaw:
    """Test `--grouping raw` scenarios"""

    def test_run_basic(self, tmp_path, capsys):
        """Two matching tensors produce ConfigRecord, 2 ComparisonRecords, and SummaryRecord."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        assert isinstance(records[0], ConfigRecord)

        assert len(_get_comparisons(records)) == 2

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.skipped == 0

    def test_filter(self, tmp_path, capsys):
        """--filter selects only the matching tensor, producing 1 ComparisonRecord."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, filter="tensor_a", grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        assert len(_get_comparisons(records)) == 1

    def test_no_baseline_skip(self, tmp_path, capsys):
        """Target tensor missing from baseline emits a SkipRecord with reason baseline_load_failed."""
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == "baseline_load_failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.skipped == 1

    def test_step_range(self, tmp_path, capsys):
        """--start_step/--end_step restricts comparison to a single step out of three."""
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=3)
        args = _make_args(
            baseline_path, target_path, start_step=1, end_step=1, grouping="raw"
        )

        records, _ = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1

    def test_all_valid_records(self, tmp_path, capsys):
        """Every emitted JSON record is a valid _OutputRecord subclass."""
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=2)
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        assert all(isinstance(r, _OutputRecord) for r in records)

    def test_comparison_failed(self, tmp_path, capsys):
        """Completely different tensors produce a failed ComparisonRecord."""
        torch.manual_seed(42)
        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=torch.randn(10, 10)
        )
        target_path = _create_rank_dump(
            tmp_path / "target",
            rank=0,
            name="tensor_a",
            tensor=torch.randn(10, 10) * 100,
        )
        args = _make_args(
            baseline_path, target_path, grouping="raw", diff_threshold=1e-3
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert not comparisons[0].diff.passed
        assert comparisons[0].category == "failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1

    def test_shape_mismatch(self, tmp_path, capsys):
        """Different shapes produce shape_mismatch=True and category='failed'."""
        torch.manual_seed(42)
        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=torch.randn(4, 8)
        )
        target_path = _create_rank_dump(
            tmp_path / "target", rank=0, name="tensor_a", tensor=torch.randn(4, 10)
        )
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].shape_mismatch is True
        assert comparisons[0].diff is None
        assert comparisons[0].category == "failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1

    def test_unify_shape_leading_dims(self, tmp_path, capsys):
        """Leading singleton dims on baseline are squeezed to match target shape."""
        torch.manual_seed(42)
        base_tensor = torch.randn(4, 8)
        baseline_tensor = base_tensor.unsqueeze(0)  # (1, 4, 8)
        target_tensor = base_tensor + torch.randn(4, 8) * 0.0001  # (4, 8)

        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=baseline_tensor
        )
        target_path = _create_rank_dump(
            tmp_path / "target", rank=0, name="tensor_a", tensor=target_tensor
        )
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1

        comp = comparisons[0]
        assert comp.shape_mismatch is False
        assert comp.baseline.shape == [1, 4, 8]
        assert comp.target.shape == [4, 8]
        assert comp.unified_shape == [4, 8]
        assert comp.diff is not None
        assert comp.diff.passed

    def test_dtype_mismatch_downcast(self, tmp_path, capsys):
        """Baseline float32 vs target bfloat16 produces diff_downcast."""
        torch.manual_seed(42)
        baseline_tensor = torch.randn(4, 8, dtype=torch.float32)
        target_tensor = (baseline_tensor + torch.randn(4, 8) * 0.0001).to(
            torch.bfloat16
        )

        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=baseline_tensor
        )
        target_path = _create_rank_dump(
            tmp_path / "target", rank=0, name="tensor_a", tensor=target_tensor
        )
        args = _make_args(
            baseline_path, target_path, grouping="raw", diff_threshold=0.01
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff_downcast is not None
        assert comparisons[0].downcast_dtype is not None

    def test_mixed_summary(self, tmp_path, capsys):
        """One passed, one failed, one skipped tensor in a single run."""
        torch.manual_seed(42)
        similar_tensor = torch.randn(4, 4)
        different_baseline = torch.randn(4, 4)
        different_target = torch.randn(4, 4) * 100
        extra_tensor = torch.randn(4, 4)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_rank_dump(baseline_dir, rank=0, name="similar", tensor=similar_tensor)
        _create_rank_dump(
            baseline_dir, rank=0, name="different", tensor=different_baseline
        )

        _create_rank_dump(
            target_dir,
            rank=0,
            name="similar",
            tensor=similar_tensor + torch.randn(4, 4) * 0.0001,
        )
        _create_rank_dump(target_dir, rank=0, name="different", tensor=different_target)
        _create_rank_dump(target_dir, rank=0, name="extra", tensor=extra_tensor)

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            diff_threshold=1e-3,
        )

        records, _ = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.skipped == 1
        assert summary.total == 3

    def test_filter_empty_result(self, tmp_path, capsys):
        """--filter matching nothing produces summary with total=0."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        args = _make_args(
            baseline_path, target_path, filter="nonexistent_pattern", grouping="raw"
        )

        records, _ = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 0

    def test_raw_multi_rank(self, tmp_path, capsys):
        """Two ranks in raw grouping produce two ComparisonRecords (one per rank)."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 4)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for rank in range(2):
            _create_rank_dump(baseline_dir, rank=rank, name="hidden", tensor=tensor)
            _create_rank_dump(
                target_dir,
                rank=rank,
                name="hidden",
                tensor=tensor + torch.randn(4, 4) * 0.0001,
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2

    def test_text_output_smoke(self, tmp_path, capsys):
        """Text output format renders without errors and contains Config/Summary sections."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        args = _make_args(
            baseline_path, target_path, output_format="text", grouping="raw"
        )
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Config:" in output
        assert "Summary:" in output

    def test_text_output_with_failure(self, tmp_path, capsys):
        """Text output with a failed comparison renders failure info."""
        torch.manual_seed(42)
        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=torch.randn(10, 10)
        )
        target_path = _create_rank_dump(
            tmp_path / "target",
            rank=0,
            name="tensor_a",
            tensor=torch.randn(10, 10) * 100,
        )
        args = _make_args(
            baseline_path, target_path, output_format="text", grouping="raw"
        )
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Summary:" in output
        assert "failed" in output.lower()

    def test_duplicate_dump_pairing(self, tmp_path, capsys):
        """Same name dumped twice (different values) pairs by duplicate_index: 0th↔0th, 1st↔1st."""
        torch.manual_seed(42)
        tensor_v0 = torch.randn(4, 4)
        tensor_v1 = torch.randn(4, 4)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir in [baseline_dir, target_dir]:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(_dumper_module, "_get_rank", lambda: 0)
                dumper = _Dumper(
                    config=DumperConfig(
                        enable=True,
                        dir=str(side_dir),
                        exp_name=_FIXED_EXP_NAME,
                    )
                )
                dumper.__dict__["_static_meta"] = {"world_rank": 0, "world_size": 1}

                dumper.dump("tensor_a", tensor_v0)
                dumper.dump("tensor_a", tensor_v1)
                dumper.step()

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2
        assert all(c.diff is not None and c.diff.passed for c in comparisons)

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2


class TestEntrypointGroupingLogical:
    """Test `--grouping logical` scenarios"""

    def test_no_dims_single_rank(self, tmp_path, capsys):
        """Single-rank dumps without dims fall back to raw loading."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path)

        records, _ = _run_and_parse(args, capsys)
        assert len(_get_comparisons(records)) == 2
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.skipped == 0

    def test_tp_unshard_same_size(self, tmp_path, capsys):
        """Both sides TP=2: shards are concatenated before comparison."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8)
        full_target = full_baseline + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        target_path = _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_target,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1
        assert summary.passed == 1

    def test_tp_unshard_different_sizes(self, tmp_path, capsys):
        """Baseline TP=4 vs target TP=2: different shard counts are handled correctly."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8)
        full_target = full_baseline + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="hidden",
            tp_size=4,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        target_path = _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_target,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        _assert_single_comparison_passed(records)

    def test_one_side_dims_single_baseline(self, tmp_path, capsys):
        """Baseline has no dims (single rank), target has TP shards: unshard target only."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)
        target_full = full_tensor + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=full_tensor
        )

        target_path = _create_tp_sharded_dumps(
            target_dir,
            full_tensor=target_full,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        _assert_single_comparison_passed(records)

    @pytest.mark.parametrize(
        "bad_side, expected_reason",
        [
            ("baseline", "baseline_load_failed"),
            ("target", "target_load_failed"),
        ],
    )
    def test_ambiguous_no_dims_skip(self, tmp_path, capsys, bad_side, expected_reason):
        """Multi-rank without dims on one side produces a SkipRecord with the appropriate reason."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        good_dir = target_dir if bad_side == "baseline" else baseline_dir
        bad_dir = baseline_dir if bad_side == "baseline" else target_dir

        _create_rank_dump(good_dir, rank=0, name="hidden", tensor=tensor)
        for rank, shard in [(0, tensor[:, :4]), (1, tensor[:, 4:])]:
            _create_rank_dump(bad_dir, rank=rank, name="hidden", tensor=shard)

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
        )

        records, _ = _run_and_parse(args, capsys)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == expected_reason

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.skipped == 1

    def test_summary_counts_unshard(self, tmp_path, capsys):
        """Two TP-sharded tensors: summary counts total=2, passed=2, skipped=0."""
        torch.manual_seed(42)
        full_a = torch.randn(4, 8)
        full_b = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for tensor_name, tensor in [("t_a", full_a), ("t_b", full_b)]:
            baseline_path = _create_tp_sharded_dumps(
                baseline_dir,
                full_tensor=tensor,
                name=tensor_name,
                tp_size=2,
                shard_dim=1,
                dims_str="b h(tp)",
            )
            target_tensor = tensor + torch.randn_like(tensor) * 0.0001
            target_path = _create_tp_sharded_dumps(
                target_dir,
                full_tensor=target_tensor,
                name=tensor_name,
                tp_size=2,
                shard_dim=1,
                dims_str="b h(tp)",
            )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.skipped == 0

    def test_multi_step_tp(self, tmp_path, capsys):
        """Two steps with TP=2 shards: concat mode merges into one comparison."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_tensor,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
            num_steps=2,
        )
        target_path = _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_tensor + torch.randn(4, 8) * 0.0001,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
            num_steps=2,
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        # concat along dim 0 (fallback, no token dim) → 2 steps × [4, 8] = [8, 8]
        assert comparisons[0].baseline.shape == [8, 8]

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1
        assert summary.passed == 1

    def test_cp_axis_unshard(self, tmp_path, capsys):
        """CP-sharded tensors are correctly concatenated along the sequence dim."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            shards = list(full_tensor.chunk(2, dim=1))
            for cp_rank in range(2):
                _create_rank_dump(
                    side_dir,
                    rank=cp_rank,
                    name="attn_out",
                    tensor=shards[cp_rank],
                    dims="b s(cp) h",
                    parallel_info={"cp_rank": cp_rank, "cp_size": 2},
                )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "attn_out"

    def test_filter_logical(self, tmp_path, capsys):
        """--filter in logical grouping selects only matching tensor bundles."""
        torch.manual_seed(42)
        full_a = torch.randn(4, 8)
        full_b = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for tensor_name, tensor in [("t_a", full_a), ("t_b", full_b)]:
            _create_tp_sharded_dumps(
                baseline_dir,
                full_tensor=tensor,
                name=tensor_name,
                tp_size=2,
                shard_dim=1,
                dims_str="b h(tp)",
            )
            _create_tp_sharded_dumps(
                target_dir,
                full_tensor=tensor + torch.randn_like(tensor) * 0.0001,
                name=tensor_name,
                tp_size=2,
                shard_dim=1,
                dims_str="b h(tp)",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            filter="t_a",
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].name == "t_a"

    def test_mixed_dims_logical(self, tmp_path, capsys):
        """TP-sharded and single-rank tensors in the same logical run both compare successfully."""
        torch.manual_seed(42)
        full_tp_tensor = torch.randn(4, 8)
        single_tensor = torch.randn(4, 4)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_tp_tensor,
            name="tensor_a",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_tp_tensor + torch.randn(4, 8) * 0.0001,
            name="tensor_a",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        _create_rank_dump(baseline_dir, rank=0, name="tensor_b", tensor=single_tensor)
        _create_rank_dump(
            target_dir,
            rank=0,
            name="tensor_b",
            tensor=single_tensor + torch.randn(4, 4) * 0.0001,
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2
        assert all(c.diff is not None and c.diff.passed for c in comparisons)
        assert {c.name for c in comparisons} == {"tensor_a", "tensor_b"}

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2

    def test_cp_tp_unshard(self, tmp_path, capsys):
        """CP=2 + TP=2: multi-axis shards are unsharded before comparison."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 16)
        full_target = full_baseline + torch.randn(4, 8, 16) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_cp_tp_sharded_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="hidden",
                cp_size=2,
                tp_size=2,
                seq_dim=1,
                head_dim=2,
                dims_str="b s(cp) h(tp)",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_cp_tp_different_sizes(self, tmp_path, capsys):
        """Baseline CP=2+TP=2 vs target CP=1+TP=4: both sides independently unsharder."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 16)
        full_target = full_baseline + torch.randn(4, 8, 16) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_cp_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="hidden",
            cp_size=2,
            tp_size=2,
            seq_dim=1,
            head_dim=2,
            dims_str="b s(cp) h(tp)",
        )

        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_target,
            name="hidden",
            tp_size=4,
            shard_dim=2,
            dims_str="b s h(tp)",
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        _assert_single_comparison_passed(records)

    def test_ep_cp_tp_three_axis_unshard(self, tmp_path, capsys):
        """EP=2 + CP=2 + TP=2: three-axis shards are unsharded before comparison."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 16, 32)
        full_target = full_baseline + torch.randn(4, 8, 16, 32) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_ep_cp_tp_sharded_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="hidden",
                ep_size=2,
                cp_size=2,
                tp_size=2,
                expert_dim=1,
                seq_dim=2,
                head_dim=3,
                dims_str="b e(ep) s(cp) h(tp)",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_cp_zigzag_unshard(self, tmp_path, capsys):
        """CP=2 zigzag reorder is correctly undone through the full pipeline."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_cp_zigzag_tp_sharded_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="attn_out",
                cp_size=2,
                tp_size=1,
                seq_dim=1,
                head_dim=2,
                dims_str="b s(cp:zigzag) h",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "attn_out"

    def test_cp_zigzag_tp_unshard(self, tmp_path, capsys):
        """CP=2 zigzag + TP=2: multi-axis unshard with reorder through full pipeline."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 16)
        full_target = full_baseline + torch.randn(4, 8, 16) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_cp_zigzag_tp_sharded_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="hidden",
                cp_size=2,
                tp_size=2,
                seq_dim=1,
                head_dim=2,
                dims_str="b s(cp:zigzag) h(tp)",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_recompute_pseudo_replicated_verification(self, tmp_path, capsys):
        """Recompute pseudo-axis with identical original/recompute tensors → passed."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir in [baseline_dir, target_dir]:
            _create_recompute_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                original_tensor=tensor,
                recompute_tensor=tensor.clone(),
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_recompute_pseudo_mismatch_warning(self, tmp_path, capsys):
        """Recompute pseudo-axis with differing original/recompute → failed replicated_checks."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 8)
        mismatched_tensor = tensor + torch.randn(4, 8) * 10.0

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir in [baseline_dir, target_dir]:
            _create_recompute_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                original_tensor=tensor,
                recompute_tensor=mismatched_tensor,
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1

        recompute_checks: list[ReplicatedCheckResult] = [
            c for c in comparisons[0].replicated_checks if c.axis == "recompute_pseudo"
        ]
        assert len(recompute_checks) > 0
        assert any(not c.passed for c in recompute_checks)

    def test_tp_partial_reduction_unshard(self, tmp_path, capsys):
        """TP=2 with partial reduction: element-wise sum reconstructs full tensor."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8)
        full_target = full_baseline + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_tp_partial_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="attn_out",
            tp_size=2,
            dims_str="b h(tp:partial)",
        )
        target_path = _create_tp_partial_dumps(
            target_dir,
            full_tensor=full_target,
            name="attn_out",
            tp_size=2,
            dims_str="b h(tp:partial)",
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "attn_out"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1
        assert summary.passed == 1

    def test_tp_partial_vs_single_rank(self, tmp_path, capsys):
        """Baseline single rank vs target TP=2 partial: unshard target then compare."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)
        target_full = full_tensor + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        baseline_path = _create_rank_dump(
            baseline_dir, rank=0, name="attn_out", tensor=full_tensor
        )
        target_path = _create_tp_partial_dumps(
            target_dir,
            full_tensor=target_full,
            name="attn_out",
            tp_size=2,
            dims_str="b h(tp:partial)",
        )

        args = _make_args(baseline_path, target_path, diff_threshold=0.01)

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "attn_out"

    def test_cp_concat_tp_partial_reduction(self, tmp_path, capsys):
        """CP=2 concat + TP=2 partial reduction: multi-axis unshard."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 16)
        full_target = full_baseline + torch.randn(4, 8, 16) * 0.001

        for side_dir, full_tensor in [
            (tmp_path / "baseline", full_baseline),
            (tmp_path / "target", full_target),
        ]:
            side_dir.mkdir()
            cp_chunks = list(full_tensor.chunk(2, dim=1))
            rank = 0
            for cp_rank in range(2):
                for tp_rank in range(2):
                    _create_rank_dump(
                        side_dir,
                        rank=rank,
                        name="hidden",
                        tensor=cp_chunks[cp_rank] / 2,
                        dims="b s(cp) h(tp:partial)",
                        parallel_info={
                            "cp_rank": cp_rank,
                            "cp_size": 2,
                            "tp_rank": tp_rank,
                            "tp_size": 2,
                        },
                    )
                    rank += 1

        args = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_cp_zigzag_sp_same_dim_unshard(self, tmp_path, capsys):
        """CP=2 zigzag + SP=2 on same seq dim: multi-axis unshard + reorder."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_cp_zigzag_sp_sharded_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="hidden",
                cp_size=2,
                sp_size=2,
                dims_str="b s(cp:zigzag,sp) h",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"


class TestEntrypointConcatMode:
    """Test concat token-aligner mode through the full entrypoint pipeline."""

    @staticmethod
    def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()
        return baseline_dir, target_dir

    @staticmethod
    def _create_both_sides(
        tmp_path: Path,
        *,
        baseline_steps: list[torch.Tensor],
        target_steps: list[torch.Tensor],
        name: str = "hidden",
        dims: str | None = None,
    ) -> tuple[Path, Path]:
        """Create multi-step rank-0 dumps for both sides and return exp paths."""
        baseline_dir, target_dir = TestEntrypointConcatMode._make_dirs(tmp_path)

        for side_dir, steps in [
            (baseline_dir, baseline_steps),
            (target_dir, target_steps),
        ]:
            _create_multi_step_rank_dump(
                side_dir,
                rank=0,
                name=name,
                tensors_per_step=steps,
                dims=dims,
            )

        return baseline_dir / _FIXED_EXP_NAME, target_dir / _FIXED_EXP_NAME

    @staticmethod
    def _run_concat(
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        *,
        baseline_steps: list[torch.Tensor],
        target_steps: list[torch.Tensor],
        name: str = "hidden",
        dims: str | None = None,
        diff_threshold: float = 0.01,
    ) -> list[AnyRecord]:
        """Create both-side dumps, run comparator, return parsed records."""
        baseline_path, target_path = TestEntrypointConcatMode._create_both_sides(
            tmp_path,
            baseline_steps=baseline_steps,
            target_steps=target_steps,
            name=name,
            dims=dims,
        )
        args: Namespace = _make_args(
            baseline_path, target_path, diff_threshold=diff_threshold
        )
        records, _ = _run_and_parse(args, capsys)
        return records

    def test_concat_multi_step_different_data(self, tmp_path, capsys):
        """Multi-step concat with different data per step + truncation."""
        torch.manual_seed(42)

        # baseline: 2 steps [5,4] + [3,4] → concat → [8,4]
        baseline_step0 = torch.randn(5, 4)
        baseline_step1 = torch.randn(3, 4)
        baseline_concat = torch.cat([baseline_step0, baseline_step1], dim=0)

        # target: 1 step [6,4] — will be truncated to min(8,6)=6
        target_step0 = baseline_concat[:6] + torch.randn(6, 4) * 0.0001

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[baseline_step0, baseline_step1],
            target_steps=[target_step0],
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        # truncated to min(8,6) = 6 along concat dim
        assert comparisons[0].baseline.shape == [6, 4]
        assert comparisons[0].target.shape == [6, 4]

    def test_concat_multi_step_tp_unshard(self, tmp_path, capsys):
        """Multi-step different data + TP=2 unshard + concat."""
        torch.manual_seed(42)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        # 2 steps: [4,8] each → concat → [8,8]
        full_step0 = torch.randn(4, 8)
        full_step1 = torch.randn(4, 8)

        _create_multi_step_tp_sharded_dumps(
            baseline_dir,
            full_tensors_per_step=[full_step0, full_step1],
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        _create_multi_step_tp_sharded_dumps(
            target_dir,
            full_tensors_per_step=[
                full_step0 + torch.randn(4, 8) * 0.0001,
                full_step1 + torch.randn(4, 8) * 0.0001,
            ],
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        # 2 steps × [4, 8] concat along dim 0 (fallback) → [8, 8]
        assert comparisons[0].baseline.shape == [8, 8]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_unequal_step_counts(self, tmp_path, capsys):
        """Baseline 3 steps vs target 2 steps with truncation."""
        torch.manual_seed(42)

        # baseline: 3 steps [3]+[4]+[2] = 9 tokens along dim 0
        b_step0 = torch.randn(3, 4)
        b_step1 = torch.randn(4, 4)
        b_step2 = torch.randn(2, 4)
        b_concat = torch.cat([b_step0, b_step1, b_step2], dim=0)

        # target: 2 steps [5]+[3] = 8 tokens along dim 0
        t_step0 = b_concat[:5] + torch.randn(5, 4) * 0.0001
        t_step1 = b_concat[5:8] + torch.randn(3, 4) * 0.0001

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1, b_step2],
            target_steps=[t_step0, t_step1],
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        # truncated to min(9,8) = 8
        assert comparisons[0].baseline.shape == [8, 4]
        assert comparisons[0].target.shape == [8, 4]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_token_dim_nonzero(self, tmp_path, capsys):
        """Token dim at dim=1 (dims='b t h') — concat along dim 1."""
        torch.manual_seed(42)

        # 2 steps: [2,5,4] + [2,3,4] → concat along dim 1 → [2,8,4]
        b_step0 = torch.randn(2, 5, 4)
        b_step1 = torch.randn(2, 3, 4)
        b_concat = torch.cat([b_step0, b_step1], dim=1)

        t_step0 = b_concat[:, :5, :] + torch.randn(2, 5, 4) * 0.0001
        t_step1 = b_concat[:, 5:, :] + torch.randn(2, 3, 4) * 0.0001

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1],
            target_steps=[t_step0, t_step1],
            dims="b t h",
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].baseline.shape == [2, 8, 4]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_seq_dim_fallback(self, tmp_path, capsys):
        """No 't' dim but 's' dim present (dims='b s h') → concat along s."""
        torch.manual_seed(42)

        # 2 steps: [2,5,4] + [2,3,4] → concat along dim 1 (s) → [2,8,4]
        b_step0 = torch.randn(2, 5, 4)
        b_step1 = torch.randn(2, 3, 4)
        b_concat = torch.cat([b_step0, b_step1], dim=1)

        t_step0 = b_concat[:, :5, :] + torch.randn(2, 5, 4) * 0.0001
        t_step1 = b_concat[:, 5:, :] + torch.randn(2, 3, 4) * 0.0001

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1],
            target_steps=[t_step0, t_step1],
            dims="b s h",
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].baseline.shape == [2, 8, 4]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_no_dims_fallback(self, tmp_path, capsys):
        """No dims annotation → fallback to concat along dim 0."""
        torch.manual_seed(42)

        # 2 steps: [5,4] + [3,4] → concat along dim 0 → [8,4]
        b_step0 = torch.randn(5, 4)
        b_step1 = torch.randn(3, 4)
        b_concat = torch.cat([b_step0, b_step1], dim=0)

        t_step0 = b_concat[:5] + torch.randn(5, 4) * 0.0001
        t_step1 = b_concat[5:] + torch.randn(3, 4) * 0.0001

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1],
            target_steps=[t_step0, t_step1],
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].baseline.shape == [8, 4]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_preserves_step_order(self, tmp_path, capsys):
        """Verify step0 data precedes step1 data in the concatenated result."""
        # deterministic integer data: step0=[1,2,3], step1=[4,5]
        b_step0 = torch.tensor([[1.0], [2.0], [3.0]])
        b_step1 = torch.tensor([[4.0], [5.0]])

        # target: same data, single step [1,2,3,4,5]
        t_full = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1],
            target_steps=[t_full],
        )
        comp = _assert_single_comparison_passed(records)
        # if order were wrong, diff would not pass with exact integer data
        assert comp.baseline.shape == [5, 1]
        assert comp.diff is not None
        assert comp.diff.max_abs_diff == 0.0

    def test_concat_aux_tensors_not_filtered(self, tmp_path, capsys):
        """Concat mode does not filter aux tensors — all participate in comparison."""
        torch.manual_seed(42)

        baseline_dir, target_dir = self._make_dirs(tmp_path)

        hidden = torch.randn(4, 8)
        input_ids = torch.randint(0, 100, (4,))
        positions = torch.arange(4)

        _create_rank_dump(
            baseline_dir,
            rank=0,
            name="hidden_states",
            tensor=hidden,
            extra_dumps=[("input_ids", input_ids), ("positions", positions)],
        )
        _create_rank_dump(
            target_dir,
            rank=0,
            name="hidden_states",
            tensor=hidden + torch.randn(4, 8) * 0.0001,
            extra_dumps=[("input_ids", input_ids), ("positions", positions)],
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        # all 3 tensors should be compared (not filtered out)
        names = {c.name for c in comparisons}
        assert "hidden_states" in names
        assert "input_ids" in names
        assert "positions" in names
        assert len(comparisons) == 3

    def test_concat_aligner_plan_fields(self, tmp_path, capsys):
        """ComparisonRecord.aligner_plan reports mode='concat' with plan=None."""
        torch.manual_seed(42)

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[torch.randn(3, 4), torch.randn(2, 4)],
            target_steps=[torch.randn(3, 4), torch.randn(2, 4)],
            diff_threshold=100.0,
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        plan = comparisons[0].aligner_plan
        assert plan is not None
        assert plan.token_aligner_mode == "concat_steps"
        assert plan.token_aligner_plan is None

    def test_concat_comparison_fails(self, tmp_path, capsys):
        """Completely different data → comparison fails."""
        torch.manual_seed(42)
        b_step0 = torch.randn(4, 4)
        b_step1 = torch.randn(3, 4)

        # target: completely different random data
        torch.manual_seed(99)
        t_step0 = torch.randn(4, 4) * 100
        t_step1 = torch.randn(3, 4) * 100

        records = self._run_concat(
            tmp_path,
            capsys,
            baseline_steps=[b_step0, b_step1],
            target_steps=[t_step0, t_step1],
            diff_threshold=1e-6,
        )
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert not comparisons[0].diff.passed

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1
        assert summary.passed == 0

    def test_concat_multi_step_cp_unshard(self, tmp_path, capsys):
        """Multi-step different data + CP=2 unshard along seq dim + concat."""
        torch.manual_seed(42)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        # 2 steps: [4,8,6] each → concat along seq dim (dim 1) → [4,16,6]
        full_step0 = torch.randn(4, 8, 6)
        full_step1 = torch.randn(4, 8, 6)

        for side_dir, steps in [
            (baseline_dir, [full_step0, full_step1]),
            (
                target_dir,
                [
                    full_step0 + torch.randn(4, 8, 6) * 0.0001,
                    full_step1 + torch.randn(4, 8, 6) * 0.0001,
                ],
            ),
        ]:
            for cp_rank in range(2):
                per_step_shards: list[torch.Tensor] = [
                    t.chunk(2, dim=1)[cp_rank] for t in steps
                ]
                _create_multi_step_rank_dump(
                    side_dir,
                    rank=cp_rank,
                    name="attn_out",
                    tensors_per_step=per_step_shards,
                    dims="b s(cp) h",
                    parallel_info={"cp_rank": cp_rank, "cp_size": 2},
                )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        # CP unshard: [4,4,6] × 2 ranks → [4,8,6] per step
        # concat along seq dim (dim 1): 2 steps × [4,8,6] → [4,16,6]
        assert comparisons[0].baseline.shape == [4, 16, 6]
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_concat_thd_cp_zigzag(self, tmp_path: Path, capsys) -> None:
        """Concat mode with THD CP=2 zigzag (Megatron format) — unshard + reorder works."""
        torch.manual_seed(42)
        cp_size: int = 2
        seq_lens: list[int] = [100, 64]
        total_tokens: int = sum(seq_lens)
        total_per_rank: int = 128
        num_steps: int = 2

        full_tensor: torch.Tensor = torch.randn(total_tokens + 92)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        baseline_path: Path = _create_thd_cp_zigzag_dumps(
            baseline_dir,
            full_tensor=full_tensor,
            name="hidden_states",
            seq_lens=seq_lens,
            cp_size=cp_size,
            total_per_rank=total_per_rank,
            num_steps=num_steps,
        )

        target_tensor: torch.Tensor = full_tensor + torch.randn_like(full_tensor) * 1e-5
        target_path: Path = _create_thd_cp_zigzag_dumps(
            target_dir,
            full_tensor=target_tensor,
            name="hidden_states",
            seq_lens=seq_lens,
            cp_size=cp_size,
            total_per_rank=total_per_rank,
            num_steps=num_steps,
        )

        args: Namespace = _make_args(
            baseline_path,
            target_path,
            token_aligner="concat_steps",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        hidden_comparisons: list[ComparisonRecord] = [
            c for c in comparisons if c.name == "hidden_states"
        ]
        assert len(hidden_comparisons) >= 1
        assert all(c.diff is not None and c.diff.passed for c in hidden_comparisons)


class TestEntrypointAxisAligner:
    """Test cross-framework dim reordering through the full entrypoint pipeline."""

    def test_axis_swap_different_dim_order(self, tmp_path, capsys):
        """Baseline dims 'b h d' vs target dims 'b d h': axis swapper rearranges baseline to match."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_rank_dump(
            baseline_dir,
            rank=0,
            name="hidden",
            tensor=full_tensor,
            dims="b h d",
        )
        _create_rank_dump(
            target_dir,
            rank=0,
            name="hidden",
            tensor=full_tensor.permute(0, 2, 1).contiguous(),
            dims="b d h",
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=1e-3,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"
        assert comp.baseline.shape == [4, 16, 8]
        assert comp.target.shape == [4, 16, 8]

    def test_axis_swap_with_tp_unshard(self, tmp_path, capsys):
        """Baseline TP=2 with dims 'b h(tp) d' vs target TP=2 with dims 'b d h(tp)': unshard + axis swap."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_tensor,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp) d",
        )
        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_tensor.permute(0, 2, 1).contiguous(),
            name="hidden",
            tp_size=2,
            shard_dim=2,
            dims_str="b d h(tp)",
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=1e-3,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"

    def test_squeeze_dim_one_side(self, tmp_path, capsys):
        """SGLang dims 't h' vs Megatron dims 't 1 h': axis aligner squeezes the singleton dim."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_rank_dump(
            baseline_dir,
            rank=0,
            name="hidden",
            tensor=full_tensor,
            dims="t h",
        )
        _create_rank_dump(
            target_dir,
            rank=0,
            name="hidden",
            tensor=full_tensor.unsqueeze(1),
            dims="t 1 h",
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=1e-3,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.name == "hidden"
        assert comp.baseline.shape == [4, 8]
        assert comp.target.shape == [4, 8]


class TestEntrypointReplicatedAxis:
    """Test replicated-axis scenarios through the full entrypoint pipeline."""

    def test_replicated_axis_identical_replicas_passed(self, tmp_path, capsys):
        """CP2 TP2, TP replicated and identical → passed, replicated_checks all passed."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.0001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_replicated_tp_sharded_cp_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="attn_out",
                cp_size=2,
                tp_size=2,
                seq_dim=1,
                dims_str="b s(cp) d",
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comp = _assert_single_comparison_passed(records)
        assert comp.warnings == []
        assert all(c.passed for c in comp.replicated_checks)

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 1

    def test_replicated_mismatch_fails(self, tmp_path, capsys):
        """CP2 TP2, TP replicas differ (> atol) → failed with replicated_checks."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.0001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir, full_tensor in [
            (baseline_dir, full_baseline),
            (target_dir, full_target),
        ]:
            _create_replicated_tp_sharded_cp_dumps(
                side_dir,
                full_tensor=full_tensor,
                name="attn_out",
                cp_size=2,
                tp_size=2,
                seq_dim=1,
                dims_str="b s(cp) d",
                tp_noise=0.5,
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.01,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].category == "failed"
        assert any(not c.passed for c in comparisons[0].replicated_checks)

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1

    def test_summary_counts_failed_from_replicated_checks_only(self, tmp_path, capsys):
        """Diff itself passes but TP replicas differ → summary.failed=1 from replicated_checks."""
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8, 6)
        full_target = full_baseline + torch.randn(4, 8, 6) * 0.0001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_replicated_tp_sharded_cp_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="attn_out",
            cp_size=2,
            tp_size=2,
            seq_dim=1,
            dims_str="b s(cp) d",
            tp_noise=0.5,
        )
        _create_replicated_tp_sharded_cp_dumps(
            target_dir,
            full_tensor=full_target,
            name="attn_out",
            cp_size=2,
            tp_size=2,
            seq_dim=1,
            dims_str="b s(cp) d",
            tp_noise=0.5,
        )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            diff_threshold=0.5,
        )

        records, _ = _run_and_parse(args, capsys)
        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1

        comp = comparisons[0]
        assert comp.diff is not None
        assert comp.diff.passed
        assert any(not c.passed for c in comp.replicated_checks)
        assert comp.category == "failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1
        assert summary.passed == 0


class TestEntrypointAlignment:
    """Test `--grouping logical` with token alignment (aux tensors present)."""

    def test_sglang_multi_step_alignment(self, tmp_path, capsys):
        """SGLang multi-step dumps with aux tensors auto-trigger alignment."""
        torch.manual_seed(42)
        hidden_dim = 8

        hidden_step0 = torch.randn(5, hidden_dim)
        hidden_step1 = torch.randn(2, hidden_dim)

        exp_paths: list[Path] = []
        for side_dir in ["baseline", "target"]:
            d = tmp_path / side_dir
            d.mkdir()

            dumper = _Dumper(
                config=DumperConfig(
                    enable=True,
                    dir=str(d),
                    exp_name=_FIXED_EXP_NAME,
                )
            )

            # Step 0: prefill with 2 sequences (3+2 tokens)
            dumper.dump("input_ids", torch.tensor([10, 20, 30, 40, 50]))
            dumper.dump("positions", torch.tensor([0, 1, 2, 0, 1]))
            dumper.dump("seq_lens", torch.tensor([3, 2]))
            dumper.dump("req_pool_indices", torch.tensor([7, 3]))
            dumper.dump("rids", ["A", "B"])
            dumper.dump("hidden_states", hidden_step0)
            dumper.step()

            # Step 1: decode (1 token per sequence)
            dumper.dump("input_ids", torch.tensor([31, 51]))
            dumper.dump("positions", torch.tensor([3, 2]))
            dumper.dump("seq_lens", torch.tensor([1, 1]))
            dumper.dump("req_pool_indices", torch.tensor([7, 3]))
            dumper.dump("rids", ["A", "B"])
            dumper.dump("hidden_states", hidden_step1)
            dumper.step()

            exp_paths.append(d / _FIXED_EXP_NAME)

        args = _make_args(
            exp_paths[0], exp_paths[1], grouping="logical", token_aligner="smart"
        )
        records, _ = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        # AUX_NAMES are filtered out after plan computation → only hidden_states remains
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden_states"
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 1
        assert summary.failed == 0
        assert summary.skipped == 0

    def test_sglang_vs_megatron_cross_framework(self, tmp_path, capsys):
        """SGLang 4-step thd baseline vs Megatron 1-step thd target align correctly."""
        torch.manual_seed(42)
        hidden_dim: int = 8

        all_hiddens: torch.Tensor = torch.randn(11, hidden_dim)
        seq_a_hiddens: torch.Tensor = all_hiddens[:6]
        seq_b_hiddens: torch.Tensor = all_hiddens[6:]

        # --- SGLang baseline: 1 prefill + 3 decode ---
        sglang_dir: Path = tmp_path / "baseline"
        sglang_dir.mkdir()
        sglang_dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(sglang_dir),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        # Step 0: prefill — seq A (3 tokens) + seq B (2 tokens)
        sglang_dumper.dump("input_ids", torch.tensor([10, 20, 30, 40, 50]))
        sglang_dumper.dump("positions", torch.tensor([0, 1, 2, 0, 1]))
        sglang_dumper.dump("seq_lens", torch.tensor([3, 2]))
        sglang_dumper.dump("req_pool_indices", torch.tensor([7, 3]))
        sglang_dumper.dump("rids", ["A", "B"])
        sglang_dumper.dump(
            "hidden_states",
            torch.stack(
                [
                    seq_a_hiddens[0],
                    seq_a_hiddens[1],
                    seq_a_hiddens[2],
                    seq_b_hiddens[0],
                    seq_b_hiddens[1],
                ]
            ),
        )
        sglang_dumper.step()

        # Steps 1-3: decode — 1 token per sequence
        decode_data: list[dict[str, object]] = [
            {
                "input_ids": torch.tensor([31, 51]),
                "positions": torch.tensor([3, 2]),
                "hidden": torch.stack([seq_a_hiddens[3], seq_b_hiddens[2]]),
            },
            {
                "input_ids": torch.tensor([32, 52]),
                "positions": torch.tensor([4, 3]),
                "hidden": torch.stack([seq_a_hiddens[4], seq_b_hiddens[3]]),
            },
            {
                "input_ids": torch.tensor([33, 53]),
                "positions": torch.tensor([5, 4]),
                "hidden": torch.stack([seq_a_hiddens[5], seq_b_hiddens[4]]),
            },
        ]
        for step_data in decode_data:
            sglang_dumper.dump("input_ids", step_data["input_ids"])
            sglang_dumper.dump("positions", step_data["positions"])
            sglang_dumper.dump("seq_lens", torch.tensor([1, 1]))
            sglang_dumper.dump("req_pool_indices", torch.tensor([7, 3]))
            sglang_dumper.dump("rids", ["A", "B"])
            sglang_dumper.dump("hidden_states", step_data["hidden"])
            sglang_dumper.step()

        # --- Megatron target: 1 step, thd [T, H] ---
        megatron_dir: Path = tmp_path / "target"
        megatron_dir.mkdir()
        megatron_dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(megatron_dir),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        # THD flat: seq A (6 tokens) + seq B (5 tokens) = 11 tokens total
        megatron_input_ids: torch.Tensor = torch.tensor(
            [10, 20, 30, 31, 32, 33, 40, 50, 51, 52, 53]
        )
        megatron_cu_seqlens: torch.Tensor = torch.tensor([0, 6, 11])

        megatron_hidden: torch.Tensor = torch.cat([seq_a_hiddens, seq_b_hiddens], dim=0)

        megatron_dumper.dump("input_ids", megatron_input_ids)
        megatron_dumper.dump("cu_seqlens_q", megatron_cu_seqlens)
        megatron_dumper.dump("hidden_states", megatron_hidden)
        megatron_dumper.step()

        # --- Run comparison ---
        args = _make_args(
            sglang_dir / _FIXED_EXP_NAME,
            megatron_dir / _FIXED_EXP_NAME,
            grouping="logical",
            token_aligner="smart",
        )

        records, _ = _run_and_parse(args, capsys)

        warning_records = [r for r in records if isinstance(r, WarningRecord)]
        layout_warnings = [
            w
            for wr in warning_records
            for w in wr.warnings
            if isinstance(w, GeneralWarning)
            and w.category == "layout_detection_fallback"
        ]
        assert len(layout_warnings) == 1

        comparisons = _get_comparisons(records)
        # AUX_NAMES filtered out → only hidden_states remains
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden_states"
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 1
        assert summary.failed == 0
        assert summary.skipped == 0

    def test_alignment_fallback_when_no_aux(self, tmp_path, capsys):
        """Without aux tensors, logical grouping skips alignment and compares per-step."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"], num_steps=2)
        args = _make_args(
            baseline_path,
            target_path,
            grouping="logical",
            token_aligner="smart",
            diff_threshold=0.1,
        )

        capsys.readouterr()
        run(args)
        captured = capsys.readouterr()
        records = _parse_jsonl(captured.out)
        warning_records = [r for r in records if isinstance(r, WarningRecord)]
        aux_missing_warnings = [
            w
            for wr in warning_records
            for w in wr.warnings
            if isinstance(w, GeneralWarning) and w.category == "aux_tensors_missing"
        ]
        assert len(aux_missing_warnings) == 1

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2


class TestEntrypointNonTensorValues:
    """Test non-tensor value comparison through the full entrypoint pipeline."""

    def test_non_tensor_float_same_value(self, tmp_path: Path, capsys) -> None:
        """Two sides dump the same float → NonTensorRecord with values_equal=True, category=passed."""
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path, name="sm_scale", baseline_value=0.125, target_value=0.125
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "sm_scale"
        assert non_tensors[0].values_equal is True
        assert non_tensors[0].category == "passed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 1
        assert summary.failed == 0

    def test_non_tensor_float_different_value(self, tmp_path: Path, capsys) -> None:
        """Two sides dump different floats → NonTensorRecord with values_equal=False, category=failed."""
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path, name="sm_scale", baseline_value=0.125, target_value=0.25
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1
        assert non_tensors[0].values_equal is False
        assert non_tensors[0].category == "failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1

    def test_non_tensor_string_value(self, tmp_path: Path, capsys) -> None:
        """String non-tensor values are compared and displayed correctly."""
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path,
            name="attn_backend",
            baseline_value="flash_attn",
            target_value="flash_attn",
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1
        assert non_tensors[0].values_equal is True
        assert non_tensors[0].baseline_type == "str"
        assert non_tensors[0].target_type == "str"

    def test_non_tensor_mixed_with_tensor(self, tmp_path: Path, capsys) -> None:
        """Tensors and non_tensors in the same dump are each handled correctly."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 4)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for side_dir in [baseline_dir, target_dir]:
            _create_non_tensor_rank_dump(
                side_dir,
                rank=0,
                name="sm_scale",
                value=0.125,
                extra_tensor_dumps=[("hidden", tensor)],
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
        )
        records, _ = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        non_tensors = _get_non_tensors(records)
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden"
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "sm_scale"
        assert non_tensors[0].values_equal is True

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 2

    def test_non_tensor_complex_object(self, tmp_path: Path, capsys) -> None:
        """Complex objects (e.g. dict containing a tensor) are displayed via repr, not skipped."""
        value = {"a": 1, "b": "hello", "c": torch.tensor([1.0, 2.0])}
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path, name="debug_info", baseline_value=value, target_value=value
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "debug_info"
        assert non_tensors[0].baseline_type == "dict"
        assert non_tensors[0].target_type == "dict"

    def test_non_tensor_none_value(self, tmp_path: Path, capsys) -> None:
        """Dumping None is displayed as NonTensorRecord, not skipped as load failure."""
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path, name="optional_param", baseline_value=None, target_value=None
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "optional_param"
        assert non_tensors[0].values_equal is True
        assert non_tensors[0].baseline_value == "None"
        assert non_tensors[0].baseline_type == "NoneType"
        assert non_tensors[0].category == "passed"

    def test_non_tensor_json_roundtrip(self, tmp_path: Path, capsys) -> None:
        """NonTensorRecord JSON output can be parsed back correctly."""
        baseline_path, target_path = _create_non_tensor_dumps(
            tmp_path, name="sm_scale", baseline_value=0.125, target_value=0.125
        )
        args = _make_args(baseline_path, target_path, grouping="raw")
        records, _ = _run_and_parse(args, capsys)

        non_tensors = _get_non_tensors(records)
        assert len(non_tensors) == 1

        json_str: str = non_tensors[0].model_dump_json()
        roundtripped = parse_record_json(json_str)
        assert isinstance(roundtripped, NonTensorRecord)
        assert roundtripped.name == "sm_scale"
        assert roundtripped.values_equal is True


# ───────────────────── Visualization integration tests ─────────────────────


class TestEntrypointVisualize:
    """Test --visualize-bundle-details integration."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_matplotlib(self) -> None:
        pytest.importorskip("matplotlib")

    def test_visualize_creates_pngs(self, tmp_path, capsys):
        """--visualize-bundle-details with --filter produces PNG files."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        viz_dir = tmp_path / "viz_out"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            filter="tensor_a",
            viz_bundle_details=True,
            viz_output_dir=str(viz_dir),
        )

        records, _ = _run_and_parse(args, capsys)
        assert len(_get_comparisons(records)) == 1

        png_files = list(viz_dir.glob("*.png"))
        assert len(png_files) == 1
        assert png_files[0].stat().st_size > 0

    def test_no_visualize_no_png(self, tmp_path, capsys):
        """Without --visualize-bundle-details, no PNGs are created."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        viz_dir = tmp_path / "viz_out"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            viz_bundle_details=False,
            viz_output_dir=str(viz_dir),
        )

        _run_and_parse(args, capsys)
        assert not viz_dir.exists() or len(list(viz_dir.glob("*.png"))) == 0


# --------------------------- Assertion helpers -------------------


def _get_comparisons(records: list[AnyRecord]) -> list[ComparisonRecord]:
    return [r for r in records if isinstance(r, ComparisonRecord)]


def _get_non_tensors(records: list[AnyRecord]) -> list[NonTensorRecord]:
    return [r for r in records if isinstance(r, NonTensorRecord)]


def _assert_single_comparison_passed(records: list[AnyRecord]) -> ComparisonRecord:
    comparisons = _get_comparisons(records)
    assert len(comparisons) == 1
    assert comparisons[0].diff is not None
    assert comparisons[0].diff.passed
    return comparisons[0]


# --------------------------- Utils ------------------------------


def _make_dumper(directory: Path) -> _Dumper:
    return _Dumper(config=DumperConfig(enable=True, dir=str(directory)))


def _create_dumps(
    tmp_path: Path,
    tensor_names: list[str],
    *,
    baseline_names: list[str] | None = None,
    num_steps: int = 1,
) -> tuple[Path, Path]:
    """Create baseline and target dump directories with given tensor names.

    If baseline_names is None, uses the same names as tensor_names.
    Each step dumps all names with the same tensor (different per baseline/target).
    """
    if baseline_names is None:
        baseline_names = tensor_names

    d_baseline = tmp_path / "baseline"
    d_target = tmp_path / "target"
    d_baseline.mkdir()
    d_target.mkdir()

    torch.manual_seed(42)
    baseline_tensor = torch.randn(10, 10)
    target_tensor = baseline_tensor + torch.randn(10, 10) * 0.01

    exp_paths: list[Path] = []
    for d, names, tensor in [
        (d_baseline, baseline_names, baseline_tensor),
        (d_target, tensor_names, target_tensor),
    ]:
        dumper = _make_dumper(d)
        for _ in range(num_steps):
            for name in names:
                dumper.dump(name, tensor)
            dumper.step()
        exp_paths.append(d / dumper._config.exp_name)

    return exp_paths[0], exp_paths[1]


def _create_non_tensor_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    value: object,
    extra_tensor_dumps: list[tuple[str, torch.Tensor]] | None = None,
) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )
        dumper.__dict__["_static_meta"] = {"world_rank": rank, "world_size": 1}

        dumper.dump(name, value)
        for extra_name, extra_tensor in extra_tensor_dumps or []:
            dumper.dump(extra_name, extra_tensor)
        dumper.step()

    return directory / _FIXED_EXP_NAME


def _create_non_tensor_dumps(
    tmp_path: Path,
    *,
    name: str,
    baseline_value: object,
    target_value: object,
) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline"
    target_dir = tmp_path / "target"
    baseline_dir.mkdir()
    target_dir.mkdir()

    baseline_path = _create_non_tensor_rank_dump(
        baseline_dir, rank=0, name=name, value=baseline_value
    )
    target_path = _create_non_tensor_rank_dump(
        target_dir, rank=0, name=name, value=target_value
    )
    return baseline_path, target_path


def _make_args(baseline_path: Path, target_path: Path, **overrides) -> Namespace:
    defaults = dict(
        baseline_path=str(baseline_path),
        target_path=str(target_path),
        start_step=0,
        end_step=1000000,
        diff_threshold=1e-3,
        filter=None,
        output_format="json",
        grouping="logical",
        token_aligner="concat_steps",
        viz_bundle_details=False,
        viz_output_dir="/tmp/comparator_viz/",
        visualize_per_token=None,
        override_dims=[],
        override_baseline_dims=[],
        override_target_dims=[],
        override_config=None,
        allow_skip_pattern=".*",
        report_path="",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _run_and_parse(
    args: Namespace, capsys: pytest.CaptureFixture
) -> tuple[list[AnyRecord], int]:
    capsys.readouterr()
    exit_code: int = run(args)
    return _parse_jsonl(capsys.readouterr().out), exit_code


def _parse_jsonl(output: str) -> list[AnyRecord]:
    return [parse_record_json(line) for line in output.strip().splitlines()]


def _create_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    tensor: torch.Tensor,
    dims: str | None = None,
    parallel_info: dict | None = None,
    framework: str = "sglang",
    num_steps: int = 1,
    extra_dumps: list[tuple[str, object]] | None = None,
) -> Path:
    """Create a dump file via the real dumper, as if running on the given rank.

    extra_dumps: additional (name, value) pairs to dump alongside the main tensor each step.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        static_meta: dict = {"world_rank": rank, "world_size": 1}
        if parallel_info is not None:
            static_meta[f"{framework}_parallel_info"] = parallel_info
        dumper.__dict__["_static_meta"] = static_meta

        for _ in range(num_steps):
            dumper.dump(name, tensor, dims=dims)
            for extra_name, extra_value in extra_dumps or []:
                dumper.dump(extra_name, extra_value)
            dumper.step()

    return directory / _FIXED_EXP_NAME


def _create_multi_step_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    tensors_per_step: list[torch.Tensor],
    dims: str | None = None,
    parallel_info: dict | None = None,
    framework: str = "sglang",
) -> Path:
    """Create a dump file with *different* tensors per step.

    Unlike ``_create_rank_dump`` (which repeats the same tensor),
    this helper accepts a list of tensors — one per step.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        static_meta: dict = {"world_rank": rank, "world_size": 1}
        if parallel_info is not None:
            static_meta[f"{framework}_parallel_info"] = parallel_info
        dumper.__dict__["_static_meta"] = static_meta

        for tensor in tensors_per_step:
            dumper.dump(name, tensor, dims=dims)
            dumper.step()

    return directory / _FIXED_EXP_NAME


def _create_cp_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    cp_size: int,
    tp_size: int,
    seq_dim: int,
    head_dim: int,
    dims_str: str,
    num_steps: int = 1,
) -> Path:
    """Create CP+TP multi-axis sharded dump files from a full tensor."""
    cp_chunks = list(full_tensor.chunk(cp_size, dim=seq_dim))
    rank = 0
    for cp_rank in range(cp_size):
        tp_chunks = list(cp_chunks[cp_rank].chunk(tp_size, dim=head_dim))
        for tp_rank in range(tp_size):
            _create_rank_dump(
                directory,
                rank=rank,
                name=name,
                tensor=tp_chunks[tp_rank],
                dims=dims_str,
                parallel_info={
                    "cp_rank": cp_rank,
                    "cp_size": cp_size,
                    "tp_rank": tp_rank,
                    "tp_size": tp_size,
                },
                num_steps=num_steps,
            )
            rank += 1
    return directory / _FIXED_EXP_NAME


def _create_ep_cp_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    ep_size: int,
    cp_size: int,
    tp_size: int,
    expert_dim: int,
    seq_dim: int,
    head_dim: int,
    dims_str: str,
    num_steps: int = 1,
) -> Path:
    """Create EP+CP+TP three-axis sharded dump files from a full tensor."""
    ep_chunks = list(full_tensor.chunk(ep_size, dim=expert_dim))
    rank = 0
    for ep_rank in range(ep_size):
        cp_chunks = list(ep_chunks[ep_rank].chunk(cp_size, dim=seq_dim))
        for cp_rank in range(cp_size):
            tp_chunks = list(cp_chunks[cp_rank].chunk(tp_size, dim=head_dim))
            for tp_rank in range(tp_size):
                _create_rank_dump(
                    directory,
                    rank=rank,
                    name=name,
                    tensor=tp_chunks[tp_rank],
                    dims=dims_str,
                    parallel_info={
                        "ep_rank": ep_rank,
                        "ep_size": ep_size,
                        "cp_rank": cp_rank,
                        "cp_size": cp_size,
                        "tp_rank": tp_rank,
                        "tp_size": tp_size,
                    },
                    num_steps=num_steps,
                )
                rank += 1
    return directory / _FIXED_EXP_NAME


def _create_cp_zigzag_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    cp_size: int,
    tp_size: int,
    seq_dim: int,
    head_dim: int,
    dims_str: str,
    num_steps: int = 1,
) -> Path:
    """Create CP-zigzag (+optional TP) sharded dump files from a full tensor."""
    num_chunks: int = cp_size * 2
    natural_chunks: list[torch.Tensor] = list(
        full_tensor.chunk(num_chunks, dim=seq_dim)
    )

    zigzag_order: list[int] = []
    for i in range(cp_size):
        zigzag_order.append(i)
        zigzag_order.append(num_chunks - 1 - i)

    zigzagged: torch.Tensor = torch.cat(
        [natural_chunks[idx] for idx in zigzag_order], dim=seq_dim
    )

    cp_chunks: list[torch.Tensor] = list(zigzagged.chunk(cp_size, dim=seq_dim))

    rank: int = 0
    for cp_rank in range(cp_size):
        tp_chunks: list[torch.Tensor] = (
            list(cp_chunks[cp_rank].chunk(tp_size, dim=head_dim))
            if tp_size > 1
            else [cp_chunks[cp_rank]]
        )
        for tp_rank in range(tp_size):
            parallel_info: dict[str, int] = {
                "cp_rank": cp_rank,
                "cp_size": cp_size,
            }
            if tp_size > 1:
                parallel_info["tp_rank"] = tp_rank
                parallel_info["tp_size"] = tp_size

            _create_rank_dump(
                directory,
                rank=rank,
                name=name,
                tensor=tp_chunks[tp_rank],
                dims=dims_str,
                parallel_info=parallel_info,
                num_steps=num_steps,
            )
            rank += 1

    return directory / _FIXED_EXP_NAME


def _create_cp_zigzag_sp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    cp_size: int,
    sp_size: int,
    dims_str: str,
    seq_dim: int = 1,
    num_steps: int = 1,
) -> Path:
    """Create CP-zigzag + SP sharded dump files for a seq dim (b s h format).

    Shard order (outer to inner, matching left-to-right in dims annotation):
      1. CP zigzag splits seq dim into cp_size chunks (zigzag order)
      2. SP splits each CP chunk into sp_size chunks
    """
    num_chunks: int = cp_size * 2
    natural_chunks: list[torch.Tensor] = list(
        full_tensor.chunk(num_chunks, dim=seq_dim)
    )

    zigzag_order: list[int] = []
    for i in range(cp_size):
        zigzag_order.append(i)
        zigzag_order.append(num_chunks - 1 - i)

    zigzagged: torch.Tensor = torch.cat(
        [natural_chunks[idx] for idx in zigzag_order], dim=seq_dim
    )
    cp_chunks: list[torch.Tensor] = list(zigzagged.chunk(cp_size, dim=seq_dim))

    rank: int = 0
    for cp_rank in range(cp_size):
        sp_chunks: list[torch.Tensor] = list(
            cp_chunks[cp_rank].chunk(sp_size, dim=seq_dim)
        )
        for sp_rank in range(sp_size):
            _create_rank_dump(
                directory,
                rank=rank,
                name=name,
                tensor=sp_chunks[sp_rank],
                dims=dims_str,
                parallel_info={
                    "cp_rank": cp_rank,
                    "cp_size": cp_size,
                    "sp_rank": sp_rank,
                    "sp_size": sp_size,
                },
                num_steps=num_steps,
            )
            rank += 1

    return directory / _FIXED_EXP_NAME


def _create_replicated_tp_sharded_cp_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    cp_size: int,
    tp_size: int,
    seq_dim: int,
    dims_str: str,
    tp_noise: float = 0.0,
) -> Path:
    """Create CP-sharded + TP-replicated dump files from a full tensor.

    CP direction: chunks along seq_dim (sharded).
    TP direction: clones (replicated), with optional noise to simulate mismatch.
    """
    cp_chunks: list[torch.Tensor] = list(full_tensor.chunk(cp_size, dim=seq_dim))

    rank: int = 0
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            shard = cp_chunks[cp_rank].clone()
            if tp_noise > 0 and tp_rank > 0:
                shard = shard + torch.randn_like(shard) * tp_noise

            _create_rank_dump(
                directory,
                rank=rank,
                name=name,
                tensor=shard,
                dims=dims_str,
                parallel_info={
                    "cp_rank": cp_rank,
                    "cp_size": cp_size,
                    "tp_rank": tp_rank,
                    "tp_size": tp_size,
                },
            )
            rank += 1

    return directory / _FIXED_EXP_NAME


def _create_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    tp_size: int,
    shard_dim: int,
    dims_str: str,
    num_steps: int = 1,
) -> Path:
    """Create TP-sharded dump files from a full tensor via the real dumper."""
    shards = list(full_tensor.chunk(tp_size, dim=shard_dim))
    for tp_rank in range(tp_size):
        _create_rank_dump(
            directory,
            rank=tp_rank,
            name=name,
            tensor=shards[tp_rank],
            dims=dims_str,
            parallel_info={"tp_rank": tp_rank, "tp_size": tp_size},
            num_steps=num_steps,
        )
    return directory / _FIXED_EXP_NAME


def _create_multi_step_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensors_per_step: list[torch.Tensor],
    name: str,
    tp_size: int,
    shard_dim: int,
    dims_str: str,
) -> Path:
    """Create TP-sharded dump files with *different* tensors per step.

    Each step's full tensor is chunked across TP ranks, then
    ``_create_multi_step_rank_dump`` writes one file per rank.
    """
    shards_per_rank: list[list[torch.Tensor]] = [[] for _ in range(tp_size)]
    for full_tensor in full_tensors_per_step:
        shards = list(full_tensor.chunk(tp_size, dim=shard_dim))
        for tp_rank in range(tp_size):
            shards_per_rank[tp_rank].append(shards[tp_rank])

    for tp_rank in range(tp_size):
        _create_multi_step_rank_dump(
            directory,
            rank=tp_rank,
            name=name,
            tensors_per_step=shards_per_rank[tp_rank],
            dims=dims_str,
            parallel_info={"tp_rank": tp_rank, "tp_size": tp_size},
        )
    return directory / _FIXED_EXP_NAME


def _create_tp_partial_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    tp_size: int,
    dims_str: str,
    num_steps: int = 1,
) -> Path:
    """Create TP-partial dump files where each rank holds full_tensor / tp_size.

    Each rank stores an equal fraction of the full tensor so that
    element-wise summation across ranks reconstructs the original.
    """
    for tp_rank in range(tp_size):
        _create_rank_dump(
            directory,
            rank=tp_rank,
            name=name,
            tensor=full_tensor / tp_size,
            dims=dims_str,
            parallel_info={"tp_rank": tp_rank, "tp_size": tp_size},
            num_steps=num_steps,
        )
    return directory / _FIXED_EXP_NAME


def _create_recompute_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    original_tensor: torch.Tensor,
    recompute_tensor: torch.Tensor,
    dims: str = "h d",
) -> Path:
    """Create a dump with both original and recompute forward passes via monkeypatched dumper.

    The dumper naturally produces recompute_pseudo_rank=0 for original and =1 for recompute,
    plus recompute_pseudo_size=2.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )
        dumper.__dict__["_static_meta"] = {"world_rank": rank, "world_size": 1}

        # dump original forward
        mp.setattr(
            _dumper_module,
            "_detect_recompute_status",
            lambda: _RecomputeStatus.ORIGINAL,
        )
        dumper.dump(name, original_tensor, dims=dims)

        # dump recompute forward
        mp.setattr(
            _dumper_module,
            "_detect_recompute_status",
            lambda: _RecomputeStatus.RECOMPUTE,
        )
        dumper.dump(name, recompute_tensor, dims=dims)

        dumper.step()

    return directory / _FIXED_EXP_NAME


def _zigzag_split_seq(seq_natural: torch.Tensor, *, cp_size: int) -> list[torch.Tensor]:
    """Split a natural-order seq into per-rank zigzag segments."""
    num_chunks: int = cp_size * 2
    chunks: list[torch.Tensor] = list(seq_natural.chunk(num_chunks, dim=0))
    order: list[int] = []
    for i in range(cp_size):
        order.append(i)
        order.append(num_chunks - 1 - i)
    zigzagged: torch.Tensor = torch.cat([chunks[i] for i in order], dim=0)
    return list(zigzagged.chunk(cp_size, dim=0))


def _create_thd_cp_zigzag_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    seq_lens: list[int],
    cp_size: int,
    total_per_rank: int,
    dims_str: str = "t(cp:zigzag)",
    num_steps: int = 1,
) -> Path:
    """Create THD CP-zigzag sharded dump files simulating Megatron forward.

    Args:
        full_tensor: 1D tensor of shape [T] in natural order.
        seq_lens: per-seq token counts in natural order (e.g. [100, 64]).
        cp_size: context parallelism size.
        total_per_rank: total tokens per rank (including padding).
        dims_str: dims annotation for the main tensor.
    """
    # Build per-rank tensors from natural-order full_tensor
    offset: int = 0
    rank_segments: list[list[torch.Tensor]] = [[] for _ in range(cp_size)]

    for seq_len in seq_lens:
        seq_natural: torch.Tensor = full_tensor[offset : offset + seq_len]
        seq_ranks: list[torch.Tensor] = _zigzag_split_seq(seq_natural, cp_size=cp_size)
        for rank_idx in range(cp_size):
            rank_segments[rank_idx].append(seq_ranks[rank_idx])
        offset += seq_len

    # Build cu_seqlens from seq_lens (global, replicated across ranks)
    cu_seqlens_values: list[int] = [0]
    for slen in seq_lens:
        cu_seqlens_values.append(cu_seqlens_values[-1] + slen)

    # Pad to total_per_rank per rank (global pad = last cu_seqlens entry to total_per_rank * cp_size)
    total_global: int = total_per_rank * cp_size
    if cu_seqlens_values[-1] < total_global:
        pad_global: int = total_global - cu_seqlens_values[-1]
        cu_seqlens_values.append(total_global)
        pad_per_rank: int = pad_global // cp_size
        for rank_idx in range(cp_size):
            rank_segments[rank_idx].append(torch.zeros(pad_per_rank))

    cu_seqlens_q: torch.Tensor = torch.tensor(cu_seqlens_values, dtype=torch.int64)

    # Dump each rank
    for cp_rank in range(cp_size):
        rank_tensor: torch.Tensor = torch.cat(rank_segments[cp_rank], dim=0)
        assert (
            rank_tensor.shape[0] == total_per_rank
        ), f"rank {cp_rank}: expected {total_per_rank} tokens, got {rank_tensor.shape[0]}"

        _create_rank_dump(
            directory,
            rank=cp_rank,
            name=name,
            tensor=rank_tensor,
            dims=dims_str,
            parallel_info={
                "cp_rank": cp_rank,
                "cp_size": cp_size,
            },
            framework="megatron",
            num_steps=num_steps,
            extra_dumps=[
                ("cu_seqlens_q", cu_seqlens_q),
                ("input_ids", rank_tensor.to(torch.int64)),
            ],
        )

    return directory / _FIXED_EXP_NAME


class TestEntrypointPerTokenVisualization:
    """Test --visualize-per-token CLI flag integration."""

    def test_visualize_per_token_creates_png(self, tmp_path: Path, capsys) -> None:
        """--visualize-per-token with dims metadata produces per-token data in records."""
        pytest.importorskip("matplotlib")

        torch.manual_seed(42)
        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        baseline_tensor: torch.Tensor = torch.randn(10, 10)
        target_tensor: torch.Tensor = baseline_tensor + torch.randn(10, 10) * 0.01

        for name in ["tensor_a", "tensor_b"]:
            _create_rank_dump(
                baseline_dir,
                rank=0,
                name=name,
                tensor=baseline_tensor,
                dims="t h",
            )
            _create_rank_dump(
                target_dir,
                rank=0,
                name=name,
                tensor=target_tensor,
                dims="t h",
            )

        baseline_path: Path = baseline_dir / _FIXED_EXP_NAME
        target_path: Path = target_dir / _FIXED_EXP_NAME

        output_png: Path = tmp_path / "per_token.png"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            visualize_per_token=str(output_png),
        )
        records, _ = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2

        # per_token_rel_diff should be populated
        for comp in comparisons:
            assert comp.diff is not None
            assert comp.diff.per_token_rel_diff is not None
            assert isinstance(comp.diff.per_token_rel_diff, list)
            assert len(comp.diff.per_token_rel_diff) == 10

    def test_no_visualize_no_per_token(self, tmp_path: Path, capsys) -> None:
        """Without --visualize-per-token, per_token_rel_diff is None."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, _ = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.per_token_rel_diff is None


class TestEntrypointThdCpZigzag:
    """E2E entrypoint tests for THD CP zigzag format.

    Tests the full pipeline: dump creation → metadata loading → aligner plan →
    unshard + reorder → tensor comparison.
    """

    def test_sglang_vs_megatron_zigzag_cp(self, tmp_path: Path, capsys) -> None:
        """SGLang single-rank THD baseline vs Megatron CP=2 zigzag target."""
        torch.manual_seed(42)
        hidden_dim: int = 8
        cp_size: int = 2

        # Two sequences: 8 and 4 tokens (divisible by cp_size*2=4 for clean zigzag)
        seq_a_ids: list[int] = [10, 20, 30, 40, 50, 60, 70, 80]
        seq_b_ids: list[int] = [100, 200, 300, 400]
        all_ids: list[int] = seq_a_ids + seq_b_ids
        total_tokens: int = len(all_ids)
        seq_lens: list[int] = [len(seq_a_ids), len(seq_b_ids)]

        hidden_states: torch.Tensor = torch.randn(total_tokens, hidden_dim)

        # --- SGLang baseline: single rank, 1 step ---
        sglang_dir: Path = tmp_path / "baseline"
        sglang_dir.mkdir()
        sglang_dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(sglang_dir),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        positions: list[int] = list(range(seq_lens[0])) + list(range(seq_lens[1]))
        sglang_dumper.dump("input_ids", torch.tensor(all_ids))
        sglang_dumper.dump("positions", torch.tensor(positions))
        sglang_dumper.dump("seq_lens", torch.tensor(seq_lens))
        sglang_dumper.dump("rids", ["A", "B"])
        sglang_dumper.dump("hidden_states", hidden_states)
        sglang_dumper.step()

        # --- Megatron target: CP=2, zigzag, 1 step ---
        megatron_dir: Path = tmp_path / "target"
        megatron_dir.mkdir()

        # Zigzag-split input_ids and hidden_states per sequence, then concat
        ids_tensor: torch.Tensor = torch.tensor(all_ids, dtype=torch.int64)
        offset: int = 0
        rank_id_segments: list[list[torch.Tensor]] = [[] for _ in range(cp_size)]
        rank_hidden_segments: list[list[torch.Tensor]] = [[] for _ in range(cp_size)]
        for slen in seq_lens:
            seq_ids: torch.Tensor = ids_tensor[offset : offset + slen]
            seq_hidden: torch.Tensor = hidden_states[offset : offset + slen]
            zigzag_ids: list[torch.Tensor] = _zigzag_split_seq(seq_ids, cp_size=cp_size)
            zigzag_hidden: list[torch.Tensor] = _zigzag_split_seq(
                seq_hidden, cp_size=cp_size
            )
            for rank_idx in range(cp_size):
                rank_id_segments[rank_idx].append(zigzag_ids[rank_idx])
                rank_hidden_segments[rank_idx].append(zigzag_hidden[rank_idx])
            offset += slen

        cu_seqlens_q: torch.Tensor = torch.tensor(
            [0] + [sum(seq_lens[: i + 1]) for i in range(len(seq_lens))],
            dtype=torch.int64,
        )

        for cp_rank in range(cp_size):
            rank_ids: torch.Tensor = torch.cat(rank_id_segments[cp_rank])
            rank_hidden: torch.Tensor = torch.cat(rank_hidden_segments[cp_rank])
            _create_rank_dump(
                megatron_dir,
                rank=cp_rank,
                name="hidden_states",
                tensor=rank_hidden,
                dims="t(cp:zigzag) h",
                parallel_info={"cp_rank": cp_rank, "cp_size": cp_size},
                framework="megatron",
                extra_dumps=[
                    ("cu_seqlens_q", cu_seqlens_q),
                    ("input_ids", rank_ids),
                ],
            )

        # --- Run comparison ---
        args: Namespace = _make_args(
            sglang_dir / _FIXED_EXP_NAME,
            megatron_dir / _FIXED_EXP_NAME,
            grouping="logical",
            token_aligner="smart",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        hidden_comparisons: list[ComparisonRecord] = [
            c for c in comparisons if c.name == "hidden_states"
        ]
        assert len(hidden_comparisons) >= 1
        assert all(c.diff is not None and c.diff.passed for c in hidden_comparisons)

    def test_thd_cp_zigzag_unshard(self, tmp_path: Path, capsys) -> None:
        """Both sides THD CP=2 zigzag, comparison should pass."""
        torch.manual_seed(42)
        cp_size: int = 2
        seq_lens: list[int] = [100, 64]
        total_tokens: int = sum(seq_lens)
        total_per_rank: int = 128

        full_tensor: torch.Tensor = torch.randn(total_tokens + 92)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        baseline_path: Path = _create_thd_cp_zigzag_dumps(
            baseline_dir,
            full_tensor=full_tensor,
            name="hidden_states",
            seq_lens=seq_lens,
            cp_size=cp_size,
            total_per_rank=total_per_rank,
        )

        # Target: same data with small noise
        target_tensor: torch.Tensor = full_tensor + torch.randn_like(full_tensor) * 1e-5
        target_path: Path = _create_thd_cp_zigzag_dumps(
            target_dir,
            full_tensor=target_tensor,
            name="hidden_states",
            seq_lens=seq_lens,
            cp_size=cp_size,
            total_per_rank=total_per_rank,
        )

        args: Namespace = _make_args(
            baseline_path,
            target_path,
            grouping="logical",
            token_aligner="smart",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        # hidden_states should pass comparison (after unshard + reorder)
        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        hidden_comparisons: list[ComparisonRecord] = [
            c for c in comparisons if c.name == "hidden_states"
        ]
        assert len(hidden_comparisons) >= 1
        assert all(c.diff is not None and c.diff.passed for c in hidden_comparisons)


class TestEntrypointDpFilter:
    """E2E tests for DP (data parallel) filtering.

    When DP > 1, only one dp_rank has non-empty tensors; the others
    dump empty (numel=0) tensors. The comparator should filter out the
    empty dp_rank items and produce correct comparison results.
    """

    def test_dp2_sglang_both_sides(self, tmp_path: Path, capsys) -> None:
        """DP=2 sglang: both baseline and target have 1 non-empty + 1 empty dp_rank."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side, side_dir_name, data in [
            ("baseline", "baseline", tensor_data),
            ("target", "target", target_data),
        ]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            # dp_rank=0: non-empty tensor
            _create_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                tensor=data,
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 2,
                },
                framework="sglang",
            )

            # dp_rank=1: empty tensor
            _create_rank_dump(
                side_dir,
                rank=1,
                name="hidden",
                tensor=torch.empty(0, 8),
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 1,
                    "dp_size": 2,
                },
                framework="sglang",
            )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"

    def test_dp2_megatron_both_sides(self, tmp_path: Path, capsys) -> None:
        """DP=2 megatron: both baseline and target have 1 non-empty + 1 empty dp_rank."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side, side_dir_name, data in [
            ("baseline", "baseline", tensor_data),
            ("target", "target", target_data),
        ]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            # dp_rank=0: non-empty tensor
            _create_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                tensor=data,
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 2,
                },
                framework="megatron",
            )

            # dp_rank=1: empty tensor
            _create_rank_dump(
                side_dir,
                rank=1,
                name="hidden",
                tensor=torch.empty(0, 8),
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 1,
                    "dp_size": 2,
                },
                framework="megatron",
            )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"

    def test_dp2_tp2_sglang(self, tmp_path: Path, capsys) -> None:
        """DP=2 x TP=2 sglang: 4 ranks, dp_rank=0 has data, dp_rank=1 empty."""
        torch.manual_seed(42)
        full_tensor: torch.Tensor = torch.randn(10, 8)
        tp_chunks: list[torch.Tensor] = list(full_tensor.chunk(2, dim=1))

        target_full: torch.Tensor = full_tensor + torch.randn(10, 8) * 0.001
        target_tp_chunks: list[torch.Tensor] = list(target_full.chunk(2, dim=1))

        for side, side_dir_name, chunks in [
            ("baseline", "baseline", tp_chunks),
            ("target", "target", target_tp_chunks),
        ]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            rank: int = 0
            for dp_rank in range(2):
                for tp_rank in range(2):
                    tensor: torch.Tensor = (
                        chunks[tp_rank] if dp_rank == 0 else torch.empty(0, 4)
                    )
                    _create_rank_dump(
                        side_dir,
                        rank=rank,
                        name="hidden",
                        tensor=tensor,
                        dims="t h(tp)",
                        parallel_info={
                            "tp_rank": tp_rank,
                            "tp_size": 2,
                            "dp_rank": dp_rank,
                            "dp_size": 2,
                        },
                        framework="sglang",
                    )
                    rank += 1

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"

    def test_dp2_both_nonempty_raises(self, tmp_path: Path, capsys) -> None:
        """DP=2 sglang: both dp_rank=0 and dp_rank=1 have non-empty tensors => AssertionError."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side, side_dir_name, data in [
            ("baseline", "baseline", tensor_data),
            ("target", "target", target_data),
        ]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            for dp_rank in range(2):
                _create_rank_dump(
                    side_dir,
                    rank=dp_rank,
                    name="hidden",
                    tensor=data,
                    dims="t h",
                    parallel_info={
                        "tp_rank": 0,
                        "tp_size": 1,
                        "dp_rank": dp_rank,
                        "dp_size": 2,
                    },
                    framework="sglang",
                )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )

        with pytest.raises(
            AssertionError, match="Expected exactly 1 non-empty dp_rank"
        ):
            _run_and_parse(args, capsys)


class TestEntrypointDpGroupAlias:
    """E2E tests for the ``# dp:=<group>`` dp group alias feature.

    In dp_attn mode, dp_size > 1 but MLP tensors after dp_gather have data
    on all ranks.  With ``# dp:=moe_dp`` in dims, the dp filter uses
    ``moe_dp_rank/moe_dp_size`` instead of ``dp_rank/dp_size``.
    """

    def test_dp_alias_absent_group_noop(self, tmp_path: Path, capsys) -> None:
        """Single rank with ``# dp:=moe_dp`` in dims → parse_dims strips ``#``, comparison OK."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side_dir_name, data in [("baseline", tensor_data), ("target", target_data)]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            _create_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                tensor=data,
                dims="t h # dp:=moe_dp",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 1,
                },
                framework="sglang",
            )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"

    def test_dp_alias_via_override_dims(self, tmp_path: Path, capsys) -> None:
        """--override-dims adds ``# dp:=moe_dp`` → dp filter uses alias, filters correctly."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side_dir_name, data in [("baseline", tensor_data), ("target", target_data)]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            # moe_dp_rank=0: non-empty
            _create_rank_dump(
                side_dir,
                rank=0,
                name="hidden",
                tensor=data,
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 1,
                    "moe_dp_rank": 0,
                    "moe_dp_size": 2,
                },
                framework="sglang",
            )

            # moe_dp_rank=1: empty
            _create_rank_dump(
                side_dir,
                rank=1,
                name="hidden",
                tensor=torch.empty(0, 8),
                dims="t h",
                parallel_info={
                    "tp_rank": 0,
                    "tp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 1,
                    "moe_dp_rank": 1,
                    "moe_dp_size": 2,
                },
                framework="sglang",
            )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
            override_dims=["hidden:t h # dp:=moe_dp"],
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"

    def test_dp_alias_with_real_alias_group_filters(
        self, tmp_path: Path, capsys
    ) -> None:
        """Alias group present with moe_dp_size=2, one empty rank → filters correctly."""
        torch.manual_seed(42)
        tensor_data: torch.Tensor = torch.randn(10, 8)
        target_data: torch.Tensor = tensor_data + torch.randn(10, 8) * 0.001

        for side_dir_name, data in [("baseline", tensor_data), ("target", target_data)]:
            side_dir: Path = tmp_path / side_dir_name
            side_dir.mkdir()

            for moe_dp_rank in range(2):
                tensor: torch.Tensor = data if moe_dp_rank == 0 else torch.empty(0, 8)
                _create_rank_dump(
                    side_dir,
                    rank=moe_dp_rank,
                    name="hidden",
                    tensor=tensor,
                    dims="t h # dp:=moe_dp",
                    parallel_info={
                        "tp_rank": 0,
                        "tp_size": 1,
                        "dp_rank": 0,
                        "dp_size": 1,
                        "moe_dp_rank": moe_dp_rank,
                        "moe_dp_size": 2,
                    },
                    framework="sglang",
                )

        args: Namespace = _make_args(
            tmp_path / "baseline" / _FIXED_EXP_NAME,
            tmp_path / "target" / _FIXED_EXP_NAME,
            grouping="logical",
            diff_threshold=1e-3,
        )
        records, _ = _run_and_parse(args, capsys)

        comparison: ComparisonRecord = _assert_single_comparison_passed(records)
        assert comparison.name == "hidden"


class TestEntrypointMetaOverride:
    """E2E: dump with wrong dims → --override-dims / --override-config corrects at comparison time."""

    @staticmethod
    def _create_single_rank_pair(
        tmp_path: Path,
        *,
        name: str = "hidden",
        baseline_dims: str | None = "x y",
        target_dims: str | None = "x y",
    ) -> tuple[Path, Path]:
        """Create single-rank baseline+target dumps with a close tensor pair."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name=name, tensor=tensor, dims=baseline_dims
        )
        _create_rank_dump(
            target_dir, rank=0, name=name, tensor=target, dims=target_dims
        )

        return baseline_dir / _FIXED_EXP_NAME, target_dir / _FIXED_EXP_NAME

    @staticmethod
    def _assert_all_passed(
        records: list[AnyRecord], *, expected_count: int = 1
    ) -> None:
        """Assert that exactly expected_count comparisons exist and all passed."""
        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        assert len(comparisons) == expected_count
        assert all(c.diff is not None and c.diff.passed for c in comparisons)

    def test_override_dims_fixes_wrong_dims(self, tmp_path: Path, capsys) -> None:
        """Tensor dumped with wrong dims='h d' is fixed by --override-dims to 't h(tp)'."""
        torch.manual_seed(42)

        full_tensor: torch.Tensor = torch.randn(10, 8)
        tp_chunks: list[torch.Tensor] = list(full_tensor.chunk(2, dim=1))

        target_full: torch.Tensor = full_tensor + torch.randn(10, 8) * 0.001
        target_tp_chunks: list[torch.Tensor] = list(target_full.chunk(2, dim=1))

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        # Dump with WRONG dims "h d" instead of correct "t h(tp)"
        for tp_rank in range(2):
            _create_rank_dump(
                baseline_dir,
                rank=tp_rank,
                name="hidden",
                tensor=tp_chunks[tp_rank],
                dims="h d",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )
            _create_rank_dump(
                target_dir,
                rank=tp_rank,
                name="hidden",
                tensor=target_tp_chunks[tp_rank],
                dims="h d",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="logical",
            override_dims=["hidden:t h(tp)"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    @pytest.mark.parametrize(
        "baseline_dims, target_dims, override_kwarg",
        [
            ("x y", "t h", {"override_baseline_dims": ["hidden:t h"]}),
            ("t h", "x y", {"override_target_dims": ["hidden:t h"]}),
            ("x y", "x y", {"override_dims": ["hidden:t h"]}),
        ],
        ids=["baseline_only", "target_only", "both_via_override_dims"],
    )
    def test_single_side_override(
        self,
        tmp_path: Path,
        capsys,
        baseline_dims: str,
        target_dims: str,
        override_kwarg: dict,
    ) -> None:
        """Per-side override fixes the wrong dims on one or both sides."""
        baseline_path, target_path = self._create_single_rank_pair(
            tmp_path,
            baseline_dims=baseline_dims,
            target_dims=target_dims,
        )

        args = _make_args(baseline_path, target_path, grouping="raw", **override_kwarg)
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_override_config_yaml(self, tmp_path: Path, capsys) -> None:
        """--override-config YAML overrides dims."""
        baseline_path, target_path = self._create_single_rank_pair(tmp_path)

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            overrides:
              - match: "hidden"
                dims: "t h"
        """))

        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            override_config=str(yaml_path),
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_no_match_uses_original_dims(self, tmp_path: Path, capsys) -> None:
        """When override regex doesn't match, original dims from dump are used."""
        baseline_path, target_path = self._create_single_rank_pair(
            tmp_path,
            baseline_dims="t h",
            target_dims="t h",
        )

        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            override_dims=["no_match_pattern:b s d"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_selective_match_multi_tensor(self, tmp_path: Path, capsys) -> None:
        """Override matches only 'logits'; 'hidden' uses original dims."""
        torch.manual_seed(42)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        hidden_b: torch.Tensor = torch.randn(10, 8)
        hidden_t: torch.Tensor = hidden_b + torch.randn(10, 8) * 0.001
        logits_b: torch.Tensor = torch.randn(10, 4)
        logits_t: torch.Tensor = logits_b + torch.randn(10, 4) * 0.001

        for name, b_tensor, t_tensor, dims in [
            ("hidden", hidden_b, hidden_t, "t h"),
            ("logits", logits_b, logits_t, "x y"),
        ]:
            _create_rank_dump(
                baseline_dir, rank=0, name=name, tensor=b_tensor, dims=dims
            )
            _create_rank_dump(target_dir, rank=0, name=name, tensor=t_tensor, dims=dims)

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["logits:t v"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0], expected_count=2)

    def test_multiple_cli_override_dims(self, tmp_path: Path, capsys) -> None:
        """Multiple --override-dims for different tensors."""
        torch.manual_seed(42)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        hidden_b: torch.Tensor = torch.randn(10, 8)
        hidden_t: torch.Tensor = hidden_b + torch.randn(10, 8) * 0.001
        logits_b: torch.Tensor = torch.randn(10, 4)
        logits_t: torch.Tensor = logits_b + torch.randn(10, 4) * 0.001

        for name, b_tensor, t_tensor in [
            ("hidden", hidden_b, hidden_t),
            ("logits", logits_b, logits_t),
        ]:
            _create_rank_dump(
                baseline_dir, rank=0, name=name, tensor=b_tensor, dims="x y"
            )
            _create_rank_dump(
                target_dir, rank=0, name=name, tensor=t_tensor, dims="x y"
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:t h", "logits:t v"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0], expected_count=2)

    def test_per_side_dims_different_parallelism(self, tmp_path: Path, capsys) -> None:
        """baseline TP-sharded, target EP-sharded — per-side override fixes both."""
        torch.manual_seed(42)
        full_tensor: torch.Tensor = torch.randn(10, 8)
        target_full: torch.Tensor = full_tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        b_chunks: list[torch.Tensor] = list(full_tensor.chunk(2, dim=1))
        for tp_rank in range(2):
            _create_rank_dump(
                baseline_dir,
                rank=tp_rank,
                name="hidden",
                tensor=b_chunks[tp_rank],
                dims="x y",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )

        t_chunks: list[torch.Tensor] = list(target_full.chunk(2, dim=1))
        for ep_rank in range(2):
            _create_rank_dump(
                target_dir,
                rank=ep_rank,
                name="hidden",
                tensor=t_chunks[ep_rank],
                dims="x y",
                parallel_info={"ep_rank": ep_rank, "ep_size": 2},
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="logical",
            override_baseline_dims=["hidden:t h(tp)"],
            override_target_dims=["hidden:t h(ep)"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_yaml_first_match_wins_e2e(self, tmp_path: Path, capsys) -> None:
        """YAML with two matching rules: first rule wins in real pipeline."""
        baseline_path, target_path = self._create_single_rank_pair(tmp_path)

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            overrides:
              - match: "hidden"
                dims: "t h"
              - match: "hidden"
                dims: "a b"
        """))

        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            override_config=str(yaml_path),
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_cli_overrides_yaml_e2e(self, tmp_path: Path, capsys) -> None:
        """CLI --override-dims wins over YAML rule for the same tensor."""
        baseline_path, target_path = self._create_single_rank_pair(tmp_path)

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            overrides:
              - match: "hidden"
                dims: "a b"
        """))

        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            override_dims=["hidden:t h"],
            override_config=str(yaml_path),
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_override_injects_dims_when_absent(self, tmp_path: Path, capsys) -> None:
        """Override injects dims into meta even when dump had no dims annotation."""
        baseline_path, target_path = self._create_single_rank_pair(
            tmp_path,
            baseline_dims=None,
            target_dims=None,
        )

        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            override_dims=["hidden:t h"],
        )
        self._assert_all_passed(_run_and_parse(args, capsys)[0])

    def test_non_tensor_unaffected_by_override(self, tmp_path: Path, capsys) -> None:
        """Non-tensor values pass through without error even with active override."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 4)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        for side_dir in [baseline_dir, target_dir]:
            _create_non_tensor_rank_dump(
                side_dir,
                rank=0,
                name="sm_scale",
                value=0.125,
                extra_tensor_dumps=[("hidden", tensor)],
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:x y"],
        )
        records, _ = _run_and_parse(args, capsys)

        non_tensors: list[NonTensorRecord] = [
            r for r in records if isinstance(r, NonTensorRecord)
        ]
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "sm_scale"
        assert non_tensors[0].values_equal

        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden"

        summary: SummaryRecord = [r for r in records if isinstance(r, SummaryRecord)][0]
        assert summary.failed == 0


class TestExitCode:
    """Tests for exit code behavior based on comparison results."""

    def test_all_passed(self):
        """All passed → exit 0."""
        summary = SummaryRecord(total=3, passed=3, failed=0, skipped=0)
        assert (
            _compute_exit_code(summary, allow_skip_pattern=".*", skipped_names=[]) == 0
        )

    def test_has_failed_and_passed(self):
        """Has failed and passed → exit 1."""
        summary = SummaryRecord(total=4, passed=2, failed=2, skipped=0)
        assert (
            _compute_exit_code(summary, allow_skip_pattern=".*", skipped_names=[]) == 1
        )

    def test_all_failed(self):
        """All failed (0 passed) → exit 1."""
        summary = SummaryRecord(total=3, passed=0, failed=3, skipped=0)
        assert (
            _compute_exit_code(summary, allow_skip_pattern=".*", skipped_names=[]) == 1
        )

    def test_all_skipped_allow_all(self):
        """All skipped + allow_skip_pattern='.*' → exit 0."""
        summary = SummaryRecord(total=2, passed=0, failed=0, skipped=2)
        assert (
            _compute_exit_code(
                summary, allow_skip_pattern=".*", skipped_names=["a", "b"]
            )
            == 0
        )

    def test_all_skipped_forbid_all(self):
        """All skipped + allow_skip_pattern='^$' → exit 1."""
        summary = SummaryRecord(total=2, passed=0, failed=0, skipped=2)
        assert (
            _compute_exit_code(
                summary, allow_skip_pattern="^$", skipped_names=["a", "b"]
            )
            == 1
        )

    def test_passed_and_skipped_allow_all(self):
        """Passed + skipped, allow all → exit 0."""
        summary = SummaryRecord(total=3, passed=2, failed=0, skipped=1)
        assert (
            _compute_exit_code(summary, allow_skip_pattern=".*", skipped_names=["a"])
            == 0
        )

    def test_passed_and_skipped_forbid_all(self):
        """Passed + skipped + forbid all → exit 1."""
        summary = SummaryRecord(total=3, passed=2, failed=0, skipped=1)
        assert (
            _compute_exit_code(summary, allow_skip_pattern="^$", skipped_names=["a"])
            == 1
        )

    def test_skip_pattern_matches_specific_name(self):
        """Pattern matching specific name allows that skip, forbids others."""
        summary = SummaryRecord(total=4, passed=2, failed=0, skipped=2)
        assert (
            _compute_exit_code(
                summary,
                allow_skip_pattern="positions|seq_lens",
                skipped_names=["positions", "seq_lens"],
            )
            == 0
        )

    def test_skip_pattern_partial_match_forbidden(self):
        """Pattern matches some skips but not all → exit 1."""
        summary = SummaryRecord(total=4, passed=1, failed=0, skipped=3)
        assert (
            _compute_exit_code(
                summary,
                allow_skip_pattern="positions|seq_lens",
                skipped_names=["positions", "seq_lens", "hidden_states"],
            )
            == 1
        )

    def test_e2e_all_passed_exit_zero(self, tmp_path, capsys):
        """Integration: all comparisons pass → run() returns 0."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, grouping="raw")

        records, exit_code = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.passed == 2
        assert summary.failed == 0
        assert exit_code == 0

    def test_e2e_has_failed_exit_nonzero(self, tmp_path, capsys):
        """Integration: a failed comparison → run() returns 1."""
        torch.manual_seed(42)
        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="tensor_a", tensor=torch.randn(10, 10)
        )
        target_path = _create_rank_dump(
            tmp_path / "target",
            rank=0,
            name="tensor_a",
            tensor=torch.randn(10, 10) * 100,
        )
        args = _make_args(
            baseline_path, target_path, grouping="raw", diff_threshold=1e-3
        )

        records, exit_code = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.failed == 1
        assert exit_code == 1


class TestExitCodeSubprocess:
    """E2E subprocess tests: invoke comparator as a child process and verify exit code."""

    @staticmethod
    def _run_comparator(
        baseline_path: Path,
        target_path: Path,
        *,
        grouping: str = "raw",
        allow_skip_pattern: str = ".*",
    ) -> subprocess.CompletedProcess[str]:
        cmd: list[str] = [
            sys.executable,
            "-m",
            "sglang.srt.debug_utils.comparator",
            "--baseline-path",
            str(baseline_path),
            "--target-path",
            str(target_path),
            "--grouping",
            grouping,
            "--output-format",
            "json",
            "--allow-skip-pattern",
            allow_skip_pattern,
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_all_passed_exit_zero(self, tmp_path):
        """Subprocess: all comparisons pass → exit 0."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        result = self._run_comparator(baseline_path, target_path)
        assert result.returncode == 0

    def test_failed_exit_nonzero(self, tmp_path):
        """Subprocess: failed comparison → exit 1."""
        torch.manual_seed(42)
        baseline_path = _create_rank_dump(
            tmp_path / "baseline", rank=0, name="t", tensor=torch.randn(10, 10)
        )
        target_path = _create_rank_dump(
            tmp_path / "target", rank=0, name="t", tensor=torch.randn(10, 10) * 100
        )
        result = self._run_comparator(baseline_path, target_path)
        assert result.returncode == 1

    def test_skipped_allow_all_exit_zero(self, tmp_path):
        """Subprocess: skipped comparison with allow_skip_pattern='.*' → exit 0."""
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        result = self._run_comparator(
            baseline_path, target_path, allow_skip_pattern=".*"
        )
        assert result.returncode == 0

    def test_skipped_forbid_all_exit_nonzero(self, tmp_path):
        """Subprocess: skipped comparison with allow_skip_pattern='^$' → exit 1."""
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        result = self._run_comparator(
            baseline_path, target_path, allow_skip_pattern="^$"
        )
        assert result.returncode == 1


class TestReportOutput:
    """Test JSONL report file output via ReportSink."""

    def test_default_report_path(self, tmp_path, capsys):
        """Default writes to <target>/comparator_report.jsonl with ConfigRecord + SummaryRecord."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        args = _make_args(baseline_path, target_path, grouping="raw", report_path=None)

        exit_code: int = run(args)

        report_file: Path = target_path / "comparator_report.jsonl"
        assert report_file.exists()

        report_records: list[AnyRecord] = _parse_jsonl(report_file.read_text())
        assert isinstance(report_records[0], ConfigRecord)
        assert isinstance(report_records[-1], SummaryRecord)
        assert exit_code == 0

    def test_custom_report_path(self, tmp_path, capsys):
        """--report-path writes to the specified location."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        custom_path: Path = tmp_path / "custom" / "report.jsonl"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            report_path=str(custom_path),
        )

        run(args)

        assert custom_path.exists()
        report_records: list[AnyRecord] = _parse_jsonl(custom_path.read_text())
        assert isinstance(report_records[0], ConfigRecord)
        assert isinstance(report_records[-1], SummaryRecord)

    def test_disabled_report(self, tmp_path, capsys):
        """--report-path '' disables file generation."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        args = _make_args(baseline_path, target_path, grouping="raw", report_path="")

        run(args)

        report_file: Path = target_path / "comparator_report.jsonl"
        assert not report_file.exists()

    def test_report_matches_stdout_json(self, tmp_path, capsys):
        """In json mode, report content matches stdout output."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        report_file: Path = tmp_path / "report.jsonl"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            output_format="json",
            report_path=str(report_file),
        )

        capsys.readouterr()
        run(args)

        stdout_lines: list[str] = capsys.readouterr().out.strip().splitlines()
        report_lines: list[str] = report_file.read_text().strip().splitlines()
        assert stdout_lines == report_lines

    def test_text_mode_also_writes_report(self, tmp_path, capsys):
        """Text stdout mode still writes JSONL report."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a"])
        report_file: Path = tmp_path / "report.jsonl"
        args = _make_args(
            baseline_path,
            target_path,
            grouping="raw",
            output_format="text",
            report_path=str(report_file),
        )

        run(args)

        assert report_file.exists()
        report_records: list[AnyRecord] = _parse_jsonl(report_file.read_text())
        assert isinstance(report_records[0], ConfigRecord)
        assert isinstance(report_records[-1], SummaryRecord)

    def test_streaming_flush(self, tmp_path, capsys):
        """Report file is flushed after each record (readable before close)."""
        from sglang.srt.debug_utils.comparator.output_types import report_sink

        report_file: Path = tmp_path / "stream_report.jsonl"
        report_sink.configure(
            output_format="json",
            report_path=report_file,
        )

        report_sink.add(ConfigRecord(config={"test": True}))

        content: str = report_file.read_text()
        assert len(content.strip().splitlines()) == 1
        parsed: AnyRecord = parse_record_json(content.strip())
        assert isinstance(parsed, ConfigRecord)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
