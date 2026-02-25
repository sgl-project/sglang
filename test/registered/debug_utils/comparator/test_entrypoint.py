import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import sglang.srt.debug_utils.dumper as _dumper_module
from sglang.srt.debug_utils.comparator.entrypoint import run
from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    _OutputRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
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

        records = _run_and_parse(args, capsys)
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

        records = _run_and_parse(args, capsys)
        assert len(_get_comparisons(records)) == 1

    def test_no_baseline_skip(self, tmp_path, capsys):
        """Target tensor missing from baseline emits a SkipRecord with reason baseline_load_failed."""
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path, grouping="raw")

        records = _run_and_parse(args, capsys)
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

        records = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1

    def test_all_valid_records(self, tmp_path, capsys):
        """Every emitted JSON record is a valid _OutputRecord subclass."""
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=2)
        args = _make_args(baseline_path, target_path, grouping="raw")

        records = _run_and_parse(args, capsys)
        assert all(isinstance(r, _OutputRecord) for r in records)

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


class TestEntrypointGroupingLogical:
    """Test `--grouping logical` scenarios"""

    def test_no_dims_single_rank(self, tmp_path, capsys):
        """Single-rank dumps without dims fall back to raw loading."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path)

        records = _run_and_parse(args, capsys)
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

        records = _run_and_parse(args, capsys)
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

        records = _run_and_parse(args, capsys)
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

        records = _run_and_parse(args, capsys)
        _assert_single_comparison_passed(records)

    def test_ambiguous_baseline_no_dims(self, tmp_path, capsys):
        """Multi-rank baseline without dims cannot be unsharded, so it is skipped."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        for rank, shard in [(0, full_tensor[:, :4]), (1, full_tensor[:, 4:])]:
            baseline_path = _create_rank_dump(
                baseline_dir, rank=rank, name="hidden", tensor=shard
            )

        target_path = _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_tensor,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(baseline_path, target_path)

        records = _run_and_parse(args, capsys)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == "baseline_load_failed"

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

        records = _run_and_parse(args, capsys)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.skipped == 0


# --------------------------- Assertion helpers -------------------


def _get_comparisons(records: list[AnyRecord]) -> list[ComparisonRecord]:
    return [r for r in records if isinstance(r, ComparisonRecord)]


def _assert_single_comparison_passed(records: list[AnyRecord]) -> ComparisonRecord:
    comparisons = _get_comparisons(records)
    assert len(comparisons) == 1
    assert comparisons[0].diff is not None
    assert comparisons[0].diff.passed
    return comparisons[0]


# --------------------------- Utils ------------------------------


def _make_dumper(directory: Path) -> _Dumper:
    return _Dumper(
        config=DumperConfig(enable=True, dir=str(directory), enable_http_server=False)
    )


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
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _run_and_parse(args: Namespace, capsys: pytest.CaptureFixture) -> list[AnyRecord]:
    capsys.readouterr()
    run(args)
    return _parse_jsonl(capsys.readouterr().out)


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
) -> Path:
    """Create a dump file via the real dumper, as if running on the given rank."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
                enable_http_server=False,
            )
        )

        static_meta: dict = {"world_rank": rank, "world_size": 1}
        if parallel_info is not None:
            static_meta["sglang_parallel_info"] = parallel_info
        dumper.__dict__["_static_meta"] = static_meta

        dumper.dump(name, tensor, dims=dims)
        dumper.step()

    return directory / _FIXED_EXP_NAME


def _create_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    tp_size: int,
    shard_dim: int,
    dims_str: str,
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
        )
    return directory / _FIXED_EXP_NAME


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
