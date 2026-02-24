import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

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
        output_format="text",
        match_mode="smart",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _parse_jsonl(output: str) -> list[AnyRecord]:
    return [parse_record_json(line) for line in output.strip().splitlines()]


class TestEntrypointPerRank:
    def test_run_basic(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, match_mode="per-rank")
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Config:" in output
        assert "rel_diff" in output
        assert "Summary:" in output
        assert "Skip" not in output

    def test_filter(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(
            baseline_path, target_path, filter="tensor_a", match_mode="per-rank"
        )
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "rel_diff" in output

    def test_no_baseline_skip(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path, match_mode="per-rank")
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Skip:" in output
        assert "baseline_load_failed" in output

    def test_step_range(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=3)
        args = _make_args(
            baseline_path, target_path, start_step=1, end_step=1, match_mode="per-rank"
        )
        capsys.readouterr()

        run(args)

        output = capsys.readouterr().out
        assert "Summary:" in output


class TestEntrypointJsonl:
    def test_jsonl_basic(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(
            baseline_path, target_path, output_format="json", match_mode="per-rank"
        )
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        assert isinstance(records[0], ConfigRecord)

        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 2

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.skipped == 0

    def test_jsonl_skip(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(
            baseline_path, target_path, output_format="json", match_mode="per-rank"
        )
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == "baseline_load_failed"

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.skipped == 1

    def test_jsonl_all_valid_records(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["t"], num_steps=2)
        args = _make_args(
            baseline_path, target_path, output_format="json", match_mode="per-rank"
        )
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        assert all(isinstance(r, _OutputRecord) for r in records)


def _save_dump_file(
    directory: Path,
    *,
    step: int,
    rank: int,
    dump_index: int,
    name: str,
    tensor: torch.Tensor,
    dims: str | None = None,
    parallel_info_key: str = "sglang_parallel_info",
    parallel_info: dict | None = None,
    extra_tags: dict | None = None,
) -> None:
    """Directly create a dump .pt file with full control over metadata."""
    tags = {"step": step, "rank": rank, "dump_index": dump_index, "name": name}
    if extra_tags:
        tags.update(extra_tags)

    filename = "___".join(f"{k}={v}" for k, v in tags.items()) + ".pt"
    meta = dict(**tags)
    if dims is not None:
        meta["dims"] = dims
    if parallel_info is not None:
        meta[parallel_info_key] = parallel_info

    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"value": tensor, "meta": meta}, directory / filename)


def _create_tp_sharded_dumps(
    directory: Path,
    *,
    full_tensor: torch.Tensor,
    name: str,
    tp_size: int,
    shard_dim: int,
    dims_str: str,
    step: int = 0,
) -> None:
    """Create TP-sharded dump files from a full tensor."""
    shards = list(full_tensor.chunk(tp_size, dim=shard_dim))
    for tp_rank in range(tp_size):
        _save_dump_file(
            directory,
            step=step,
            rank=tp_rank,
            dump_index=tp_rank + 1,
            name=name,
            tensor=shards[tp_rank],
            dims=dims_str,
            parallel_info={"tp_rank": tp_rank, "tp_size": tp_size},
        )


class TestEntrypointUnshard:
    """Integration tests for the unshard pipeline path."""

    def test_smart_mode_no_dims_single_rank(self, tmp_path, capsys):
        """Single-rank dumps without dims: load_and_unshard returns raw tensor."""
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, output_format="json")
        capsys.readouterr()

        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 2
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.skipped == 0

    def test_tp_unshard_same_size(self, tmp_path, capsys):
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8)
        full_target = full_baseline + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_target,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(
            baseline_dir, target_dir, output_format="json", diff_threshold=0.01
        )
        capsys.readouterr()
        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden"
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 1
        assert summary.passed == 1

    def test_tp_unshard_different_sizes(self, tmp_path, capsys):
        torch.manual_seed(42)
        full_baseline = torch.randn(4, 8)
        full_target = full_baseline + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _create_tp_sharded_dumps(
            baseline_dir,
            full_tensor=full_baseline,
            name="hidden",
            tp_size=4,
            shard_dim=1,
            dims_str="b h(tp)",
        )
        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_target,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(
            baseline_dir, target_dir, output_format="json", diff_threshold=0.01
        )
        capsys.readouterr()
        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_one_side_dims_single_baseline(self, tmp_path, capsys):
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)
        target_full = full_tensor + torch.randn(4, 8) * 0.001

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _save_dump_file(
            baseline_dir,
            step=0,
            rank=0,
            dump_index=1,
            name="hidden",
            tensor=full_tensor,
        )

        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=target_full,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(
            baseline_dir, target_dir, output_format="json", diff_threshold=0.01
        )
        capsys.readouterr()
        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        comparisons = [r for r in records if isinstance(r, ComparisonRecord)]
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_ambiguous_baseline_no_dims(self, tmp_path, capsys):
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8)

        baseline_dir = tmp_path / "baseline"
        target_dir = tmp_path / "target"

        _save_dump_file(
            baseline_dir,
            step=0,
            rank=0,
            dump_index=1,
            name="hidden",
            tensor=full_tensor[:, :4],
        )
        _save_dump_file(
            baseline_dir,
            step=0,
            rank=1,
            dump_index=2,
            name="hidden",
            tensor=full_tensor[:, 4:],
        )

        _create_tp_sharded_dumps(
            target_dir,
            full_tensor=full_tensor,
            name="hidden",
            tp_size=2,
            shard_dim=1,
            dims_str="b h(tp)",
        )

        args = _make_args(baseline_dir, target_dir, output_format="json")
        capsys.readouterr()
        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        skips = [r for r in records if isinstance(r, SkipRecord)]
        assert len(skips) == 1
        assert skips[0].reason == "baseline_load_failed"

    def test_summary_counts_unshard(self, tmp_path, capsys):
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
            target_tensor = tensor + torch.randn_like(tensor) * 0.0001
            _create_tp_sharded_dumps(
                target_dir,
                full_tensor=target_tensor,
                name=tensor_name,
                tp_size=2,
                shard_dim=1,
                dims_str="b h(tp)",
            )

        args = _make_args(
            baseline_dir, target_dir, output_format="json", diff_threshold=0.01
        )
        capsys.readouterr()
        run(args)

        records = _parse_jsonl(capsys.readouterr().out)
        summary = records[-1]
        assert isinstance(summary, SummaryRecord)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.skipped == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
