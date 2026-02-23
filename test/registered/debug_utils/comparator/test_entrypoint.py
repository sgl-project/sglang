import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.entrypoint import run
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
) -> tuple[Path, Path]:
    """Create baseline and target dump directories with given tensor names.

    If baseline_names is None, uses the same names as tensor_names.
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

    dumper_baseline = _make_dumper(d_baseline)
    for name in baseline_names:
        dumper_baseline.dump(name, baseline_tensor)
    dumper_baseline.step()

    dumper_target = _make_dumper(d_target)
    for name in tensor_names:
        dumper_target.dump(name, target_tensor)
    dumper_target.step()

    return (
        d_baseline / dumper_baseline._config.exp_name,
        d_target / dumper_target._config.exp_name,
    )


def _make_args(baseline_path: Path, target_path: Path, **overrides) -> Namespace:
    defaults = dict(
        baseline_path=str(baseline_path),
        target_path=str(target_path),
        start_id=0,
        end_id=1000000,
        diff_threshold=1e-3,
        filter=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestEntrypoint:
    def test_run_basic(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path)

        run(args)

        output = capsys.readouterr().out
        assert "df_target" in output
        assert "df_baseline" in output
        assert output.count("Check:") == 2
        assert "tensor_a" in output
        assert "tensor_b" in output
        assert "rel_diff" in output
        assert "Skip" not in output

    def test_filter(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args = _make_args(baseline_path, target_path, filter="tensor_a")

        run(args)

        output = capsys.readouterr().out
        assert output.count("Check:") == 1
        assert "tensor_a" in output

    def test_no_baseline_skip(self, tmp_path, capsys):
        baseline_path, target_path = _create_dumps(
            tmp_path,
            tensor_names=["tensor_a", "tensor_extra"],
            baseline_names=["tensor_a"],
        )
        args = _make_args(baseline_path, target_path)

        run(args)

        output = capsys.readouterr().out
        assert output.count("Check:") == 1
        assert "Skip:" in output
        assert "since no baseline" in output

    def test_step_range(self, tmp_path, capsys):
        d_baseline = tmp_path / "baseline"
        d_target = tmp_path / "target"
        d_baseline.mkdir()
        d_target.mkdir()

        torch.manual_seed(42)
        tensor = torch.randn(10, 10)

        exp_paths = []
        for d in [d_baseline, d_target]:
            dumper = _make_dumper(d)
            dumper.dump("t", tensor)
            dumper.step()
            dumper.dump("t", tensor)
            dumper.step()
            dumper.dump("t", tensor)
            dumper.step()
            exp_paths.append(d / dumper._config.exp_name)

        args = _make_args(
            exp_paths[0],
            exp_paths[1],
            start_id=1,
            end_id=1,
        )

        run(args)

        output = capsys.readouterr().out
        assert output.count("Check:") == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
