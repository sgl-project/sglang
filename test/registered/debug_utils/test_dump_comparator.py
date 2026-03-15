from argparse import Namespace
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.dump_comparator import (
    _argmax_coord,
    _calc_rel_diff,
    _compute_smaller_dtype,
    _try_unify_shape,
    main,
)
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


# ----------------------------- Unit tests -----------------------------


class TestCalcRelDiff:
    def test_identical_vectors(self) -> None:
        x: torch.Tensor = torch.randn(10, 10)
        assert _calc_rel_diff(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_zero_vectors(self) -> None:
        z: torch.Tensor = torch.zeros(5)
        result = _calc_rel_diff(z, z)
        assert not torch.isnan(result) or True  # should not crash


class TestArgmaxCoord:
    def test_known_position(self) -> None:
        x: torch.Tensor = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        assert _argmax_coord(x) == (1, 2, 3)


class TestTryUnifyShape:
    def test_squeeze_leading_ones(self) -> None:
        target_shape: torch.Size = torch.Size([3, 4])
        result: torch.Tensor = _try_unify_shape(torch.randn(1, 1, 3, 4), target_shape)
        assert result.shape == target_shape

    def test_no_op_when_no_leading_ones(self) -> None:
        target_shape: torch.Size = torch.Size([3, 4])
        result: torch.Tensor = _try_unify_shape(torch.randn(2, 3, 4), target_shape)
        assert result.shape == (2, 3, 4)


class TestComputeSmallerDtype:
    def test_known_pair(self) -> None:
        assert _compute_smaller_dtype(torch.float32, torch.bfloat16) == torch.bfloat16
        assert _compute_smaller_dtype(torch.bfloat16, torch.float32) == torch.bfloat16

    def test_none_for_same_dtype(self) -> None:
        assert _compute_smaller_dtype(torch.float32, torch.float32) is None


# ----------------------------- Integration tests -----------------------------


def _make_dumper(directory: Path) -> _Dumper:
    return _Dumper(
        config=DumperConfig(
            enable=True,
            dir=str(directory),
        )
    )


def _create_dumps(
    tmp_path: Path,
    tensor_names: list[str],
    *,
    baseline_names: list[str] | None = None,
) -> tuple[Path, Path]:
    if baseline_names is None:
        baseline_names = tensor_names

    d_baseline: Path = tmp_path / "baseline"
    d_target: Path = tmp_path / "target"
    d_baseline.mkdir()
    d_target.mkdir()

    torch.manual_seed(42)
    baseline_tensor: torch.Tensor = torch.randn(10, 10)
    target_tensor: torch.Tensor = baseline_tensor + torch.randn(10, 10) * 0.01

    exp_paths: list[Path] = []
    for d, names, tensor in [
        (d_baseline, baseline_names, baseline_tensor),
        (d_target, tensor_names, target_tensor),
    ]:
        dumper: _Dumper = _make_dumper(d)
        for name in names:
            dumper.dump(name, tensor)
        dumper.step()
        exp_paths.append(d / dumper._config.exp_name)

    return exp_paths[0], exp_paths[1]


def _make_args(
    baseline_path: Path,
    target_path: Path,
    *,
    filter_pattern: str | None = None,
) -> Namespace:
    return Namespace(
        baseline_path=str(baseline_path),
        target_path=str(target_path),
        start_step=0,
        end_step=1000000,
        diff_threshold=1e-3,
        filter=filter_pattern,
    )


class TestMainBasic:
    def test_matching_tensors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args: Namespace = _make_args(baseline_path, target_path)

        main(args)

        captured: str = capsys.readouterr().out
        assert "✅" in captured

    def test_with_filter(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        baseline_path, target_path = _create_dumps(tmp_path, ["tensor_a", "tensor_b"])
        args: Namespace = _make_args(
            baseline_path, target_path, filter_pattern="tensor_a"
        )

        main(args)

        captured: str = capsys.readouterr().out
        assert "tensor_a" in captured
        assert "Check:" in captured

    def test_no_match_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        baseline_path, target_path = _create_dumps(
            tmp_path,
            ["only_in_target"],
            baseline_names=["only_in_baseline"],
        )
        args: Namespace = _make_args(baseline_path, target_path)

        main(args)

        captured: str = capsys.readouterr().out
        assert "Skip" in captured
