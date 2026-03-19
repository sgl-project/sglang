import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.visualizer.preprocessing import (
    _preprocess_tensor,
    _reshape_to_balanced_aspect,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestPreprocessTensor:
    def test_1d_becomes_2d(self) -> None:
        t: torch.Tensor = torch.randn(100)
        result: torch.Tensor = _preprocess_tensor(t)
        assert result.ndim == 2

    def test_3d_becomes_2d(self) -> None:
        t: torch.Tensor = torch.randn(2, 3, 4)
        result: torch.Tensor = _preprocess_tensor(t)
        assert result.ndim == 2
        assert result.numel() == t.numel()

    def test_high_dim_becomes_2d(self) -> None:
        t: torch.Tensor = torch.randn(2, 3, 4, 5)
        result: torch.Tensor = _preprocess_tensor(t)
        assert result.ndim == 2
        assert result.numel() == t.numel()

    def test_scalar_becomes_2d(self) -> None:
        t: torch.Tensor = torch.tensor(3.14)
        result: torch.Tensor = _preprocess_tensor(t)
        assert result.ndim == 2
        assert result.numel() == 1

    def test_already_2d_preserves_elements(self) -> None:
        t: torch.Tensor = torch.randn(10, 20)
        result: torch.Tensor = _preprocess_tensor(t)
        assert result.ndim == 2
        assert result.numel() == 200


class TestReshapeToBalancedAspect:
    def test_extreme_wide_gets_fixed(self) -> None:
        t: torch.Tensor = torch.randn(1, 10000)
        result: torch.Tensor = _reshape_to_balanced_aspect(t)
        h, w = result.shape
        ratio: float = max(h, w) / max(min(h, w), 1)
        assert ratio <= 5.0

    def test_extreme_tall_gets_fixed(self) -> None:
        t: torch.Tensor = torch.randn(10000, 1)
        result: torch.Tensor = _reshape_to_balanced_aspect(t)
        h, w = result.shape
        ratio: float = max(h, w) / max(min(h, w), 1)
        assert ratio <= 5.0

    def test_already_balanced_unchanged(self) -> None:
        t: torch.Tensor = torch.randn(100, 100)
        result: torch.Tensor = _reshape_to_balanced_aspect(t)
        assert result.shape == (100, 100)

    def test_preserves_numel(self) -> None:
        t: torch.Tensor = torch.randn(1, 7919)
        result: torch.Tensor = _reshape_to_balanced_aspect(t)
        assert result.numel() == t.numel()


class TestGenerateComparisonFigure:
    @pytest.fixture(autouse=True)
    def _skip_if_no_matplotlib(self) -> None:
        pytest.importorskip("matplotlib")

    def test_nested_output_dir(self, tmp_path: Path) -> None:
        from sglang.srt.debug_utils.comparator.visualizer import (
            generate_comparison_figure,
        )

        output_path: Path = tmp_path / "a" / "b" / "c" / "nested.png"

        generate_comparison_figure(
            baseline=torch.randn(10, 10),
            target=torch.randn(10, 10),
            name="nested",
            output_path=output_path,
        )

        assert output_path.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
