"""Layer 2: PNG generation tests for per-token heatmap visualizer.

Requires matplotlib — uses pytest.importorskip to gracefully skip if absent.
"""

import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)

_PNG_MAGIC: bytes = b"\x89PNG"


@pytest.fixture(autouse=True)
def _skip_if_no_matplotlib() -> None:
    pytest.importorskip("matplotlib")


def _make_comparison_record(
    *,
    name: str,
    baseline: torch.Tensor,
    target: torch.Tensor,
    seq_dim: int = 0,
) -> ComparisonRecord:
    """Build a ComparisonRecord with per-token data from raw tensors."""
    info = compare_tensor_pair(
        x_baseline=baseline,
        x_target=target,
        name=name,
        diff_threshold=1e-3,
        seq_dim=seq_dim,
    )
    return ComparisonRecord(**info.model_dump())


class TestPerTokenVisualizer:
    def test_no_data_returns_none(self, tmp_path: Path) -> None:
        """Empty records list → None returned, no file created."""
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )

        output_path: Path = tmp_path / "empty.png"
        result = generate_per_token_heatmap(records=[], output_path=output_path)

        assert result is None
        assert not output_path.exists()

    def test_no_per_token_data_returns_none(self, tmp_path: Path) -> None:
        """Records without per_token_rel_diff → None."""
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )

        info = compare_tensor_pair(
            x_baseline=torch.randn(4, 8),
            x_target=torch.randn(4, 8),
            name="no_per_token",
            diff_threshold=1e-3,
        )
        record = ComparisonRecord(**info.model_dump())

        output_path: Path = tmp_path / "no_data.png"
        result = generate_per_token_heatmap(records=[record], output_path=output_path)

        assert result is None

    def test_generates_valid_png(self, tmp_path: Path) -> None:
        """Records with per-token data → valid PNG file."""
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )

        torch.manual_seed(42)
        records: list[ComparisonRecord] = [
            _make_comparison_record(
                name=f"tensor_{i}",
                baseline=torch.randn(16, 32),
                target=torch.randn(16, 32),
            )
            for i in range(3)
        ]

        output_path: Path = tmp_path / "heatmap.png"
        result = generate_per_token_heatmap(records=records, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        with open(output_path, "rb") as f:
            magic: bytes = f.read(4)
        assert magic == _PNG_MAGIC

    def test_variable_length_sequences(self, tmp_path: Path) -> None:
        """Records with different token lengths → NaN padding, no crash."""
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )

        torch.manual_seed(42)
        records: list[ComparisonRecord] = [
            _make_comparison_record(
                name="short",
                baseline=torch.randn(4, 8),
                target=torch.randn(4, 8),
            ),
            _make_comparison_record(
                name="medium",
                baseline=torch.randn(16, 8),
                target=torch.randn(16, 8),
            ),
            _make_comparison_record(
                name="long",
                baseline=torch.randn(64, 8),
                target=torch.randn(64, 8),
            ),
        ]

        output_path: Path = tmp_path / "variable.png"
        result = generate_per_token_heatmap(records=records, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        with open(output_path, "rb") as f:
            magic: bytes = f.read(4)
        assert magic == _PNG_MAGIC

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Output path with non-existent parent dirs → dirs created automatically."""
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )

        torch.manual_seed(42)
        record = _make_comparison_record(
            name="test",
            baseline=torch.randn(8, 16),
            target=torch.randn(8, 16),
        )

        output_path: Path = tmp_path / "nested" / "deep" / "heatmap.png"
        result = generate_per_token_heatmap(records=[record], output_path=output_path)

        assert result == output_path
        assert output_path.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
