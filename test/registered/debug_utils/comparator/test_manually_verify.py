"""Visual comparison figure tests — CI sanity check + human verification.

This file serves two purposes:
1. CI sanity check: ensures generate_comparison_figure() runs without errors
   across various tensor scenarios (registered via register_cpu_ci).
2. Human verification: all generated PNGs are copied to /tmp/comparator_manual_verify/
   so they can be pulled back to a local machine for visual inspection.

Run:
    python -m pytest test/registered/debug_utils/comparator/test_manually_verify.py -x -v

Human verification:
    After running, images are at /tmp/comparator_manual_verify/.
    Each test's docstring describes the expected visual appearance.
"""

import shutil
import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default", nightly=True)

_PUBLISH_DIR: Path = Path("/tmp/comparator_manual_verify")
_PNG_MAGIC: bytes = b"\x89PNG"


@pytest.fixture(scope="session")
def publish_dir() -> Path:
    """Fixed output dir for human inspection — files are copied here after generation."""
    if _PUBLISH_DIR.exists():
        shutil.rmtree(_PUBLISH_DIR)
    _PUBLISH_DIR.mkdir(parents=True)
    return _PUBLISH_DIR


def _assert_valid_png(path: Path) -> None:
    assert path.exists(), f"PNG not created: {path}"
    assert path.stat().st_size > 0, f"PNG is empty: {path}"
    with open(path, "rb") as f:
        magic: bytes = f.read(4)
    assert magic == _PNG_MAGIC, f"Not a valid PNG: {path}"


def _generate_and_publish(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    name: str,
    tmp_path: Path,
    publish_dir: Path,
) -> Path:
    from sglang.srt.debug_utils.comparator.visualizer import (
        generate_comparison_figure,
    )

    output_path: Path = tmp_path / f"{name}.png"
    generate_comparison_figure(
        baseline=baseline,
        target=target,
        name=name,
        output_path=output_path,
    )

    _assert_valid_png(output_path)
    shutil.copy2(src=output_path, dst=publish_dir / output_path.name)
    return output_path


@pytest.fixture(autouse=True)
def _skip_if_no_matplotlib() -> None:
    pytest.importorskip("matplotlib")


class TestBundleDetailsManualVerify:
    def test_normal_small_diff(self, tmp_path: Path, publish_dir: Path) -> None:
        """Two nearly-identical tensors (randn + 0.01 noise).

        Expected: All 6 panel rows visible. Diff heatmap nearly uniform light color.
        Hist2d tightly clustered along the red diagonal line.
        """
        baseline: torch.Tensor = torch.randn(32, 64)
        target: torch.Tensor = baseline + torch.randn(32, 64) * 0.01

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="normal_small_diff",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_significant_diff(self, tmp_path: Path, publish_dir: Path) -> None:
        """Two tensors with larger differences (randn + 0.5 noise).

        Expected: All 6 panel rows visible. Diff heatmap shows noticeable structure.
        Hist2d scatter is broader, spread away from the diagonal.
        """
        baseline: torch.Tensor = torch.randn(32, 64)
        target: torch.Tensor = baseline + torch.randn(32, 64) * 0.5

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="significant_diff",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_shape_mismatch(self, tmp_path: Path, publish_dir: Path) -> None:
        """Baseline 32x64, target 16x32 — shapes do not match.

        Expected: Only 2 panel rows (baseline heatmap, target heatmap).
        No diff/histogram/hist2d/sampled panels since diff cannot be computed.
        """
        baseline: torch.Tensor = torch.randn(32, 64)
        target: torch.Tensor = torch.randn(16, 32)

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="shape_mismatch",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_large_tensor(self, tmp_path: Path, publish_dir: Path) -> None:
        """4000x4000 tensor — triggers internal downsampling.

        Expected: Figure renders normally without OOM. Downsampled panels
        should still look reasonable.
        """
        baseline: torch.Tensor = torch.randn(4000, 4000)
        target: torch.Tensor = baseline + torch.randn(4000, 4000) * 0.001

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="large_tensor",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_1d_tensor(self, tmp_path: Path, publish_dir: Path) -> None:
        """1D tensor (256,) — internally reshaped to 2D before plotting.

        Expected: All 6 panel rows visible. The heatmap shape reflects the
        reshaped 2D form, not the original 1D.
        """
        baseline: torch.Tensor = torch.randn(256)
        target: torch.Tensor = baseline + 0.01

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="1d_tensor",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_constant_tensor(self, tmp_path: Path, publish_dir: Path) -> None:
        """All-zero baseline, tiny-valued target.

        Expected: Colorbar range is extremely small. Histogram concentrates in
        a single bin. No rendering errors from near-zero variance.
        """
        baseline: torch.Tensor = torch.zeros(32, 64)
        target: torch.Tensor = torch.ones(32, 64) * 1e-8

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="constant_tensor",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )

    def test_extreme_values(self, tmp_path: Path, publish_dir: Path) -> None:
        """Tensor containing values spanning 1e-10 to 1e10.

        Expected: Log10 panels handle the wide range gracefully. No inf/nan
        artifacts in the rendered figure.
        """
        baseline: torch.Tensor = torch.randn(32, 64).abs()
        baseline[0, 0] = 1e-10
        baseline[0, 1] = 1e10
        target: torch.Tensor = baseline + torch.randn(32, 64) * 0.01

        _generate_and_publish(
            baseline=baseline,
            target=target,
            name="extreme_values",
            tmp_path=tmp_path,
            publish_dir=publish_dir,
        )


class TestPerTokenHeatmapManualVerify:
    def test_increasing_diff(self, tmp_path: Path, publish_dir: Path) -> None:
        """Per-token heatmap with linearly increasing diff across token positions.

        Expected: Heatmap shows a clear left-to-right gradient — dark/cold on
        the left (small diff), bright/hot on the right (large diff). Multiple
        rows for different tensor names. Colorbar shows log10 scale.
        """
        from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )
        from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
            compare_tensor_pair,
        )

        torch.manual_seed(42)
        seq_len: int = 64
        hidden_dim: int = 128
        num_tensors: int = 5

        records: list[ComparisonRecord] = []
        for i in range(num_tensors):
            baseline: torch.Tensor = torch.randn(seq_len, hidden_dim)
            noise_scale: torch.Tensor = torch.linspace(
                1e-6, 0.5, steps=seq_len
            ).unsqueeze(1)
            target: torch.Tensor = baseline + torch.randn_like(baseline) * noise_scale

            info = compare_tensor_pair(
                x_baseline=baseline,
                x_target=target,
                name=f"layer_{i}_hidden_states",
                diff_threshold=1e-3,
                seq_dim=0,
            )
            records.append(ComparisonRecord(**info.model_dump()))

        output_path: Path = tmp_path / "per_token_increasing_diff.png"
        result = generate_per_token_heatmap(records=records, output_path=output_path)

        assert result is not None
        _assert_valid_png(output_path)
        shutil.copy2(src=output_path, dst=publish_dir / output_path.name)

    def test_single_spike(self, tmp_path: Path, publish_dir: Path) -> None:
        """Per-token heatmap where only one token position has large diff.

        Expected: Heatmap shows one bright vertical stripe at the spike position,
        rest is dark/cold.
        """
        from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord
        from sglang.srt.debug_utils.comparator.per_token_visualizer import (
            generate_per_token_heatmap,
        )
        from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
            compare_tensor_pair,
        )

        torch.manual_seed(42)
        seq_len: int = 64
        hidden_dim: int = 128
        spike_pos: int = 32
        num_tensors: int = 4

        records: list[ComparisonRecord] = []
        for i in range(num_tensors):
            baseline: torch.Tensor = torch.randn(seq_len, hidden_dim)
            target: torch.Tensor = baseline.clone()
            target[spike_pos, :] += torch.randn(hidden_dim) * 5.0

            info = compare_tensor_pair(
                x_baseline=baseline,
                x_target=target,
                name=f"layer_{i}_attn_output",
                diff_threshold=1e-3,
                seq_dim=0,
            )
            records.append(ComparisonRecord(**info.model_dump()))

        output_path: Path = tmp_path / "per_token_single_spike.png"
        result = generate_per_token_heatmap(records=records, output_path=output_path)

        assert result is not None
        _assert_valid_png(output_path)
        shutil.copy2(src=output_path, dst=publish_dir / output_path.name)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
