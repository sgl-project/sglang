import sys

import pytest

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


_DEFAULT_PERCENTILES: dict[int, float] = {
    1: -1.8,
    5: -1.5,
    50: 0.0,
    95: 1.5,
    99: 1.8,
}


def _make_stats(
    mean: float = 0.0,
    abs_mean: float = 0.8,
    std: float = 1.0,
    min: float = -2.0,
    max: float = 2.0,
    percentiles: dict[int, float] | None = None,
) -> TensorStats:
    return TensorStats(
        mean=mean,
        abs_mean=abs_mean,
        std=std,
        min=min,
        max=max,
        percentiles=percentiles if percentiles is not None else _DEFAULT_PERCENTILES,
    )


_DEFAULT_ABS_DIFF_PERCENTILES: dict[int, float] = {
    1: 0.0001,
    5: 0.0001,
    50: 0.0002,
    95: 0.0004,
    99: 0.0005,
}


def _make_diff(
    rel_diff: float = 0.0001,
    max_abs_diff: float = 0.0005,
    mean_abs_diff: float = 0.0002,
    abs_diff_percentiles: dict[int, float] | None = None,
    diff_threshold: float = 1e-3,
    passed: bool = True,
) -> DiffInfo:
    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        abs_diff_percentiles=(
            abs_diff_percentiles
            if abs_diff_percentiles is not None
            else _DEFAULT_ABS_DIFF_PERCENTILES
        ),
        max_diff_coord=[2, 3],
        baseline_at_max=1.0,
        target_at_max=1.0005,
        diff_threshold=diff_threshold,
        passed=passed,
    )


def _make_tensor_info(
    shape: list[int] | None = None,
    dtype: str = "torch.float32",
    stats: TensorStats | None = None,
    sample: str | None = None,
) -> TensorInfo:
    return TensorInfo(
        shape=shape if shape is not None else [4, 8],
        dtype=dtype,
        stats=stats if stats is not None else _make_stats(),
        sample=sample,
    )


# Snapshot strings below are intentionally spelled out in full per test.
# The shared skeleton (stats block, diff block) looks duplicated, but keeping
# each test self-contained makes failures immediately readable without chasing
# helper functions.  Do not extract common fragments.
class TestFormatComparison:
    def test_normal(self):
        info = TensorComparisonInfo(
            name="test",
            baseline=_make_tensor_info(
                stats=_make_stats(mean=0.1, std=1.0, min=-2.0, max=2.0),
            ),
            target=_make_tensor_info(
                stats=_make_stats(mean=0.1001, std=1.0001, min=-2.0001, max=2.0001),
            ),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.1000 vs 0.1001 (diff: 0.0001)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0001 (diff: 0.0001)\n"
            "[min] -2.0000 vs -2.0001 (diff: -0.0001)\n"
            "[max] 2.0000 vs 2.0001 (diff: 0.0001)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_shape_mismatch(self):
        info = TensorComparisonInfo(
            name="mismatch",
            baseline=_make_tensor_info(shape=[3, 4]),
            target=_make_tensor_info(shape=[5, 6]),
            unified_shape=[3, 4],
            shape_mismatch=True,
        )

        assert format_comparison(info) == (
            "Raw [shape] [3, 4] vs [5, 6]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [3, 4] vs [5, 6]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "⚠️ Shape mismatch"
        )

    def test_with_downcast(self):
        info = TensorComparisonInfo(
            name="downcast",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(dtype="torch.bfloat16"),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(
                rel_diff=0.002, max_abs_diff=0.005, mean_abs_diff=0.001, passed=False
            ),
            diff_downcast=_make_diff(
                rel_diff=0.0001, max_abs_diff=0.0005, mean_abs_diff=0.0002, passed=True
            ),
            downcast_dtype="torch.bfloat16",
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[🟠dtype] torch.float32 vs torch.bfloat16\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.bfloat16\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "❌ rel_diff=0.002\t❌ max_abs_diff=0.005\t✅ mean_abs_diff=0.001\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "When downcast to torch.bfloat16: "
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_with_shape_unification(self):
        info = TensorComparisonInfo(
            name="unify",
            baseline=_make_tensor_info(shape=[1, 1, 4, 8]),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [1, 1, 4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "Unify shape: [1, 1, 4, 8] -> [4, 8] "
            "(to match [4, 8])\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_with_samples(self):
        info = TensorComparisonInfo(
            name="samples",
            baseline=_make_tensor_info(sample="tensor([0.1, 0.2, ...])"),
            target=_make_tensor_info(sample="tensor([0.1, 0.3, ...])"),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "x_baseline(sample)=tensor([0.1, 0.2, ...])\n"
            "x_target(sample)=tensor([0.1, 0.3, ...])"
        )

    def test_empty_percentiles(self):
        stats_no_quantiles = _make_stats(percentiles={})

        info = TensorComparisonInfo(
            name="no_quantiles",
            baseline=_make_tensor_info(stats=stats_no_quantiles),
            target=_make_tensor_info(stats=stats_no_quantiles),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(abs_diff_percentiles={}),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
