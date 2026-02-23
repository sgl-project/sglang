import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparison.printer import (
    print_comparison,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_stats(
    mean: float = 0.0,
    std: float = 1.0,
    min: float = -2.0,
    max: float = 2.0,
    p1: float = -1.8,
    p5: float = -1.5,
    p95: float = 1.5,
    p99: float = 1.8,
) -> TensorStats:
    return TensorStats(
        mean=mean, std=std, min=min, max=max, p1=p1, p5=p5, p95=p95, p99=p99
    )


def _make_diff(
    rel_diff: float = 0.0001,
    max_abs_diff: float = 0.0005,
    mean_abs_diff: float = 0.0002,
) -> DiffInfo:
    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_diff_coord=(2, 3),
        baseline_at_max=1.0,
        target_at_max=1.0005,
    )


class TestPrintComparison:
    def test_normal(self, capsys):
        info = TensorComparisonInfo(
            name="test",
            baseline=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(mean=0.1, std=1.0, min=-2.0, max=2.0),
            ),
            target=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(mean=0.1001, std=1.0001, min=-2.0001, max=2.0001),
            ),
            unified_shape=torch.Size([4, 8]),
            shape_mismatch=False,
            diff=_make_diff(),
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.1000 vs 0.1001 (diff: 0.0001)\n"
            "[std] 1.0000 vs 1.0001 (diff: 0.0001)\n"
            "[min] -2.0000 vs -2.0001 (diff: -0.0001)\n"
            "[max] 2.0000 vs 2.0001 (diff: 0.0001)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
        )

    def test_shape_mismatch(self, capsys):
        info = TensorComparisonInfo(
            name="mismatch",
            baseline=TensorInfo(
                shape=torch.Size([3, 4]),
                dtype=torch.float32,
                stats=_make_stats(),
            ),
            target=TensorInfo(
                shape=torch.Size([5, 6]),
                dtype=torch.float32,
                stats=_make_stats(),
            ),
            unified_shape=torch.Size([3, 4]),
            shape_mismatch=True,
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([3, 4]) vs torch.Size([5, 6])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] torch.Size([3, 4]) vs torch.Size([5, 6])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "⚠️ Shape mismatch\n"
        )

    def test_with_downcast(self, capsys):
        info = TensorComparisonInfo(
            name="downcast",
            baseline=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(),
            ),
            target=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.bfloat16,
                stats=_make_stats(),
            ),
            unified_shape=torch.Size([4, 8]),
            shape_mismatch=False,
            diff=_make_diff(rel_diff=0.002, max_abs_diff=0.005, mean_abs_diff=0.001),
            diff_downcast=_make_diff(
                rel_diff=0.0001, max_abs_diff=0.0005, mean_abs_diff=0.0002
            ),
            downcast_dtype=torch.bfloat16,
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[🟠dtype] torch.float32 vs torch.bfloat16\n"
            "After unify [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.bfloat16\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "❌ rel_diff=0.002\t❌ max_abs_diff=0.005\t✅ mean_abs_diff=0.001\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
            "When downcast to torch.bfloat16: "
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
        )

    def test_with_shape_unification(self, capsys):
        info = TensorComparisonInfo(
            name="unify",
            baseline=TensorInfo(
                shape=torch.Size([1, 1, 4, 8]),
                dtype=torch.float32,
                stats=_make_stats(),
            ),
            target=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(),
            ),
            unified_shape=torch.Size([4, 8]),
            shape_mismatch=False,
            diff=_make_diff(),
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([1, 1, 4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "Unify shape: torch.Size([1, 1, 4, 8]) -> torch.Size([4, 8]) "
            "(to match torch.Size([4, 8]))\n"
            "After unify [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
        )

    def test_with_samples(self, capsys):
        info = TensorComparisonInfo(
            name="samples",
            baseline=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(),
                sample="tensor([0.1, 0.2, ...])",
            ),
            target=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=_make_stats(),
                sample="tensor([0.1, 0.3, ...])",
            ),
            unified_shape=torch.Size([4, 8]),
            shape_mismatch=False,
            diff=_make_diff(),
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
            "x_baseline(sample)=tensor([0.1, 0.2, ...])\n"
            "x_target(sample)=tensor([0.1, 0.3, ...])\n"
        )

    def test_none_quantiles(self, capsys):
        stats_no_quantiles = TensorStats(
            mean=0.0,
            std=1.0,
            min=-2.0,
            max=2.0,
            p1=None,
            p5=None,
            p95=None,
            p99=None,
        )
        info = TensorComparisonInfo(
            name="no_quantiles",
            baseline=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=stats_no_quantiles,
            ),
            target=TensorInfo(
                shape=torch.Size([4, 8]),
                dtype=torch.float32,
                stats=stats_no_quantiles,
            ),
            unified_shape=torch.Size([4, 8]),
            shape_mismatch=False,
            diff=_make_diff(),
        )

        print_comparison(info=info, diff_threshold=1e-3)

        assert capsys.readouterr().out == (
            "Raw [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] torch.Size([4, 8]) vs torch.Size([4, 8])\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\t✅ max_abs_diff=0.0005\t✅ mean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=(2, 3) with "
            "baseline=1.0 target=1.0005\n"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
