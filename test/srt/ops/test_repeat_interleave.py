import time
from typing import Tuple

import numpy as np
import pytest
import torch

from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy as cpu_numpy_impl


def torch_ref_impl(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of cu_seqlens computation.
    Assumes grid_thw is already on the correct device (CPU here).
    Shape: [T, 3], columns: [repeat_count, H, W]
    """
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
            cu_seqlens.to(torch.int32),
        ]
    )
    return cu_seqlens


def benchmark_once(fn, grid_thw, iters: int = 1000):
    """
    Run a function `fn` on the same input `grid_thw` for `iters` times
    and measure total elapsed time.
    """
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(grid_thw)
    end = time.perf_counter()
    return (end - start), out


# (T, repeat_min, repeat_max)
GRID_TEST_CONFIGS: list[Tuple[int, int, int]] = [
    (16, 1, 4),  # small T, small repeat counts
    (128, 0, 4),  # allow repeat=0 to test edge cases
    (512, 1, 8),
    (1024, 1, 16),
]

NUM_CASES_PER_CONFIG = 10


def _generate_random_grid(T: int, repeat_min: int, repeat_max: int) -> torch.Tensor:
    """
    grid_thw: [T, 3]
    col0: repeat count
    col1, col2: arbitrary positive integers (here 1..16)
    """
    repeats = torch.randint(repeat_min, repeat_max + 1, (T, 1), dtype=torch.int32)
    th = torch.randint(1, 17, (T, 1), dtype=torch.int32)
    tw = torch.randint(1, 17, (T, 1), dtype=torch.int32)
    grid_thw = torch.cat([repeats, th, tw], dim=1)
    return grid_thw


class TestRepeatInterleave:
    @classmethod
    def setup_class(cls):
        torch.set_num_threads(1)

    def setup_method(self, method):
        torch.manual_seed(0)
        np.random.seed(0)

    @pytest.mark.parametrize(
        "T,repeat_min,repeat_max",
        GRID_TEST_CONFIGS,
    )
    @pytest.mark.parametrize("case_idx", range(NUM_CASES_PER_CONFIG))
    def test_cpu_correctness_random_cases(
        self,
        T: int,
        repeat_min: int,
        repeat_max: int,
        case_idx: int,
    ):
        torch.manual_seed(case_idx)
        np.random.seed(case_idx)

        grid_thw = _generate_random_grid(T, repeat_min, repeat_max)

        grid_clone = grid_thw.clone()

        out_torch = torch_ref_impl(grid_thw)
        out_numpy = cpu_numpy_impl(grid_thw)

        assert torch.equal(grid_thw, grid_clone), "Function modified input grid_thw!"

        assert (
            out_torch.shape == out_numpy.shape
        ), f"Shape mismatch: torch={out_torch.shape}, numpy={out_numpy.shape}"

        assert (
            out_torch.dtype == torch.int32
        ), f"Unexpected torch dtype: {out_torch.dtype}"
        assert (
            out_numpy.dtype == torch.int32
        ), f"Unexpected numpy impl dtype: {out_numpy.dtype}"

        if not torch.equal(out_torch.cpu(), out_numpy.cpu()):
            diff_idx = (out_torch.cpu() != out_numpy.cpu()).nonzero(as_tuple=False)
            idx0 = diff_idx[0].item()
            pytest.fail(
                f"Value mismatch, T={T}, case_idx={case_idx}, first differing index={idx0}, "
                f"torch={out_torch[idx0].item()}, "
                f"numpy={out_numpy[idx0].item()}"
            )

    def test_zero_repeat_edge_case(self):
        T = 4
        grid_thw = torch.tensor(
            [
                [0, 4, 4],
                [1, 2, 3],  # 6
                [2, 1, 5],  # 5, 5
                [0, 7, 7],  # 0
            ],
            dtype=torch.int32,
        )

        grid_clone = grid_thw.clone()

        out_torch = torch_ref_impl(grid_thw)
        out_numpy = cpu_numpy_impl(grid_thw)

        assert torch.equal(
            grid_thw, grid_clone
        ), "Function modified input grid_thw with zero repeats!"

        assert torch.equal(
            out_torch.cpu(), out_numpy.cpu()
        ), f"Zero-repeat case mismatch: torch={out_torch}, numpy={out_numpy}"
