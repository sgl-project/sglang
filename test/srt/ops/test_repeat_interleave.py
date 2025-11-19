import time

import numpy as np
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


def check_correctness_cpu():
    """
    Perform multiple CPU-side correctness checks:
    - Different sizes of grid_thw
    - Different ranges of repeat counts
    - Check that inputs are not modified
    - Check shape, dtype, and values are exactly the same
      between torch_ref_impl and numpy_impl_cpu
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # (T, repeat_min, repeat_max)
    test_configs = [
        (16, 1, 4),  # small T, small repeat counts
        (128, 0, 4),  # allow repeat=0 to test edge cases
        (512, 1, 8),
        (1024, 1, 16),
    ]

    num_cases_per_config = 10

    for T, repeat_min, repeat_max in test_configs:
        for _ in range(num_cases_per_config):
            # grid_thw: [T, 3]
            # col0: repeat count
            # col1, col2: arbitrary positive integers (here 1..16)
            repeats = torch.randint(
                repeat_min, repeat_max + 1, (T, 1), dtype=torch.int32
            )
            th = torch.randint(1, 17, (T, 1), dtype=torch.int32)
            tw = torch.randint(1, 17, (T, 1), dtype=torch.int32)
            grid_thw = torch.cat([repeats, th, tw], dim=1)

            # Save a copy to ensure functions do not modify inputs
            grid_clone = grid_thw.clone()

            out_torch = torch_ref_impl(grid_thw)
            out_numpy = cpu_numpy_impl(grid_thw)

            # Input should not be modified
            assert torch.equal(
                grid_thw, grid_clone
            ), "Function modified input grid_thw!"

            # Shapes must be the same
            assert (
                out_torch.shape == out_numpy.shape
            ), f"Shape mismatch: torch={out_torch.shape}, numpy={out_numpy.shape}"

            # Dtypes must be the same (should both be int32)
            assert (
                out_torch.dtype == out_numpy.dtype == torch.int32
            ), f"dtype mismatch: torch={out_torch.dtype}, numpy={out_numpy.dtype}"

            # Values must be exactly the same
            if not torch.equal(out_torch.cpu(), out_numpy.cpu()):
                diff_idx = (out_torch.cpu() != out_numpy.cpu()).nonzero(as_tuple=False)
                idx0 = diff_idx[0].item()
                raise AssertionError(
                    f"Value mismatch, T={T}, first differing index={idx0}, "
                    f"torch={out_torch[idx0].item()}, "
                    f"numpy={out_numpy[idx0].item()}"
                )

    print("CPU correctness check: PASSED.")


def main():
    # Setting number of threads to reduce noise from thread scheduling;
    # you can comment this out if you prefer default behavior.
    torch.set_num_threads(1)

    # --------------- Correctness check ---------------
    check_correctness_cpu()
    print("\nAll correctness checks passed. Starting benchmark...\n")

    # --------------- Performance benchmark ---------------
    # Typical scales:
    # T = number of rows in grid_thw
    # H, W only participate in multiplication
    configs = [
        (128, 8, 8),
        (512, 8, 8),
        (2048, 8, 8),
        (8192, 8, 8),
    ]

    iters = 2000  # number of iterations per configuration

    print("=== CPU benchmark ===")
    for T, H, W in configs:
        # Construct grid_thw: [T, 3]
        # col0: repeat count
        # col1, col2: multiplicative factors
        grid_thw = torch.randint(1, 5, (T, 3), dtype=torch.int32)
        grid_thw[:, 1] = H
        grid_thw[:, 2] = W

        t_torch, out_torch = benchmark_once(torch_ref_impl, grid_thw, iters=iters)
        t_numpy, out_numpy = benchmark_once(cpu_numpy_impl, grid_thw, iters=iters)

        # Additional safety check: results should match
        same = torch.equal(out_torch.cpu(), out_numpy.cpu())

        print(
            f"[CPU] T={T:5d}, iters={iters:4d} | "
            f"torch={t_torch*1e3:7.2f} ms, "
            f"numpy={t_numpy*1e3:7.2f} ms, "
            f"same={same}"
        )


if __name__ == "__main__":
    main()
