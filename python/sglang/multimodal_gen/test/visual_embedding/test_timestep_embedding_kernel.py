"""
    Timestep embedding kernel test
"""

import unittest

import numpy as np
import tabulate
import torch

from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    timestep_embedding,
    timestep_embedding_cuda,
    timestep_embedding_triton,
)


class TestTimestepEmbed(unittest.TestCase):
    NUM_BATCH = [1, 2, 8, 63, 256, 512, 613, 1024, 1536]
    NUM_DIM = [32, 64, 128, 256, 259, 512, 613, 1024, 2048, 4096]

    def test_correctness(self):
        device = "cuda"
        for B in self.NUM_BATCH:
            for dim in self.NUM_DIM:
                t = torch.randint(-1000, 1000, (B,), device=device, dtype=torch.int32)
                torch_output = timestep_embedding(t, dim)
                triton_output = timestep_embedding_triton(t, dim)
                cuda_output = timestep_embedding_cuda(t, dim)
                assert torch.allclose(
                    torch_output, triton_output, atol=1e-4
                ), f"({B=}, {dim=}), Max diff {(torch_output - triton_output).abs().max()}"
                assert torch.allclose(
                    torch_output, cuda_output, atol=1e-4
                ), f"({B=}, {dim=}), Max diff {(torch_output - cuda_output).abs().max()}"

    def test_dtype(self):
        pass

    def test_perf(self):
        def perf_kernel_fn(kernel_fn: callable, *args, **kwargs):
            warmup_times = 4
            repeat_times = 20
            total_time = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            for _ in range(warmup_times):
                output_fn = kernel_fn(*args, **kwargs)
            torch.cuda.synchronize()

            for _ in range(repeat_times):
                start.record()
                output_fn = kernel_fn(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)

            return total_time / repeat_times

        # TODO: hard code
        device = "cuda"
        results = []

        triton_speedups = []
        cuda_speedups = []
        for B in self.NUM_BATCH:
            for dim in self.NUM_DIM:
                t = torch.randint(-1000, 1000, (B,), device=device, dtype=torch.int32)

                time_torch = perf_kernel_fn(timestep_embedding, t, dim)
                time_triton = perf_kernel_fn(timestep_embedding_triton, t, dim)
                time_cuda = perf_kernel_fn(timestep_embedding_cuda, t, dim)
                speedup_triton = time_torch / time_triton
                speedup_cuda = time_torch / time_cuda

                results.append(
                    {
                        "Batch Size": B,
                        "Dimension": dim,
                        "Torch Time (ms)": time_torch,
                        "Triton Time (ms)": time_triton,
                        "CUDA Time (ms)": time_cuda,
                        "Speedup (Torch/Triton)": speedup_triton,
                        "Speedup (Torch/CUDA)": speedup_cuda,
                    }
                )
                triton_speedups.append(speedup_triton)
                cuda_speedups.append(speedup_cuda)

        print("=== Timestep Embedding Benchmark Results ===")
        print(
            tabulate.tabulate(
                results,
                headers="keys",
                tablefmt="fancy_grid",
                floatfmt=(".0f", ".0f", ".6f", ".6f", ".5f"),
            )
        )
        print(f"Averate Speedup triton: {np.mean(triton_speedups):.4f}")
        print(f"Averate Speedup cuda: {np.mean(cuda_speedups):.4f}")


if __name__ == "__main__":
    unittest.main()
