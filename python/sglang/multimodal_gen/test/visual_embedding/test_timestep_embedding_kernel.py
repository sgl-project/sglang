"""
    Timestep embedding kernel test
"""

import unittest
import tabulate
import torch
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.layers.triton_ops import timestep_embedding_triton

class TestTimestepEmbed(unittest.TestCase):
    NUM_BATCH = [1, 2, 8, 63, 256, 512, 613, 1024, 1536]
    NUM_DIM = [32, 64, 128, 256, 259, 512, 613, 1024, 2048, 4096]

    def test_correctness(self):
        device = "cuda"
        for B in self.NUM_BATCH:
            for dim in self.NUM_DIM:
                t = torch.randn((B,), device=device)
                torch_output = timestep_embedding(t, dim)
                triton_output = timestep_embedding_triton(t, dim)
                assert torch.allclose(torch_output, triton_output, atol=1e-6), f"{(torch_output - triton_output).abs().max()}"


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

        for B in self.NUM_BATCH:
            for dim in self.NUM_DIM:
                t = torch.randn((B,), device=device)

                time_torch = perf_kernel_fn(timestep_embedding, t, dim)
                time_triton = perf_kernel_fn(timestep_embedding_triton, t, dim)
                speedup = time_torch / time_triton

                results.append({
                    "Batch Size": B,
                    "Dimension": dim,
                    "Torch Time (ms)": time_torch,
                    "Triton Time (ms)": time_triton,
                    "Speedup (Torch/Triton)": speedup
                })

        print("=== Timestep Embedding Benchmark Results ===")
        print(tabulate.tabulate(
            results,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt=(".0f", ".0f", ".6f", ".6f", ".5f")
        ))
if __name__ == "__main__":
    unittest.main()

