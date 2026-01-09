import numpy as np
import pytest
import tabulate
import torch
from diffusers.models.embeddings import get_timestep_embedding
from sgl_kernel.elementwise import timestep_embedding as timestep_embedding_cuda

from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding


@pytest.mark.parametrize(
    "batch_size", [1, 2, 8, 128, 256, 512, 1536, 2048, 4096, 11008, 16384]
)
@pytest.mark.parametrize("dim", [32, 128, 256, 512, 1536, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.bfloat16, torch.float16]
)
def test_timestep_embedding_correctness_with_sgld(batch_size, dim, dtype):
    device = "cuda"
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=device).to(dtype)
    torch_output = timestep_embedding(t, dim)
    cuda_output = timestep_embedding_cuda(t, dim, flip_sin_to_cos=True)
    torch.testing.assert_close(torch_output, cuda_output, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 128, 256, 512, 1536, 2048, 16384])
@pytest.mark.parametrize("dim", [32, 256, 512, 1536, 8192])
@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16])
@pytest.mark.parametrize("flip_sin_to_cos", [False, True])
@pytest.mark.parametrize("downscale_freq_shift", [0, 1])
@pytest.mark.parametrize("scale", [1, 0.01])
def test_timestep_embedding_correctness_with_diffusers(
    batch_size, dim, flip_sin_to_cos, downscale_freq_shift, scale, dtype
):
    device = "cuda"
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=device).to(dtype)
    torch_output = get_timestep_embedding(
        t,
        dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=10000,
    )
    cuda_output = timestep_embedding_cuda(
        t,
        dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=10000,
    )
    torch.testing.assert_close(torch_output, cuda_output, atol=1e-3, rtol=1e-3)


def test_timestep_embedding_perf():
    NUM_BATCH = [1, 2, 8, 63, 256, 512, 613, 1024, 1536]
    NUM_DIM = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    def perf_kernel_fn(kernel_fn: callable, *args, **kwargs):
        warmup_times = 4
        repeat_times = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_times):
            output_fn = kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat_times):
            output_fn = kernel_fn(*args, **kwargs)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeat_times

    device = "cuda"
    results = []

    cuda_speedups = []
    for B in NUM_BATCH:
        for dim in NUM_DIM:
            t = torch.linspace(0, max(100000, B), steps=B, device=device).to(
                torch.int32
            )
            time_torch = perf_kernel_fn(timestep_embedding, t, dim)
            time_cuda = perf_kernel_fn(timestep_embedding_cuda, t, dim)
            speedup_cuda = time_torch / time_cuda

            results.append(
                {
                    "Batch Size": B,
                    "Dimension": dim,
                    "Torch Time (ms)": time_torch,
                    "CUDA Time (ms)": time_cuda,
                    "Speedup (CUDA)": speedup_cuda,
                }
            )
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
    print(f"Average Speedup(cuda): {np.mean(cuda_speedups):.4f}")


if __name__ == "__main__":
    pytest.main([__file__])
