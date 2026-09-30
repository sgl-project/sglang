import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
    vocab_parallel_embedding,
)
from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=10, stage="jit-kernel-benchmark", runner_config="amd")

# Key order must match the perf_report x_names.
DEFAULTS = dict(
    batch_size=120,
    hidden_size=6144,
    vocab_size=128256,
    tp_size=8,
    token_pattern="uniform",
    dtype="bf16",
)

# One-at-a-time star sweep around DEFAULTS (the full product would be 1170
# configs). Each entry overrides one field (or one coupled field group).
SWEEPS = [
    (
        "batch_size",
        get_benchmark_range(
            [1, 2, 4, 8, 16, 32, 64, 120, 256, 512, 1024, 2048, 4096],
            ci_range=[1, 120],
        ),
    ),
    ("hidden_size", get_benchmark_range([4096, 6144, 7168], ci_range=[])),
    (
        ("vocab_size", "tp_size"),
        get_benchmark_range(
            [(32000, 4), (32000, 8), (128256, 4), (128256, 8), (154880, 8)],
            ci_range=[],
        ),
    ),
    (
        "token_pattern",
        get_benchmark_range(["uniform", "all_local", "all_remote"], ci_range=[]),
    ),
    ("dtype", get_benchmark_range(["bf16", "fp16"], ci_range=[])),
]


def _make_benchmark_configs():
    # Dict keying dedupes the all-defaults config each sweep re-produces.
    configs = {}
    for keys, values in SWEEPS:
        for value in values:
            override = (
                dict(zip(keys, value)) if isinstance(keys, tuple) else {keys: value}
            )
            config = {**DEFAULTS, **override}
            configs[tuple(config.values())] = None
    return list(configs)


BENCHMARK_CONFIGS = _make_benchmark_configs()


def _dtype_from_name(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unknown dtype: {dtype}")


def _make_input_ids(
    batch_size: int, vocab_size: int, tp_size: int, token_pattern: str
) -> torch.Tensor:
    assert vocab_size % tp_size == 0
    per_partition = vocab_size // tp_size
    if token_pattern == "uniform":
        return torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int64, device="cuda"
        )
    if token_pattern == "all_local":
        return torch.randint(
            0,
            per_partition,
            (batch_size,),
            dtype=torch.int64,
            device="cuda",
        )
    if token_pattern == "all_remote":
        return torch.randint(
            per_partition,
            vocab_size,
            (batch_size,),
            dtype=torch.int64,
            device="cuda",
        )
    raise ValueError(f"Unknown token_pattern: {token_pattern}")


def _torch_vocab_parallel_embedding(
    input_ids: torch.Tensor, weight: torch.Tensor, vocab_size: int
):
    masked_input, input_mask = get_masked_input_and_mask(
        input_ids,
        0,
        weight.shape[0],
        0,
        vocab_size,
        vocab_size,
    )
    output = F.embedding(masked_input.long(), weight)
    output.masked_fill_(input_mask.unsqueeze(-1), 0)
    return output


def _triton_vocab_parallel_embedding(
    input_ids: torch.Tensor, weight: torch.Tensor, vocab_size: int
):
    return vocab_parallel_embedding(
        input_ids,
        weight,
        0,
        weight.shape[0],
        0,
        vocab_size,
        vocab_size,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "hidden_size",
            "vocab_size",
            "tp_size",
            "token_pattern",
            "dtype",
        ],
        x_vals=BENCHMARK_CONFIGS,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Fused Triton", "Compiled mask + embedding + masked_fill"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="vocab-parallel-embedding-performance",
        args={},
    )
)
def benchmark(
    batch_size: int,
    hidden_size: int,
    vocab_size: int,
    tp_size: int,
    token_pattern: str,
    dtype: str,
    provider: str,
):
    assert vocab_size % tp_size == 0
    torch_dtype = _dtype_from_name(dtype)
    per_partition = vocab_size // tp_size
    input_ids = _make_input_ids(batch_size, vocab_size, tp_size, token_pattern)
    weight = torch.randn((per_partition, hidden_size), dtype=torch_dtype, device="cuda")

    expected = _torch_vocab_parallel_embedding(input_ids, weight, vocab_size)
    actual = _triton_vocab_parallel_embedding(input_ids, weight, vocab_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if provider == "triton":
        fn = lambda: _triton_vocab_parallel_embedding(input_ids, weight, vocab_size)
    elif provider == "torch":
        fn = lambda: _torch_vocab_parallel_embedding(input_ids, weight, vocab_size)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
