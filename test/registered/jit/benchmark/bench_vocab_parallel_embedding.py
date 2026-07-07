import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.srt.layers.triton_ops.vocab_parallel_embedding import (
    vocab_parallel_embedding,
)
from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEFAULT_HIDDEN_SIZE = 6144
DEFAULT_VOCAB_SIZE = 128256
DEFAULT_TP_SIZE = 8
DEFAULT_TOKEN_PATTERN = "uniform"
DEFAULT_DTYPE = "bf16"

BATCH_SIZE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 120, 256, 512, 1024, 2048, 4096]
HIDDEN_SIZE_SWEEP = [4096, 6144, 7168]
VOCAB_TP_SWEEP = [(32000, 4), (32000, 8), (128256, 4), (128256, 8), (154880, 8)]
TOKEN_PATTERN_SWEEP = ["uniform", "all_local", "all_remote"]
DTYPE_SWEEP = ["bf16", "fp16"]


def _dedupe_configs(configs):
    seen = set()
    out = []
    for config in configs:
        if config not in seen:
            seen.add(config)
            out.append(config)
    return out


def _make_benchmark_configs():
    default_tail = (
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_VOCAB_SIZE,
        DEFAULT_TP_SIZE,
        DEFAULT_TOKEN_PATTERN,
        DEFAULT_DTYPE,
    )
    if is_in_ci():
        return [(batch_size, *default_tail) for batch_size in [1, 120]]

    configs = []
    configs.extend((batch_size, *default_tail) for batch_size in BATCH_SIZE_SWEEP)
    configs.extend(
        (
            120,
            hidden_size,
            DEFAULT_VOCAB_SIZE,
            DEFAULT_TP_SIZE,
            DEFAULT_TOKEN_PATTERN,
            DEFAULT_DTYPE,
        )
        for hidden_size in HIDDEN_SIZE_SWEEP
    )
    configs.extend(
        (
            120,
            DEFAULT_HIDDEN_SIZE,
            vocab_size,
            tp_size,
            DEFAULT_TOKEN_PATTERN,
            DEFAULT_DTYPE,
        )
        for vocab_size, tp_size in VOCAB_TP_SWEEP
    )
    configs.extend(
        (
            120,
            DEFAULT_HIDDEN_SIZE,
            DEFAULT_VOCAB_SIZE,
            DEFAULT_TP_SIZE,
            token_pattern,
            DEFAULT_DTYPE,
        )
        for token_pattern in TOKEN_PATTERN_SWEEP
    )
    configs.extend(
        (
            120,
            DEFAULT_HIDDEN_SIZE,
            DEFAULT_VOCAB_SIZE,
            DEFAULT_TP_SIZE,
            DEFAULT_TOKEN_PATTERN,
            dtype,
        )
        for dtype in DTYPE_SWEEP
    )
    return _dedupe_configs(configs)


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
