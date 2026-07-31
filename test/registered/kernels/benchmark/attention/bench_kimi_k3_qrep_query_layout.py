"""Benchmark removal of the replicated-Q post-BMM contiguous copy."""

from __future__ import annotations

import os

import flashinfer
import torch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import get_benchmark_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

NUM_HEADS = 96
QK_NOPE_DIM = 512
QK_ROPE_DIM = 64
NUM_TOKENS = get_benchmark_range([1, 17, 128], [1, 17, 128])
PROVIDERS = ["copy_then_pack", "strided_view_pack"]
if os.environ.get("SGLANG_QREP_BENCH_REVERSE") == "1":
    PROVIDERS.reverse()


@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, provider: str):
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(32541 + num_tokens)
    q_nope_storage = torch.randn(
        NUM_HEADS,
        num_tokens,
        QK_NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_nope_view = q_nope_storage.transpose(0, 1)
    q_rope = torch.randn(
        num_tokens,
        NUM_HEADS,
        QK_ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_nope = torch.randn(
        num_tokens,
        QK_NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_rope = torch.randn(
        num_tokens,
        QK_ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cos_sin_cache = torch.randn(
        2048,
        QK_ROPE_DIM,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
    q_fp8 = torch.empty(
        (num_tokens, NUM_HEADS, QK_NOPE_DIM + QK_ROPE_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    k_fp8 = torch.empty(
        (num_tokens, QK_NOPE_DIM + QK_ROPE_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    def fn(q_nope: torch.Tensor):
        if provider == "copy_then_pack":
            q_nope = q_nope.contiguous()
        return flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=False,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_fp8[..., QK_NOPE_DIM:],
            k_rope_out=k_fp8[..., QK_NOPE_DIM:],
            q_nope_out=q_fp8[..., :QK_NOPE_DIM],
            k_nope_out=k_fp8[..., :QK_NOPE_DIM],
            enable_pdl=True,
        )

    return marker.do_bench(
        fn,
        input_args=(q_nope_view,),
        graph_clone_args=(0,),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
