"""DSV4 sparse-prefill public-path CUDA stream latency on SM90.

CUDA events time each provider's public API call; this is neither host
wall-clock allocation latency nor a raw-kernel-only measurement. FlashMLA BF16
and Q8 allocate H64 output plus max-logit/LSE auxiliaries through their public
contracts, while CuTe allocates one compact H16 output. Those contract-level
differences are intentionally part of the result.

Q8 inputs are quantized before timing. Its Q cast and DSV4 KV
gather/dequantize/requantize pipeline are excluded, so Q8 numbers represent a
prequantized attention call rather than backend or end-to-end latency.
"""

from __future__ import annotations

import importlib

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


H_LOGICAL = 16
H_FLASHMLA = 64
D_QK = 512
D_V = 512

CASES = get_benchmark_range(
    full_range=[
        (4096, 49152, 128),
        (4096, 49152, 384),
        (4096, 49152, 512),
        (4096, 49152, 640),
        (4096, 49152, 1152),
    ],
    ci_range=[
        (64, 2048, 128),
        (64, 2048, 512),
        (64, 2048, 640),
        (64, 2048, 1152),
    ],
)


def _make_case(tq: int, skv: int, topk: int):
    generator = torch.Generator(device="cuda").manual_seed(1000 + skv + topk)
    q = (
        torch.randn(
            (tq, H_LOGICAL, D_QK),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    kv = (
        torch.randn(
            (skv, 1, D_QK),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    indices = torch.randint(
        0,
        skv,
        (tq, 1, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    topk_length = torch.full((tq,), topk, dtype=torch.int32, device="cuda")
    sink = torch.linspace(0.05, 0.90, H_LOGICAL, dtype=torch.float32, device="cuda")

    q_padded = (
        torch.randn(
            (tq, H_FLASHMLA, D_QK),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    q_padded[:, :H_LOGICAL].copy_(q)
    sink_padded = torch.linspace(
        -0.5, 0.9, H_FLASHMLA, dtype=torch.float32, device="cuda"
    )
    sink_padded[:H_LOGICAL].copy_(sink)
    return q, q_padded, kv, indices, topk_length, sink, sink_padded


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tq", "skv", "topk"],
        x_vals=CASES,
        line_arg="provider",
        line_vals=["flashmla_h64", "flashmla_q8_h64", "cute_h16"],
        line_names=[
            "FlashMLA BF16 H64 public (out + max + LSE)",
            "FlashMLA Q8 H64 public, prequantized (out + max + LSE)",
            "CuTe BF16 H16 public (compact out only)",
        ],
        styles=[("orange", "--"), ("green", "-."), ("blue", "-")],
        ylabel="public-path CUDA stream latency (us)",
        plot_name="dsv4-cute-sparse-mla-h16-sm90-performance",
        args={},
    )
)
def bench_cute_sparse_mla_h16_sm90(tq: int, skv: int, topk: int, provider: str):
    if not is_sm90_supported():
        raise RuntimeError("native H16 CuTe sparse MLA benchmark requires SM90 CUDA")

    _q, q_padded, kv, indices, topk_length, _sink, sink_padded = _make_case(
        tq, skv, topk
    )
    kv_compact = kv.squeeze(1)
    indices_compact = indices.squeeze(1)
    sm_scale = D_QK**-0.5

    if provider == "flashmla_h64":
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        def fn():
            return flash_mla_sparse_fwd(
                q=q_padded,
                kv=kv,
                indices=indices,
                sm_scale=sm_scale,
                d_v=D_V,
                attn_sink=sink_padded,
                topk_length=topk_length,
            )

    elif provider == "flashmla_q8_h64":
        from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import (
            sparse_mla_q8kv8_prefill_fwd,
        )

        q_fp8 = q_padded.to(torch.float8_e4m3fn)
        kv_fp8 = kv.to(torch.float8_e4m3fn)
        identity_scale = torch.ones((), dtype=torch.float32, device="cuda")

        def fn():
            return sparse_mla_q8kv8_prefill_fwd(
                q=q_fp8,
                kv=kv_fp8,
                indices=indices,
                sm_scale=sm_scale,
                q_scale=identity_scale,
                kv_scale=identity_scale,
                d_v=D_V,
                attn_sink=sink_padded,
                topk_length=topk_length,
            )

    elif provider == "cute_h16":
        cute_h16 = importlib.import_module(
            "sglang.kernels.ops.attention.dsv4.cute_sparse_mla_h16"
        )

        def fn():
            return cute_h16.cute_sparse_mla_h16_fwd(
                q=q_padded,
                kv=kv_compact,
                indices=indices_compact,
                topk_length=topk_length,
                attn_sink=sink_padded,
                sm_scale=sm_scale,
            )

    else:
        raise ValueError(f"unknown provider: {provider}")

    # CUDA-event timing retains public-wrapper work/synchronization issued on
    # the stream, but must not be interpreted as host wall-clock or raw-kernel
    # latency; see the module docstring for each provider's output contract.
    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    bench_cute_sparse_mla_h16_sm90.run(print_data=True)
