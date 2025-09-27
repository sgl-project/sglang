# Benchmarks SGLang RMSNorm kernels versus vLLM and FlashInfer across
# (batch_size, seq_len, hidden_size) and prints speed-up.
import argparse
import itertools
import re
from typing import List, Optional, Tuple, Union

import sgl_kernel
import torch
import torch.nn as nn
import triton
import triton.testing
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from sgl_kernel.utils import is_arch_support_pdl
from vllm import _custom_ops as vllm_ops


def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_sglang(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()

    if residual is not None:
        sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps, enable_pdl=enable_pdl)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_flashinfer = rmsnorm_flashinfer(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_sglang = rmsnorm_sglang(
        x.clone(), weight, residual.clone() if residual is not None else None
    )

    if use_residual:
        output_naive = output_naive[0]
        output_flashinfer = output_flashinfer[0]
        output_vllm = output_vllm[0]
        output_sglang = output_sglang[0]

    print(f"Naive output={output_naive}")
    print(f"FlashInfer output={output_flashinfer}")
    print(f"VLLM output={output_vllm}")
    print(f"SGLang output={output_sglang}")

    if (
        torch.allclose(output_naive, output_flashinfer, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_sglang, atol=1e-2, rtol=1e-2)
    ):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


default_batch_sizes = [2**i for i in range(0, 7, 2)]  # 1, 4, 16, 64
default_seq_lens = [2**i for i in range(6, 11, 1)]  # 64, 128, 256, 512, 1024
default_hidden_sizes = [32 * 128, 48 * 128]  # 4096, 6144


def make_configs(bsizes: List[int], slens: List[int], hsizes: List[int]) -> List[Tuple]:
    return list(itertools.product(bsizes, slens, hsizes))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size"],
        x_vals=[],
        line_arg="provider",
        line_vals=["huggingface", "flashinfer", "vllm", "sglang"],
        line_names=["HuggingFace", "FlashInfer", "vLLM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-")],
        ylabel="µs (median)  or  × (speed-up)",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, hidden_size, provider, use_residual):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if use_residual else None

    # timing helper
    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms, qmin, qmax = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
        return 1000 * ms, 1000 * qmax, 1000 * qmin

    if provider == "huggingface":
        return timed(
            lambda: rmsnorm_naive(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "flashinfer":
        return timed(
            lambda: rmsnorm_flashinfer(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "vllm":
        return timed(
            lambda: rmsnorm_vllm(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "sglang":
        return timed(
            lambda: rmsnorm_sglang(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )

    # provider == "speedup"
    t_ref, _, _ = timed(
        lambda: rmsnorm_vllm(
            x.clone(),
            weight,
            residual.clone() if residual is not None else None,
        )
    )
    t_sgl, _, _ = timed(
        lambda: rmsnorm_sglang(
            x.clone(),
            weight,
            residual.clone() if residual is not None else None,
        )
    )
    spd = t_ref / t_sgl
    return (spd, spd, spd)


if __name__ == "__main__":
    p = argparse.ArgumentParser("RMSNorm kernel benchmark")
    p.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    p.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    p.add_argument("--hidden_sizes", type=str2int_list, default=default_hidden_sizes)
    p.add_argument(
        "--use_residual", action="store_true", help="Whether to use residual connection"
    )
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    # coerce lists
    if isinstance(args.batch_sizes, str):
        args.batch_sizes = str2int_list(args.batch_sizes)
    if isinstance(args.seq_lens, str):
        args.seq_lens = str2int_list(args.seq_lens)
    if isinstance(args.hidden_sizes, str):
        args.hidden_sizes = str2int_list(args.hidden_sizes)

    # patch perf_report grid
    benchmark_grid = make_configs(args.batch_sizes, args.seq_lens, args.hidden_sizes)
    if hasattr(benchmark, "benchmarks"):
        benchmark.benchmarks.x_vals = benchmark_grid
    else:
        benchmark.benchmark.x_vals = benchmark_grid

    if args.verify_only:
        ok = calculate_diff(4, 128, args.hidden_sizes[0], args.use_residual)
        print("✅ sanity pass" if ok else "❌ mismatch")
    else:
        benchmark.run(print_data=True, use_residual=args.use_residual)
