"""Bench the hybrid ``mla_kv_pack_quantize_fp8`` against an inlined naive Triton baseline."""

import itertools
from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import (
    mla_kv_pack_quantize_fp8 as hybrid_pack,
)
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-kernel-benchmark-1-gpu-large")


@triton.jit
def _triton_mla_kv_pack_quantize_fp8_kernel(
    k_nope_ptr,
    k_pe_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    k_scale_inv,
    v_scale_inv,
    s_total,
    k_nope_stride_t,
    k_nope_stride_h,
    k_pe_stride_t,
    v_stride_t,
    v_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    QK_NOPE: tl.constexpr,
    QK_ROPE: tl.constexpr,
    V_HEAD: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    t_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    t_mask = t_idx < s_total
    nope_idx = tl.arange(0, QK_NOPE)
    rope_idx = tl.arange(0, QK_ROPE)
    v_idx = tl.arange(0, V_HEAD)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    nope_off = (
        t_idx[:, None] * k_nope_stride_t + pid_h * k_nope_stride_h + nope_idx[None, :]
    )
    k_nope = tl.load(k_nope_ptr + nope_off, mask=t_mask[:, None])
    pe_off = t_idx[:, None] * k_pe_stride_t + rope_idx[None, :]
    k_pe = tl.load(k_pe_ptr + pe_off, mask=t_mask[:, None])
    v_off = t_idx[:, None] * v_stride_t + pid_h * v_stride_h + v_idx[None, :]
    v = tl.load(v_ptr + v_off, mask=t_mask[:, None])
    k_nope_fp8 = (k_nope.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    k_pe_fp8 = (k_pe.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    v_fp8 = (v.to(tl.float32) * v_scale_inv).to(FP8_DTYPE)
    k_out_base = t_idx[:, None] * k_out_stride_t + pid_h * k_out_stride_h
    tl.store(
        k_out_ptr + k_out_base + nope_idx[None, :], k_nope_fp8, mask=t_mask[:, None]
    )
    tl.store(
        k_out_ptr + k_out_base + QK_NOPE + rope_idx[None, :],
        k_pe_fp8,
        mask=t_mask[:, None],
    )
    v_out_off = (
        t_idx[:, None] * v_out_stride_t + pid_h * v_out_stride_h + v_idx[None, :]
    )
    tl.store(v_out_ptr + v_out_off, v_fp8, mask=t_mask[:, None])
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _triton_pack(k_nope, k_pe, v, k_out, v_out):
    s, num_heads, qk_nope = k_nope.shape
    qk_rope = k_pe.shape[-1]
    v_head = v.shape[-1]
    k_pe_2d = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
    enable_pdl = is_arch_support_pdl()
    if s < 512:
        block_s, num_warps, num_stages = 1, 1, 2
    elif s < 2048:
        block_s, num_warps, num_stages = 4, 2, 3
    else:
        block_s, num_warps, num_stages = 16, 4, 3
    extra = {"launch_pdl": True} if enable_pdl else {}
    grid = (triton.cdiv(s, block_s), num_heads)
    _triton_mla_kv_pack_quantize_fp8_kernel[grid](
        k_nope,
        k_pe_2d,
        v,
        k_out,
        v_out,
        1.0,
        1.0,
        s,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe_2d.stride(0),
        v.stride(0),
        v.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        QK_NOPE=qk_nope,
        QK_ROPE=qk_rope,
        V_HEAD=v_head,
        FP8_DTYPE=tl.float8e4nv,
        BLOCK_S=block_s,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )


QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
NUM_HEADS = 32
NUM_LAYERS = 8

BS_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 64, 256, 1024, 4096, 8192, 16384],
    ci_range=[1, 64, 1024, 4096, 16384],
)

LINE_VALS = ["hybrid", "triton"]
LINE_NAMES = ["hybrid (v0+v1_flat)", "naive Triton"]
STYLES = [("green", "-"), ("red", "--")]
CONFIGS = list(itertools.product(BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="mla-kv-pack-quantize-fp8-performance",
        args={},
    )
)
def benchmark(batch_size: int, provider: str) -> Tuple[float, float, float]:
    k_nope = torch.randn(
        (NUM_LAYERS, batch_size, NUM_HEADS, QK_NOPE),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    k_pe = torch.randn(
        (NUM_LAYERS, batch_size, 1, QK_ROPE),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    v = torch.randn(
        (NUM_LAYERS, batch_size, NUM_HEADS, V_HEAD),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    k_out = torch.empty(
        (NUM_LAYERS, batch_size, NUM_HEADS, QK_NOPE + QK_ROPE),
        dtype=torch.float8_e4m3fn,
        device=DEFAULT_DEVICE,
    )
    v_out = torch.empty(
        (NUM_LAYERS, batch_size, NUM_HEADS, V_HEAD),
        dtype=torch.float8_e4m3fn,
        device=DEFAULT_DEVICE,
    )
    torch.cuda.synchronize()

    if provider == "hybrid":

        def fn():
            for i in range(NUM_LAYERS):
                hybrid_pack(k_nope[i], k_pe[i], v[i], k_out=k_out[i], v_out=v_out[i])

    else:

        def fn():
            for i in range(NUM_LAYERS):
                _triton_pack(k_nope[i], k_pe[i], v[i], k_out[i], v_out[i])

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


if __name__ == "__main__":
    benchmark.run(print_data=True)
