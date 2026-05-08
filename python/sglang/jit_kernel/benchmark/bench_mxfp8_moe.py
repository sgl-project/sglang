from __future__ import annotations

import sys
from typing import Any

import torch
import triton

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.mxfp8 import (
    es_sm100_mxfp8_blockscaled_grouped_quant,
    es_sm100_mxfp8_blockscaled_moe_grouped_gemm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-benchmark-1-gpu-large")


def is_sm100_supported(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    return (torch.cuda.get_device_capability(device)[0] == 10) and (
        torch.version.cuda >= "12.8"
    )


_SM100_SUPPORTED = is_sm100_supported()


def _probe_sgl_kernel_group_mm() -> tuple[bool, str]:
    if not _SM100_SUPPORTED:
        return False, "MXFP8 MoE benchmark requires sm100+ with CUDA 12.8+."
    try:
        import sgl_kernel  # noqa: F401
    except Exception as e:
        return False, f"import sgl_kernel failed: {e}"
    if not hasattr(sgl_kernel, "es_sm100_mxfp8_blockscaled_grouped_mm"):
        return False, "sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_mm is missing."
    try:
        pass

        # We assume if it's imported, it works
    except Exception as e:
        return False, f"calling sgl-kernel grouped_mm op failed: {e}"
    return True, ""


_SGL_KERNEL_AVAILABLE, _SGL_KERNEL_REASON = _probe_sgl_kernel_group_mm()


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


def _prepare_case(
    total_tokens: int, n_g: int, k_g: int, num_experts: int, dtype: torch.dtype
) -> dict[str, Any]:
    device = torch.device("cuda")
    base = total_tokens // num_experts
    rem = total_tokens % num_experts
    m_per_expert = [base + (1 if i < rem else 0) for i in range(num_experts)]

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    a_blockscale_offset = 0
    a_blockscale_offsets = []
    b_blockscale_offset = 0
    b_blockscale_offsets = []
    tokens_per_expert_list = []
    expert_ranges = []
    problem_sizes = []

    a_list = []
    b_list = []
    for g in range(num_experts):
        m_g = m_per_expert[g]
        tokens_per_expert_list.append(m_g)
        expert_ranges.append((expert_offset, expert_offset + m_g))
        expert_offsets.append(expert_offset)
        expert_offset += m_g

        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g

        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += align(m_g, 128)

        b_blockscale_offsets.append(b_blockscale_offset)
        b_blockscale_offset += n_g  # n_g already align to 128 in practice

        problem_sizes.append([m_g, n_g, k_g])

        a = torch.randn((m_g, k_g), device=device, dtype=dtype) * 0.1
        b = torch.randn((n_g, k_g), device=device, dtype=dtype) * 0.1
        a_list.append(a)
        b_list.append(b)

    a = torch.concat(a_list, dim=0)
    b = torch.concat(b_list, dim=0)

    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(
        device=device, dtype=torch.int32
    )
    _a_blockscale_offsets = torch.tensor(a_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _b_blockscale_offsets = torch.tensor(b_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _tokens_per_expert = torch.tensor(tokens_per_expert_list).to(
        device=device, dtype=torch.int32
    )
    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)

    a_quant = torch.zeros_like(a, dtype=torch.float8_e4m3fn, device=device)
    a_scale_factor = torch.zeros(
        (a_blockscale_offset, k_g // 32), dtype=torch.uint8, device=device
    )

    b_quant = torch.zeros_like(b, dtype=torch.float8_e4m3fn, device=device)
    b_scale_factor = torch.zeros(
        (num_experts * n_g, k_g // 32), dtype=torch.uint8, device=device
    )

    # Use a global workspace to avoid allocating 1GB every time
    workspace = torch.empty((1024, 1024, 1024), dtype=torch.uint8, device=device)

    es_sm100_mxfp8_blockscaled_grouped_quant(
        a,
        _tokens_per_expert,
        _expert_offsets,
        _a_blockscale_offsets,
        a_quant,
        a_scale_factor,
    )

    es_sm100_mxfp8_blockscaled_grouped_quant(
        b,
        torch.ones_like(_tokens_per_expert) * n_g,
        _aux_expert_offsets,
        _b_blockscale_offsets,
        b_quant,
        b_scale_factor,
    )

    b_quant = b_quant.view(num_experts, n_g, k_g)
    b_scale_factor = b_scale_factor.view(num_experts, n_g, k_g // 32)

    sgl_b_quant = b_quant.transpose(1, 2)
    sgl_b_scale_factor = b_scale_factor.transpose(1, 2)

    return {
        "a": a,
        "b": b.view(num_experts, n_g, k_g),
        "b_quant": b_quant,
        "a_quant": a_quant,
        "b_scale_factor": b_scale_factor,
        "a_scale_factor": a_scale_factor,
        "expert_offsets": _expert_offsets,
        "a_blockscale_offsets": _a_blockscale_offsets,
        "tokens_per_expert": _tokens_per_expert,
        "problem_sizes": _problem_sizes,
        "sgl_b_quant": sgl_b_quant,
        "sgl_b_scale_factor": sgl_b_scale_factor,
        "workspace": workspace,
        "expert_ranges": expert_ranges,
        "dtype": dtype,
    }


def _sgl_kernel_group_mm(case: dict[str, Any]) -> torch.Tensor:
    from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_mm

    a_quant = case["a_quant"]
    sgl_b_quant = case["sgl_b_quant"]
    a_scale_factor = case["a_scale_factor"]
    sgl_b_scale_factor = case["sgl_b_scale_factor"]
    problem_sizes = case["problem_sizes"]
    expert_offsets = case["expert_offsets"]
    a_blockscale_offsets = case["a_blockscale_offsets"]
    dtype = case["dtype"]

    total_tokens = a_quant.shape[0]
    n_g = sgl_b_quant.shape[2]

    # sgl-kernel takes output pre-allocated
    d = torch.empty((total_tokens, n_g), device=a_quant.device, dtype=dtype)
    es_sm100_mxfp8_blockscaled_grouped_mm(
        d,
        a_quant,
        sgl_b_quant,
        a_scale_factor,
        sgl_b_scale_factor,
        problem_sizes,
        expert_offsets,
        a_blockscale_offsets,
    )
    return d


shape_range = get_benchmark_range(
    full_range=[
        # (total_tokens, n_g, k_g, num_experts)
        (1024, 4096, 4096, 64),
        (2048, 4096, 4096, 64),
        (4096, 4096, 4096, 64),
    ]
    + [
        (total_tokens, n_g, k_g, num_experts)
        for total_tokens in [32 * (2**i) for i in range(9)]  # 32 to 8192
        for n_g, k_g, num_experts in [
            # DeepSeek-V3/R1, gateup, TP = 1, EP = 8
            (4096, 7168, 32),
            # DeepSeek-V3/R1, down, TP = 1, EP = 8
            (7168, 2048, 32),
        ]
    ],
    ci_range=[(1024, 2048, 2048, 8)],
)

line_vals = ["jit"]
line_names = ["JIT MXFP8 MoE GroupMM"]
styles = [("green", "-")]

if _SGL_KERNEL_AVAILABLE:
    line_vals.append("sgl_kernel")
    line_names.append("sgl-kernel MXFP8 MoE GroupMM")
    styles.append(("orange", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["total_tokens", "n_g", "k_g", "num_experts"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="mxfp8-moe-groupmm-performance",
        args={},
    )
)
def benchmark(total_tokens, n_g, k_g, num_experts, provider):
    case = _prepare_case(total_tokens, n_g, k_g, num_experts, torch.bfloat16)

    if provider == "jit":
        fn = lambda: es_sm100_mxfp8_blockscaled_moe_grouped_gemm(
            case["b_quant"],
            case["a_quant"],
            case["b_scale_factor"],
            case["a_scale_factor"],
            case["expert_offsets"],
            case["a_blockscale_offsets"],
            case["tokens_per_expert"],
            case["workspace"],
            case["dtype"],
        )
    elif provider == "sgl_kernel":
        fn = lambda: _sgl_kernel_group_mm(case)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Warm up
    fn()

    # Profile
    if provider == "jit":
        torch.cuda.nvtx.range_push("jit")
        fn()
        torch.cuda.nvtx.range_pop()
    elif provider == "sgl_kernel":
        torch.cuda.nvtx.range_push("sgl_kernel")
        fn()
        torch.cuda.nvtx.range_pop()

    return run_benchmark(fn)


if __name__ == "__main__":
    if not _SM100_SUPPORTED:
        print("[skip] MXFP8 MoE GroupMM benchmark requires sm100+ with CUDA 12.8+.")
        sys.exit(0)
    if not _SGL_KERNEL_AVAILABLE:
        print(f"[info] sgl-kernel baseline unavailable: {_SGL_KERNEL_REASON}")
    benchmark.run(print_data=True)
