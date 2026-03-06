import os

import torch
import triton
import triton.testing
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm as jit_fn
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.test.test_marlin_utils import marlin_quantize

try:
    from sgl_kernel import moe_wna16_marlin_gemm as _aot_import  # noqa: F401

    AOT_AVAILABLE = True
except (ImportError, AttributeError):
    AOT_AVAILABLE = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def stack_and_dev(tensors):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


# Fixed problem dimensions
E = 8
SIZE_K = 4096
SIZE_N = 4096
GROUP_SIZE = 128
TOPK = 2
QUANT_TYPE = scalar_types.uint4b8
DTYPE = torch.float16
BLOCK_SIZE_M = 64

# Quantize weights once (per-expert)
torch.manual_seed(0)
_qweight_l, _scales_l, _w_ref_l = [], [], []
for i in range(E):
    _w = torch.randn((SIZE_N, SIZE_K), dtype=DTYPE, device="cuda") / 20
    _perm = torch.randperm(SIZE_K)
    _w_ref, _qw, _s, _, _, _ = marlin_quantize(_w, QUANT_TYPE, GROUP_SIZE, False, _perm)
    _w_ref_l.append(_w_ref.T)
    _qweight_l.append(_qw)
    _scales_l.append(_s)

_qweight = stack_and_dev(_qweight_l).contiguous()
_scales = stack_and_dev(_scales_l)

_sms = torch.cuda.get_device_properties("cuda").multi_processor_count


def _make_inputs(size_m):
    a = torch.randn((size_m, SIZE_K), dtype=DTYPE, device="cuda") / 10
    score = torch.randn((size_m, E), dtype=DTYPE, device="cuda")
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, TOPK)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, E
    )

    max_workspace_size = (SIZE_N // 64) * (sorted_token_ids.size(0) // BLOCK_SIZE_M)
    max_workspace_size = min(max_workspace_size, _sms * 4)
    workspace = torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")

    c = torch.empty((size_m * TOPK, SIZE_N), dtype=DTYPE, device="cuda")

    return (
        a,
        c,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        workspace,
    )


def _run_jit(
    a,
    c,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    workspace,
    size_m,
):
    return jit_fn(
        a,
        c,
        _qweight,
        None,
        _scales,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=BLOCK_SIZE_M,
        top_k=TOPK,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=QUANT_TYPE,
        size_m=size_m,
        size_n=SIZE_N,
        size_k=SIZE_K,
        is_k_full=True,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def _run_aot(
    a,
    c,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    workspace,
    size_m,
):
    return torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        a,
        c,
        _qweight,
        None,
        _scales,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=BLOCK_SIZE_M,
        top_k=TOPK,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type_id=QUANT_TYPE.id,
        size_m=size_m,
        size_n=SIZE_N,
        size_k=SIZE_K,
        is_k_full=True,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    size_m = 16
    a, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, ntp, workspace = (
        _make_inputs(size_m)
    )
    c_jit = c.clone()
    c_aot = c.clone()
    _run_jit(
        a, c_jit, topk_weights, sorted_token_ids, expert_ids, ntp, workspace, size_m
    )
    _run_aot(
        a, c_aot, topk_weights, sorted_token_ids, expert_ids, ntp, workspace, size_m
    )
    torch.testing.assert_close(c_jit, c_aot, rtol=1e-3, atol=1e-3)
    print("Correctness check passed (JIT vs AOT)")


if IS_CI:
    m_range = [1, 16, 128]
else:
    m_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

if AOT_AVAILABLE:
    line_vals = ["jit", "aot"]
    line_names = ["JIT Kernel", "AOT Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["jit"]
    line_names = ["JIT Kernel"]
    styles = [("blue", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size_m"],
        x_vals=m_range,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="moe-wna16-marlin-gemm-performance",
        args={},
    )
)
def benchmark(size_m, provider):
    a, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, ntp, workspace = (
        _make_inputs(size_m)
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: _run_jit(
            a,
            c.clone(),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            ntp,
            workspace,
            size_m,
        )
    elif provider == "aot":
        fn = lambda: _run_aot(
            a,
            c.clone(),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            ntp,
            workspace,
            size_m,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
