import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.kimi_k2_moe_fused_gate import (
    kimi_k2_moe_fused_gate as jit_kimi_k2_moe_fused_gate,
)

try:
    from sgl_kernel import kimi_k2_moe_fused_gate as aot_kimi_k2_moe_fused_gate

    AOT_AVAILABLE = True
except ImportError:
    aot_kimi_k2_moe_fused_gate = None
    AOT_AVAILABLE = False

NUM_EXPERTS = 384


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    m, topk = 16, 4
    input_tensor = torch.randn(m, NUM_EXPERTS, dtype=torch.float32, device=DEFAULT_DEVICE)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=DEFAULT_DEVICE) * 0.1
    jit_out, jit_idx = jit_kimi_k2_moe_fused_gate(input_tensor, bias, topk, True, 1.0, False)
    aot_out, aot_idx = aot_kimi_k2_moe_fused_gate(input_tensor, bias, topk, True, 1.0, False)
    torch.testing.assert_close(jit_out, aot_out, rtol=0, atol=0)
    torch.testing.assert_close(jit_idx, aot_idx, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


M_LIST = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    ci_range=[16, 128],
)

TOPK_LIST = get_benchmark_range(
    full_range=[2, 4, 6],
    ci_range=[2, 4],
)

configs = list(itertools.product(M_LIST, TOPK_LIST))

line_vals = ["jit", "pytorch"]
line_names = ["SGL JIT Kernel", "PyTorch"]
styles = [("blue", "-"), ("red", "--")]

if AOT_AVAILABLE:
    line_vals.insert(1, "aot")
    line_names.insert(1, "SGL AOT Kernel")
    styles.insert(1, ("green", "-."))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="kimi-k2-moe-fused-gate-performance",
        args={},
    )
)
def benchmark(m, topk, provider):
    input_tensor = torch.randn(
        m, NUM_EXPERTS, dtype=torch.float32, device=DEFAULT_DEVICE
    )
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=DEFAULT_DEVICE) * 0.1

    if provider == "jit":

        def fn():
            return jit_kimi_k2_moe_fused_gate(
                input_tensor, bias, topk, True, 1.0, False
            )

    elif provider == "aot":

        def fn():
            return aot_kimi_k2_moe_fused_gate(
                input_tensor, bias, topk, True, 1.0, False
            )

    else:  # pytorch

        def fn():
            sigmoid_vals = torch.sigmoid(input_tensor)
            biased = sigmoid_vals + bias.unsqueeze(0)
            _, top_idx = torch.topk(biased, topk, dim=-1)
            scores = torch.gather(sigmoid_vals, 1, top_idx)
            score_sum = scores.sum(dim=-1, keepdim=True)
            return scores / score_sum.clamp(min=1e-6), top_idx

    return run_benchmark(fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
