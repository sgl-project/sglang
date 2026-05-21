import itertools

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.moe import fused_share_gate_sigmoid_mul


def jit_fused_share_gate_sigmoid_mul(
    hidden_state, share_gate_weight, share_expert_output, **kwargs
) -> torch.Tensor:
    output = fused_share_gate_sigmoid_mul(
        hidden_state, share_gate_weight, share_expert_output
    )
    return output


@torch.inference_mode()
def torch_share_gate_sigmoid_mul(
    hidden_state, share_gate, share_expert_output, **kwargs
) -> torch.Tensor:
    x = share_gate(hidden_state)
    x = F.sigmoid(x)
    x = x * share_expert_output
    return x


BS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(0, 16)],
    ci_range=[16],
)
HIDDEN_SIZE_LIST = get_benchmark_range(
    full_range=[
        1024,
        1536,
        2048,
    ],
    ci_range=[
        1024,
    ],
)

LINE_VALS = [
    "torch",
    "jit",
]
LINE_NAMES = [
    "PyTorch",
    "SGL JIT Kernel",
]
STYLES = [
    ("orange", "-"),
    ("blue", "--"),
]

configs = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_share_gate_sigmoid_mul-performance",
        args={},
    )
)
def benchmark(hidden_size: int, batch_size: int, provider: str):
    hidden_state = torch.randn(
        (batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    share_gate_weight = torch.randn(
        1, hidden_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    share_expert_output = torch.randn(
        batch_size, hidden_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    share_gate = torch.nn.Linear(share_gate_weight.shape[-1], 1, bias=False)
    share_gate.weight = torch.nn.Parameter(share_gate_weight)
    kwargs = {
        "hidden_state": hidden_state,
        "share_gate": share_gate,
        "share_gate_weight": share_gate_weight,
        "share_expert_output": share_expert_output,
    }
    FN_MAP = {
        "torch": torch_share_gate_sigmoid_mul,
        "jit": jit_fused_share_gate_sigmoid_mul,
    }
    fn = lambda: FN_MAP[provider](**kwargs)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
