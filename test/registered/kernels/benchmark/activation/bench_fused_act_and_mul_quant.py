"""
Benchmark for fused activation + per-token-group FP8 quantization JIT kernel.

Simulates DeepSeek V3/V4 EP8 production scenario:
  - 256 experts total, EP8 → 32 local experts (87.5% filtered)
  - Two modes: TMA (expert_step=128, sorted+padded) and Non-TMA (expert_step=1, per-token)

Compares:
  - "unfused": run_activation() + sglang_per_token_group_quant_fp8()
  - "fused":   run_activation_quant()

Pre-allocates output buffers to measure pure kernel time.
"""

import torch

from sglang.kernels.ops.activation.activation import (
    run_activation,
    run_activation_quant,
)
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

GROUP_SIZE = 128


# ---------------------------------------------------------------------------
# Helper: create expert_ids with given filter ratio
# ---------------------------------------------------------------------------


def _make_expert_ids(
    num_tokens: int, expert_step: int, filter_ratio: float
) -> torch.Tensor:
    """Create expert_ids with specified filter ratio."""
    num_blocks = (num_tokens + expert_step - 1) // expert_step
    expert_ids = torch.randint(low=0, high=32, size=(num_blocks,), dtype=torch.int32)
    if filter_ratio > 0:
        num_filtered = int(num_blocks * filter_ratio)
        if num_filtered > 0:
            perm = torch.randperm(num_blocks)[:num_filtered]
            expert_ids[perm] = -1
    return expert_ids.cuda()


# ---------------------------------------------------------------------------
# TMA mode (expert_step=128, production MoE sorted layout)
# ---------------------------------------------------------------------------


@marker.parametrize("op_name", ["silu", "gelu", "gelu_tanh"], ["silu"])
@marker.parametrize("hidden_dim", [2048, 4096, 8192], [2048, 4096])
@marker.parametrize(
    "num_tokens",
    [128, 256, 512, 1024, 4096, 16384, 32768, 65536],
    [4096, 32768],
)
@marker.parametrize("filter_ratio", [0.0, 0.5, 0.875], [0.875])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark_tma(
    op_name: str, hidden_dim: int, num_tokens: int, filter_ratio: float, impl: str
):
    """TMA mode: expert_step=128."""
    torch.manual_seed(42)
    expert_step = 128
    input_dim = hidden_dim * 2
    x = create_random(num_tokens, input_dim)
    expert_ids = _make_expert_ids(num_tokens, expert_step, filter_ratio)

    # Pre-allocate
    act_out = torch.empty(num_tokens, hidden_dim, dtype=x.dtype, device=x.device)
    fused_q = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device=x.device
    )
    fused_scale = torch.empty(
        num_tokens, hidden_dim // GROUP_SIZE, dtype=torch.float32, device=x.device
    )

    if impl == "unfused":

        def fn():
            run_activation(
                op_name, x, act_out, expert_ids=expert_ids, expert_step=expert_step
            )
            return sglang_per_token_group_quant_fp8(act_out, GROUP_SIZE)
    else:

        def fn():
            return run_activation_quant(
                op_name,
                x,
                output_q=fused_q,
                output_scale=fused_scale,
                expert_ids=expert_ids,
                expert_step=expert_step,
                group_size=GROUP_SIZE,
            )

    return marker.do_bench(fn)


# ---------------------------------------------------------------------------
# Non-TMA mode (expert_step=1, per-token routing)
# ---------------------------------------------------------------------------


@marker.parametrize("op_name", ["silu", "gelu", "gelu_tanh"], ["silu"])
@marker.parametrize("hidden_dim", [2048, 4096, 8192], [2048, 4096])
@marker.parametrize(
    "num_tokens",
    [8, 32, 128, 512, 1024, 4096, 16384, 32768, 65536],
    [128, 4096, 32768],
)
@marker.parametrize("filter_ratio", [0.0, 0.5, 0.875], [0.875])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark_non_tma(
    op_name: str, hidden_dim: int, num_tokens: int, filter_ratio: float, impl: str
):
    """Non-TMA mode: expert_step=1."""
    torch.manual_seed(42)
    expert_step = 1
    input_dim = hidden_dim * 2
    x = create_random(num_tokens, input_dim)
    expert_ids = _make_expert_ids(num_tokens, expert_step, filter_ratio)

    # Pre-allocate
    act_out = torch.empty(num_tokens, hidden_dim, dtype=x.dtype, device=x.device)
    fused_q = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device=x.device
    )
    fused_scale = torch.empty(
        num_tokens, hidden_dim // GROUP_SIZE, dtype=torch.float32, device=x.device
    )

    if impl == "unfused":

        def fn():
            run_activation(
                op_name, x, act_out, expert_ids=expert_ids, expert_step=expert_step
            )
            return sglang_per_token_group_quant_fp8(act_out, GROUP_SIZE)
    else:

        def fn():
            return run_activation_quant(
                op_name,
                x,
                output_q=fused_q,
                output_scale=fused_scale,
                expert_ids=expert_ids,
                expert_step=expert_step,
                group_size=GROUP_SIZE,
            )

    return marker.do_bench(fn)


# ---------------------------------------------------------------------------
# No-filter mode (Dense MLP scenario, no expert routing)
# ---------------------------------------------------------------------------


@marker.parametrize("op_name", ["silu", "gelu", "gelu_tanh"], ["silu"])
@marker.parametrize("hidden_dim", [2048, 4096, 8192], [2048, 4096])
@marker.parametrize(
    "num_tokens",
    [1, 8, 32, 128, 512, 1024, 4096, 16384, 32768, 65536],
    [1, 128, 1024, 16384],
)
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark_dense(op_name: str, hidden_dim: int, num_tokens: int, impl: str):
    """Dense MLP: no expert filtering."""
    input_dim = hidden_dim * 2
    x = create_random(num_tokens, input_dim)

    # Pre-allocate
    act_out = torch.empty(num_tokens, hidden_dim, dtype=x.dtype, device=x.device)
    fused_q = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device=x.device
    )
    fused_scale = torch.empty(
        num_tokens, hidden_dim // GROUP_SIZE, dtype=torch.float32, device=x.device
    )

    if impl == "unfused":

        def fn():
            run_activation(op_name, x, act_out)
            return sglang_per_token_group_quant_fp8(act_out, GROUP_SIZE)
    else:

        def fn():
            return run_activation_quant(
                op_name,
                x,
                output_q=fused_q,
                output_scale=fused_scale,
                group_size=GROUP_SIZE,
            )

    return marker.do_bench(fn)


if __name__ == "__main__":
    print("=" * 80)
    print("TMA Mode (expert_step=128, multi filter ratio)")
    print("=" * 80)
    benchmark_tma.run()

    print("\n" + "=" * 80)
    print("Non-TMA Mode (expert_step=1, multi filter ratio)")
    print("=" * 80)
    benchmark_non_tma.run()

    print("\n" + "=" * 80)
    print("Dense MLP Mode (no expert filter)")
    print("=" * 80)
    benchmark_dense.run()
