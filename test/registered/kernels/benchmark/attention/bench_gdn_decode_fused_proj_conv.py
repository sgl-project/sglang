import itertools
from unittest import mock

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.attention import triton_gdn_fused_proj
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous,
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")

# Qwen3.5-35B-A3B GDN decode, TP-local: 16 qk heads / 32 v heads, head_dim 128,
# conv width 4.
NUM_QK_HEADS = 16
NUM_V_HEADS = 32
HEAD_DIM = 128
WIDTH = 4
QKV_DIM = (2 * NUM_QK_HEADS + NUM_V_HEADS) * HEAD_DIM
V_DIM = NUM_V_HEADS * HEAD_DIM

BS_LIST = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    ci_range=[1, 32],
)
configs = list(itertools.product(BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["unfused", "fused_generic", "fused_amd"],
        line_names=[
            "split + causal_conv1d_update",
            "fused (generic kernel)",
            "fused (AMD kernel)",
        ],
        styles=[("red", "--"), ("blue", "-"), ("green", "-.")],
        ylabel="us",
        plot_name="gdn-decode-fused-proj-conv-performance",
        args={},
    )
)
def benchmark(batch: int, provider: str):
    dtype = torch.bfloat16
    qkvz = torch.randn(batch, QKV_DIM + V_DIM, dtype=dtype, device=DEFAULT_DEVICE)
    ba = torch.randn(batch, 2 * NUM_V_HEADS, dtype=dtype, device=DEFAULT_DEVICE)
    weight = torch.randn(QKV_DIM, WIDTH, dtype=dtype, device=DEFAULT_DEVICE)
    state = torch.randn(
        batch + 1, QKV_DIM, WIDTH - 1, dtype=dtype, device=DEFAULT_DEVICE
    )
    indices = torch.arange(batch, dtype=torch.int32, device=DEFAULT_DEVICE)

    if provider == "unfused":

        def fn():
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                qkvz, ba, NUM_QK_HEADS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM
            )
            return (
                causal_conv1d_update(
                    mixed_qkv,
                    state,
                    weight,
                    None,
                    "silu",
                    conv_state_indices=indices,
                ),
                z,
                b,
                a,
            )

    else:
        is_hip = provider == "fused_amd"

        def fn():
            with mock.patch.object(triton_gdn_fused_proj, "_is_hip", is_hip):
                return fused_qkvzba_causal_conv1d_update_contiguous(
                    qkvz,
                    ba,
                    state,
                    weight,
                    None,
                    indices,
                    qkv_dim=QKV_DIM,
                    v_dim=V_DIM,
                    num_v_heads=NUM_V_HEADS,
                    head_v_dim=HEAD_DIM,
                    activation="silu",
                )

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
