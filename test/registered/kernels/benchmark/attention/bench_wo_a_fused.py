"""Fused inverse-RoPE + WO-A + MXFP8 against the three-launch Triton chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.wo_a import (
    fused_rope_wo_a_bf16,
    wo_a_bf16_small_batch_mxfp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="4-gpu-b200"
)

# TP4 DSV4: 64 attention heads padded in the KV buffer, 16 local, 2 o-groups.
LOCAL_HEADS = 16
HEAD_DIM = 512
ROPE_DIM = 64
MAX_POS = 4096


def _inputs(tokens: int):
    """Both paths get the identical unpadded [T, 16, 512] activation.

    Production pads the attention output to 64 heads, so the real ``x`` is a
    strided view with row stride 32768; the kernel reads the stride off the
    tensor and handles either. Benchmarking unpadded keeps the comparison fair,
    since the Triton chain would otherwise be handed a different tensor.
    """
    device = "cuda"
    torch.manual_seed(0)
    o = torch.randn(tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2, 1024, 4096, dtype=torch.bfloat16, device=device).mul_(0.05)
    angles = torch.outer(
        torch.arange(MAX_POS, device=device, dtype=torch.float32),
        1.0 / 10000.0 ** (torch.arange(32, device=device, dtype=torch.float32) / 32),
    )
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).reshape(MAX_POS, ROPE_DIM)
    freqs_cis = torch.view_as_complex(freqs.view(MAX_POS, 32, 2).contiguous())
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int32, device=device)
    return o, weight, freqs, freqs_cis, positions


@marker.parametrize("tokens", [1, 2, 4, 8, 16], [1, 8])
@marker.benchmark("impl", ["fused", "triton_chain"])
def benchmark(tokens: int, impl: str):
    # The Triton path this replaces only covers the split-K small-batch shape;
    # T=1 goes through a separate GEMV with no fused quantization.
    if impl == "triton_chain" and not 2 <= tokens <= 8:
        marker.skip(f"wo_a_bf16_small_batch_mxfp8 requires 2 <= T <= 8, got {tokens}")

    o, weight, freqs, freqs_cis, positions = _inputs(tokens)
    grouped = o.view(tokens, 2, LOCAL_HEADS * HEAD_DIM // 2)

    if impl == "fused":

        def fn(x, weight, freqs, positions):
            return fused_rope_wo_a_bf16(x, weight, freqs, positions)

        args = (grouped, weight, freqs, positions)
    else:
        # rope in place, then the split-K GEMM, then the reduce+quantize launch.
        def fn(o, weight, freqs_cis, positions):
            fused_rope_inplace(
                o[..., -ROPE_DIM:], None, freqs_cis, positions=positions, inverse=True
            )
            return wo_a_bf16_small_batch_mxfp8(
                o.view(tokens, 2, LOCAL_HEADS * HEAD_DIM // 2), weight
            )

        args = (o, weight, freqs_cis, positions)

    # The 16 MiB weight read dominates; cloning it each replay keeps the
    # measurement L2-cold, which is the serving-realistic case.
    return marker.do_bench(fn, input_args=args, memory_args=(weight,))


if __name__ == "__main__":
    benchmark.run()
