"""Cold-L2 packet conversion latency, excluding routing/quantization/NCCL."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.quantization.moe_fp8_packet import (
    pack_moe_fp8,
    unpack_moe_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_fp8_packet import pack_reference, unpack_reference

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def measure(rows, operation, provider):
    hidden, topk = 4096, 9
    q = torch.randn((rows, hidden), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand((hidden // 128, rows), device="cuda").t()
    weights = torch.rand((rows, topk), device="cuda")
    ids = torch.randint(-1, 289, (rows, topk), device="cuda", dtype=torch.int32)
    if operation == "pack":
        fn = pack_moe_fp8 if provider == "fused" else pack_reference
        args = q, scale, weights, ids
    else:
        packet = pack_reference(q, scale, weights, ids)
        unpack = unpack_moe_fp8 if provider == "fused" else unpack_reference

        def fn(packet):
            return unpack(packet, hidden, topk)

        args = (packet,)

    # Keep captured outputs live: allocator reuse must not make writes L2-hot.
    # marker rotates all read arguments beyond five times the device's L2.
    outputs = []

    def retained(*inputs):
        result = fn(*inputs)
        outputs.append(result)
        return result

    return marker.do_bench(
        retained,
        input_args=args,
        graph_clone_args="all",
        warmup_iters=3,
        replay_iters=500,
        disable_log_bandwidth=True,
    )


@marker.parametrize("rows", [4096, 8192, 16384, 32768], [8192])
@marker.parametrize("operation", ["pack", "unpack"])
@marker.benchmark("provider", ["reference", "fused"])
def benchmark(rows, operation, provider):
    return measure(rows, operation, provider)


if __name__ == "__main__":
    benchmark.run()
