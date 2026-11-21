"""GLM-5.3-Flash indexer Q: three-kernel postprocessing vs. the fused path."""

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention.dsa.indexer_q import fused_kpool_q_hadamard_quant
from sglang.kernels.ops.quantization.act_quant_triton import act_quant
from sglang.kernels.ops.quantization.hadamard import hadamard_transform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)

HEADS, DIM = 32, 128
SOFTMAX_SCALE = DIM**-0.5


@torch.compile(dynamic=True)
def scale_head_gate(gate, scale):
    return gate.unsqueeze(-1) * scale * SOFTMAX_SCALE


def unfused(query, gate):
    query = hadamard_transform(query, scale=SOFTMAX_SCALE)
    q8, scale = act_quant(query, DIM, "ue8m0")
    return q8, scale_head_gate(gate, scale)


def fused(query, gate):
    return fused_kpool_q_hadamard_quant(query, gate, SOFTMAX_SCALE)


@marker.parametrize("tokens", [1, 8, 32, 128, 512, 1024], [1, 32, 1024])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark(tokens: int, impl: str):
    if not current_arch().is_sm(12):
        marker.skip("SM12x")
    query = create_random(tokens, HEADS, DIM)
    gate = create_random(tokens, HEADS, dtype=torch.float32)
    return marker.do_bench(
        unfused if impl == "unfused" else fused, input_args=(query, gate)
    )


if __name__ == "__main__":
    benchmark.run()
