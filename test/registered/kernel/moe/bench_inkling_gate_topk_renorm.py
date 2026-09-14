"""Benchmark Inkling's stable normalization in packed decode/prefill gates."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.moe.inkling_gate_topk_renorm import (
    inkling_gate_topk_renorm_v2,
)
from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import (
    sigmoid_gate_topk_renorm,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("num_tokens", [1, 16, 128, 1024, 16384], [16, 1024])
@marker.parametrize("negative_tail", [False, True])
@marker.benchmark("provider", ["jit_v2", "triton"])
def benchmark(num_tokens: int, negative_tail: bool, provider: str):
    torch.manual_seed(0)
    storage = torch.randn((num_tokens, 264), device="cuda", dtype=torch.float32)
    if negative_tail:
        storage.sub_(100.0)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    enable_pdl = is_arch_support_pdl()

    def launch(storage, bias, global_scale):
        # Clone the padded storage in marker.do_bench, then take the view.
        # Cloning the view itself would remove the production row alignment.
        logits = storage[:, :258]
        if provider == "jit_v2":
            return inkling_gate_topk_renorm_v2(
                logits,
                bias,
                global_scale,
                1.5,
                return_packed=True,
                enable_pdl=enable_pdl,
            )
        return sigmoid_gate_topk_renorm(
            logits, 6, 2, 1.5, global_scale, bias, return_packed_topk=True
        )

    with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
        return marker.do_bench(launch, input_args=(storage, bias, global_scale))


if __name__ == "__main__":
    benchmark.run()
