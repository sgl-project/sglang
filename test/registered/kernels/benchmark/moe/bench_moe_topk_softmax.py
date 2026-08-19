import torch
from sgl_kernel import topk_softmax as aot_topk_softmax

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.moe.moe_topk_softmax import topk_softmax as jit_topk_softmax
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _jit(topk_weights, topk_ids, gating_output):
    jit_topk_softmax(topk_weights, topk_ids, gating_output)


def _aot(topk_weights, topk_ids, gating_output):
    aot_topk_softmax(
        topk_weights=topk_weights, topk_ids=topk_ids, gating_output=gating_output
    )


def _torch(topk_weights, topk_ids, gating_output):
    probs = torch.softmax(gating_output.float(), dim=-1)
    return probs.topk(topk_weights.shape[-1], dim=-1)


FN_MAP = {
    "jit": _jit,
    "aot": _aot,
    "torch": _torch,
}


# 32/128/256/512 take the warp-specialized power-of-two path; 12/160 fall back to
# the two-pass path through the softmax workspace. 512 is kept because it sits on
# the boundary between the two.
@marker.parametrize("num_tokens", [128, 512, 1024, 4096, 8192, 32768], [512, 4096])
@marker.parametrize("num_experts", [32, 128, 256, 512, 12, 160], [256, 160])
@marker.parametrize("topk", [1, 2, 4, 8], [2])
@marker.benchmark("impl", ["jit", "aot", "torch"])
def benchmark(num_tokens: int, num_experts: int, topk: int, impl: str):
    if topk > num_experts:
        marker.skip("topk must be <= num_experts")

    gating_output = create_random(num_tokens, num_experts, dtype=torch.float32)
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(topk_weights, topk_ids, gating_output),
        # Only the gating logits are read, so they are the only arg worth
        # rotating to defeat the L2 cache; the two outputs are written every
        # iteration.
        graph_clone_args=(2,),
        # Routing is latency-bound at these sizes, so an achieved-bandwidth
        # number is not meaningful; report latency only.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
