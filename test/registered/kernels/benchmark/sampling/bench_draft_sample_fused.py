import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-small",
    disabled="requires SM100; no benchmark lane runs SM100",
)


def fused(logits, temperatures, top_ks):
    from sglang.srt.speculative.spec_utils import sample_draft_proposal

    return sample_draft_proposal(logits, temperatures, top_ks, fused=True)


def stock(logits, temperatures, top_ks):
    from sglang.srt.speculative.spec_utils import sample_draft_proposal

    return sample_draft_proposal(logits, temperatures, top_ks)


def softmax_only(logits, temperatures, top_ks):
    from sglang.kernels.ops.sampling import softmax

    return softmax(logits, temperatures=temperatures)


FN_MAP = {"fused": fused, "stock": stock, "softmax": softmax_only}


# The draft proposal under --speculative-use-rejection-sampling: fp32 logits, one temperature
# and top_k per row; "softmax" is the softmax alone, the floor of the fused kernel.
@marker.parametrize("vocab_size", [129280, 151936, 163840, 248320], [248320])
@marker.parametrize("batch_size", marker.range(9, pattern="pow2"), [1, 64])
@marker.benchmark("impl", ["fused", "stock", "softmax"])
def benchmark(vocab_size: int, batch_size: int, impl: str):
    logits = create_random(batch_size, vocab_size, dtype=torch.float32)
    temperatures = torch.linspace(0.5, 1.5, batch_size, device=logits.device).view(
        -1, 1
    )
    top_ks = torch.full((batch_size,), 20, dtype=torch.int32, device=logits.device)
    return marker.do_bench(FN_MAP[impl], input_args=(logits, temperatures, top_ks))


if __name__ == "__main__":
    if not current_arch().is_sm(10):
        print("The fused draft proposal requires an SM100-family CUDA GPU. Skipping.")
    else:
        benchmark.run()
