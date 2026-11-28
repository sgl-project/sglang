"""Benchmark the EAGLE verify's target-only tree sampling: multi-CTA kernels vs the stock
one-CTA-per-request kernel, on a 4-token chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.speculative import multi_cta_verify_sampling as mcta
from sglang.kernels.ops.speculative.sampling import (
    tree_speculative_sampling_target_only as stock_tree,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-b200"
)

TOKENS = 4
# The read inputs; the outputs and draft_probs writes are not cloned per iteration.
_READS = (
    "candidates",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
    "uniform_samples",
    "uniform_samples_for_final_sampling",
    "target_probs",
)


def _inputs(batch_size: int, vocab_size: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(batch_size)
    shape = (batch_size, TOKENS, vocab_size)
    target_probs = torch.softmax(
        3.0 * torch.randn(shape, device="cuda", generator=g), dim=-1
    )
    next_token = torch.arange(1, TOKENS + 1, device="cuda")
    next_token[-1] = -1
    return dict(
        candidates=torch.randint(
            0, vocab_size, (batch_size, TOKENS), device="cuda", generator=g
        ),
        retrive_index=torch.arange(batch_size * TOKENS, device="cuda").view(
            batch_size, TOKENS
        ),
        retrive_next_token=next_token.expand(batch_size, TOKENS).contiguous(),
        retrive_next_sibling=torch.full(
            (batch_size, TOKENS), -1, dtype=torch.int64, device="cuda"
        ),
        uniform_samples=torch.rand(batch_size, TOKENS, device="cuda", generator=g),
        uniform_samples_for_final_sampling=torch.rand(
            batch_size, device="cuda", generator=g
        ),
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        predicts=torch.empty(batch_size * TOKENS, dtype=torch.int32, device="cuda"),
        accept_index=torch.empty(batch_size, TOKENS, dtype=torch.int32, device="cuda"),
        accept_token_num=torch.empty(batch_size, dtype=torch.int32, device="cuda"),
    )


@marker.parametrize("vocab_size", [128256, 151936, 248320], [248320])
@marker.parametrize("batch_size", [1, 4, 16, 64, 256], [1, 64])
@marker.benchmark("impl", ["multi_cta", "stock"], gate=["multi_cta"])
def benchmark(vocab_size: int, batch_size: int, impl: str):
    kwargs = _inputs(batch_size, vocab_size)
    if impl == "multi_cta":
        if not mcta.applies(kwargs["target_probs"]):
            marker.skip("the multi-CTA kernels are checked on SM 10.x only")
        fn = mcta.tree_speculative_sampling_target_only
    else:
        fn = stock_tree
    return marker.do_bench(fn, input_kwargs=kwargs, graph_clone_kwargs=_READS)


if __name__ == "__main__":
    benchmark.run()
