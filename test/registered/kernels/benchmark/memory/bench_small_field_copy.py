from unittest import mock

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE
from sglang.kernels.ops.memory import small_field_copy as small_field_copy_module
from sglang.kernels.ops.memory.small_field_copy import small_field_copy
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_DRAFT_TOKENS = 7


def make_post_verify_inputs(batch_size: int, sampling: str):
    # The fields eagle_post_verify_graphs._post_verify_inputs stages each round, with mamba track indices.
    num_tokens = batch_size * NUM_DRAFT_TOKENS
    retrieve = torch.randint(
        -1, NUM_DRAFT_TOKENS, (3, batch_size, NUM_DRAFT_TOKENS), device=DEFAULT_DEVICE
    )
    srcs = [
        torch.randint(0, 1 << 17, (num_tokens,), device=DEFAULT_DEVICE),
        *retrieve,
        torch.randint(1, 1 << 17, (batch_size,), device=DEFAULT_DEVICE),
        torch.randint(0, 1 << 10, (batch_size,), device=DEFAULT_DEVICE),
        torch.randint(0, 1 << 11, (batch_size,), device=DEFAULT_DEVICE),
    ]
    if sampling == "sampled":
        srcs += [
            torch.rand(batch_size, 1, device=DEFAULT_DEVICE),
            torch.randint(
                1, 64, (batch_size,), dtype=torch.int32, device=DEFAULT_DEVICE
            ),
            torch.rand(batch_size, device=DEFAULT_DEVICE),
        ]
    return [torch.empty_like(src) for src in srcs], srcs


def torch_foreach_copy(dsts, srcs):
    # The post-verify fallback: one multi-tensor copy per dtype.
    groups = {}
    for dst, src in zip(dsts, srcs):
        group_dsts, group_srcs = groups.setdefault(src.dtype, ([], []))
        group_dsts.append(dst)
        group_srcs.append(src)
    for group_dsts, group_srcs in groups.values():
        torch._foreach_copy_(group_dsts, group_srcs, non_blocking=True)


@marker.parametrize("sampling", ["greedy", "sampled"])
@marker.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64], [1, 16])
@marker.benchmark("impl", ["per_element", "vec16", "torch_foreach"])
def benchmark(batch_size: int, sampling: str, impl: str):
    torch.manual_seed(42)
    dsts, srcs = make_post_verify_inputs(batch_size, sampling)
    fn = torch_foreach_copy if impl == "torch_foreach" else small_field_copy
    with mock.patch.object(small_field_copy_module, "_VEC", impl == "vec16"):
        if impl != "torch_foreach":
            assert small_field_copy(dsts, srcs)
        return marker.do_bench(
            fn,
            input_args=(dsts, srcs),
            memory_args=(srcs,),
            memory_output=(dsts,),
        )


if __name__ == "__main__":
    benchmark.run()
