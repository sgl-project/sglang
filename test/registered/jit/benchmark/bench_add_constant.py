import torch

from sglang.jit_kernel.add_constant import _jit_add_constant_module, add_constant
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

CONSTANT = 7


@marker.parametrize(
    "size",
    [128, 1024, 1025, 4096, 4097, 65536, 2**20, 2**22, 2**24],
    [4096, 2**20],
)
@marker.benchmark("provider", ["jit_module", "jit_wrapper", "torch"])
def benchmark(size: int, provider: str):
    src = torch.arange(size, dtype=torch.int32, device=DEFAULT_DEVICE)

    if provider == "jit_module":
        dst = torch.empty_like(src)
        module = _jit_add_constant_module(CONSTANT)

        def fn(src):
            module.add_constant(dst, src)

        return marker.do_bench(
            fn,
            input_args=(src,),
            graph_clone_args=(0,),
            memory_args=(src,),
            memory_output=(dst,),
        )
    elif provider == "jit_wrapper":

        def fn(src):
            return add_constant(src, CONSTANT)

    else:

        def fn(src):
            return src + CONSTANT

    return marker.do_bench(
        fn,
        input_args=(src,),
        graph_clone_args=(0,),
        memory_args=(src,),
    )


if __name__ == "__main__":
    benchmark.run()
