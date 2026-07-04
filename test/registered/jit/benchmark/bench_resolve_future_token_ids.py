import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.resolve_future_token_ids import resolve_future_token_ids_cuda
from sglang.srt.utils import get_compiler_backend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


def _torch_resolve(input_ids, future_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


_compiled_resolve = torch.compile(
    _torch_resolve, dynamic=True, backend=get_compiler_backend()
)


FN_MAP = {
    "jit": resolve_future_token_ids_cuda,
    "torch_compile": _compiled_resolve,
    "torch": _torch_resolve,
}


@marker.parametrize("size", [2**n for n in range(4, 16)], [256, 4096])  # 16 … 32K
@marker.benchmark("provider", ["jit", "torch_compile", "torch"])
def benchmark(size: int, provider: str):
    map_size = 8192
    future_map = torch.randint(
        0, 50000, (map_size,), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    input_ids = torch.randint(
        -map_size + 1, 50000, (size,), dtype=torch.int64, device=DEFAULT_DEVICE
    )

    return marker.do_bench(
        FN_MAP[provider],
        input_args=(input_ids, future_map),
        # both args are read; input_ids is also written in-place -> clone both.
        graph_clone_args=(0, 1),
        memory_args=(input_ids, future_map),
        memory_output=(input_ids,),  # in-place write to input_ids
    )


if __name__ == "__main__":
    benchmark.run()
