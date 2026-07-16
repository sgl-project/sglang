import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.clamp_position import clamp_position_cuda
from sglang.srt.utils import get_compiler_backend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=13, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=16, stage="jit-kernel-benchmark", runner_config="amd")


def _torch_clamp_position(seq_lens):
    return torch.clamp(seq_lens - 1, min=0).to(torch.int64)


_compiled_clamp_position = torch.compile(
    _torch_clamp_position, dynamic=True, backend=get_compiler_backend()
)

FN_MAP = {
    "jit": clamp_position_cuda,
    "torch_compile": _compiled_clamp_position,
    "torch": _torch_clamp_position,
}


@marker.parametrize("size", [2**n for n in range(4, 16)], [256, 4096])
@marker.benchmark("provider", ["jit", "torch_compile", "torch"])
def benchmark(size: int, provider: str):
    seq_lens = torch.randint(
        0, 10000, (size,), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(seq_lens,),
        graph_clone_args=(0,),
    )


if __name__ == "__main__":
    benchmark.run()
