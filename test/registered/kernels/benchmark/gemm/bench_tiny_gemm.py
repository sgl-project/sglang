import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_empty, create_random
from sglang.kernels.ops.gemm.tiny_gemm import tiny_k_gemm_bf16, tiny_n_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


def _tiny_impl(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    if w.shape[1] < w.shape[0]:
        tiny_k_gemm_bf16(x, w, out=out)
    else:
        tiny_n_gemm_bf16(x, w, out=out)


def _torch_impl(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    torch.mm(x, w.t(), out=out)


FN_MAP = {
    "tiny": _tiny_impl,
    "torch": _torch_impl,
}


@marker.parametrize(
    "n,k",
    [(144, 7168), (896, 7168), (256, 4096), (1536, 128)],
    ci_vals=[(144, 7168)],
)
@marker.parametrize("m", list(range(1, 17)), [1, 8])
@marker.benchmark("impl", ["tiny", "torch"])
def benchmark(n: int, k: int, m: int, impl: str):
    x = create_random(m, k)
    w = create_random(n, k)
    out = create_empty(m, n)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, w, out),
        memory_args=(x, w),
        memory_output=(out,),
    )


if __name__ == "__main__":
    benchmark.run()
