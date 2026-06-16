import torch
import torch.nn.functional as F
from sgl_kernel import gelu_and_mul as gelu_and_mul_aot
from sgl_kernel import gelu_tanh_and_mul as gelu_tanh_and_mul_aot
from sgl_kernel import silu_and_mul as silu_and_mul_aot

from sglang.jit_kernel.activation import gelu_and_mul as gelu_and_mul_jit
from sglang.jit_kernel.activation import gelu_tanh_and_mul as gelu_tanh_and_mul_jit
from sglang.jit_kernel.activation import relu2 as relu2_jit
from sglang.jit_kernel.activation import silu_and_mul as silu_and_mul_jit
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-benchmark-1-gpu-large")


@torch.compile
def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.silu(lhs) * rhs


@torch.compile
def gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.gelu(lhs, approximate="none") * rhs


@torch.compile
def gelu_tanh_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.gelu(lhs, approximate="tanh") * rhs


OPS = {
    "silu": (silu_and_mul_aot, silu_and_mul_jit, silu_and_mul),
    "gelu": (gelu_and_mul_aot, gelu_and_mul_jit, gelu_and_mul),
    "gelu_tanh": (gelu_tanh_and_mul_aot, gelu_tanh_and_mul_jit, gelu_tanh_and_mul),
}


@marker.parametrize("op_name", ["silu", "gelu", "gelu_tanh"])
@marker.parametrize("dim", [1024, 4096, 6144, 8192], [4096])
@marker.parametrize("batch_size", [2**x for x in range(0, 15)], [8, 512])
@marker.benchmark("impl", ["aot", "jit", "torch"])
def benchmark(op_name: str, dim: int, batch_size: int, impl: str):
    x = create_random(batch_size, dim * 2)
    aot_op, jit_op, torch_op = OPS[op_name]
    fn = {"aot": aot_op, "jit": jit_op, "torch": torch_op}[impl]
    return marker.do_bench(fn, input_args=(x,))


def _make_expert_ids(num_tokens: int, skip_ratio: float) -> torch.Tensor:
    expert_ids = torch.randint(low=0, high=8, size=(num_tokens,), dtype=torch.int32)
    if skip_ratio > 0:
        skip = torch.rand(num_tokens) < skip_ratio
        expert_ids[skip] = -1
    return expert_ids


@marker.parametrize("op_name", ["silu", "gelu"])
@marker.parametrize("dim", [1024, 4096, 8192], [4096])
@marker.parametrize("batch_size", [64, 256, 1024, 4096, 16384], [1024])
@marker.parametrize("skip_ratio", [0.0, 0.25, 0.5], [0.25])
@marker.benchmark("impl", ["unfiltered", "filtered"])
def benchmark_filter(
    op_name: str, dim: int, batch_size: int, skip_ratio: float, impl: str
):
    torch.random.manual_seed(42)
    x = create_random(batch_size, dim * 2)
    jit_fn = silu_and_mul_jit if op_name == "silu" else gelu_and_mul_jit
    extra_kwargs = {}
    expert_ids = _make_expert_ids(batch_size, skip_ratio)
    if impl == "filtered":
        extra_kwargs = {"expert_ids": expert_ids.to(x.device), "expert_step": 1}

    # NOTE: get the unmasked part from `experts_ids`
    real_skip_ratio = (expert_ids == -1).sum().item() / batch_size
    effective_bytes = int(x.nbytes * (1 - real_skip_ratio) * 1.5)
    return marker.do_bench(
        jit_fn,
        input_args=(x,),
        input_kwargs=extra_kwargs,
        memory_args=None,  # x is dynamic (counted in extra_memory_footprint)
        memory_output=None,  # same, output is dynamic
        extra_memory_footprint=effective_bytes,
    )


@torch.compile
def relu2_torch(input: torch.Tensor) -> torch.Tensor:
    return F.relu(input).pow(2)


@marker.parametrize("dim", [1024, 4096, 6144, 8192], [4096])
@marker.parametrize("batch_size", [2**x for x in range(0, 15)], [8, 512])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark_unary(dim: int, batch_size: int, impl: str):
    x = create_random(batch_size, dim)
    fn = {"jit": relu2_jit, "torch": relu2_torch}[impl]
    return marker.do_bench(fn, input_args=(x,))


if __name__ == "__main__":
    benchmark.run()
    benchmark_filter.run()
    benchmark_unary.run()
