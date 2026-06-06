import torch
import torch.nn.functional as F
from sgl_kernel import gelu_and_mul as gelu_and_mul_aot
from sgl_kernel import gelu_tanh_and_mul as gelu_tanh_and_mul_aot
from sgl_kernel import silu_and_mul as silu_and_mul_aot

from sglang.jit_kernel.activation import gelu_and_mul as gelu_and_mul_jit
from sglang.jit_kernel.activation import gelu_tanh_and_mul as gelu_tanh_and_mul_jit
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
@marker.kernel("impl", ["aot", "jit", "torch"], reference="torch")
class ActivationMul:
    def inputs(self, op_name: str, dim: int, batch_size: int):
        return marker.io(create_random(batch_size, dim * 2), op_name=op_name)

    def run(self, impl: str, x: torch.Tensor, op_name: str):
        aot_op, jit_op, torch_op = OPS[op_name]
        fn = {"aot": aot_op, "jit": jit_op, "torch": torch_op}[impl]
        return fn(x)


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
@marker.kernel(
    "impl",
    ["unfiltered", "filtered"],
    correctness=False,
    reason="filtered zeroes expert_ids==-1 rows; output differs from unfiltered by design",
)
class ActivationMulFilter:
    def inputs(self, op_name: str, dim: int, batch_size: int, skip_ratio: float):
        x = create_random(batch_size, dim * 2)
        expert_ids = _make_expert_ids(batch_size, skip_ratio).to(x.device)
        return marker.io(x, op_name=op_name, expert_ids=expert_ids)

    def run(self, impl: str, x: torch.Tensor, op_name: str, expert_ids: torch.Tensor):
        jit_fn = silu_and_mul_jit if op_name == "silu" else gelu_and_mul_jit
        if impl == "filtered":
            return jit_fn(x, expert_ids=expert_ids, expert_step=1)
        return jit_fn(x)

    def bench_kwargs(
        self, impl: str, x: torch.Tensor, op_name: str, expert_ids: torch.Tensor
    ):
        real_skip_ratio = (expert_ids == -1).sum().item() / expert_ids.numel()
        effective_bytes = int(x.nbytes * (1 - real_skip_ratio) * 1.5)
        return {
            "memory_args": None,
            "memory_output": None,
            "extra_memory_footprint": effective_bytes,
        }


if __name__ == "__main__":
    marker.main(ActivationMul, ActivationMulFilter)
