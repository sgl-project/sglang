from importlib.util import find_spec

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def jit_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.sampling.softmax_jit import softmax

    return softmax(logits, temperatures, out=logits)


def torch_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits / temperatures.view(-1, 1), dim=-1)


def cake_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    # FlashInfer only routes SM100 / SM103 to its Cake Blackwell softmax
    if find_spec("flashinfer.jit.cake_blackwell_softmax") is None:
        marker.skip("this FlashInfer does not ship the Cake Blackwell softmax")
    from flashinfer.sampling import softmax

    return softmax(logits, temperature=temperatures)


FN_MAP = {"jit": jit_softmax, "torch": torch_softmax, "cake": cake_softmax}


# fp32 logits with one temperature per row: what `Sampler.forward` hands the softmax
@marker.parametrize("vocab_size", [32000, 129280, 163840, 201088, 262144], [163840])
@marker.parametrize("batch_size", [2**n for n in range(0, 13)], [1, 64, 1024])
@marker.benchmark("impl", ["jit", "cake", "torch"])
def benchmark(vocab_size: int, batch_size: int, impl: str):
    logits = create_random(batch_size, vocab_size, dtype=torch.float32)
    temperatures = torch.linspace(0.5, 1.5, batch_size, device=logits.device)
    return marker.do_bench(FN_MAP[impl], input_args=(logits, temperatures))


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        print("The JIT softmax requires an SM100-family CUDA GPU. Skipping.")
    else:
        benchmark.run()
