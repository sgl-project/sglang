from typing import Optional, Tuple

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def vllm_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        return sglang_scaled_fp8_quant(input, scale)
    return ops.scaled_fp8_quant(input, scale)


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


FN_MAP = {
    "vllm": vllm_scaled_fp8_quant,
    "sglang": sglang_scaled_fp8_quant,
}

LINE_VALS = ["vllm", "sglang"] if VLLM_AVAILABLE else ["sglang"]


@marker.parametrize("element_count", [2**n for n in range(10, 20)], [16384])
@marker.benchmark("provider", LINE_VALS)
def benchmark(element_count: int, provider: str):
    dtype = torch.float16
    x = torch.randn(element_count, 4096, device=DEFAULT_DEVICE, dtype=dtype)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(x,),
        graph_clone_args=(0,),  # x is read
        memory_args=(x,),
        # returns (output_fp8, scale); counts both
    )


if __name__ == "__main__":
    benchmark.run()
