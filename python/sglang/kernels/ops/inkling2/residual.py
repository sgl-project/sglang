import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

PDL = is_arch_support_pdl()


@triton.jit
def _residual_add_kernel(output, inputs, size, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), dtype=tl.float32)
    for index in tl.static_range(len(inputs)):
        result += tl.load(inputs[index] + offsets, offsets < size, other=0).to(
            tl.float32
        )
    tl.store(output + offsets, result, offsets < size)


def residual_add(*inputs: torch.Tensor) -> torch.Tensor:
    assert inputs and all(
        x.shape == inputs[0].shape and x.device == inputs[0].device for x in inputs
    )
    inputs = tuple(x.contiguous() for x in inputs)
    result = torch.empty_like(inputs[0])
    if result.numel():
        _residual_add_kernel[(triton.cdiv(result.numel(), 1024),)](
            result, inputs, result.numel(), BLOCK=1024, launch_pdl=PDL
        )
    return result
