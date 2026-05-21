import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
)

from .utils import make_name


@cache_once
def _jit_hisparse_transfer_module():
    return load_jit(
        make_name("hisparse_transfer"),
        cuda_files=["deepseek_v4/hisparse_transfer.cuh"],
        cuda_wrappers=[("hisparse_transfer", "hisparse_transfer")],
    )


def hisparse_offload_to_host(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    module = _jit_hisparse_transfer_module()
    module.hisparse_transfer(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)
