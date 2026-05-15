import torch


def get_cuda_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get a CUDA view of a CPU tensor using Unified Virtual Addressing (UVA).
    """
    assert cpu_tensor.is_pinned(), "CPU tensor must be pinned"
    return torch.ops.sgl_kernel.get_cuda_view_from_cpu_tensor(cpu_tensor)
