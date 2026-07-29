import sys

import pytest
import sgl_kernel
import torch
from sgl_kernel.elementwise import copy_to_gpu_no_ce


@pytest.mark.parametrize("size", [16, 17, 32, 64, 72, 512])
def test_copy_to_gpu_no_ce(size):
    """Copies both specialized and fallback local-expert vector sizes."""
    tensor_cpu = torch.randint(0, 1000000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    assert torch.all(tensor_cpu.cuda() == tensor_gpu)


def test_copy_to_gpu_no_ce_rejects_oversized_input():
    """Rejects inputs that cannot fit in the fallback kernel's launch parameters."""
    size = 513
    tensor_cpu = torch.randint(0, 1000000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")

    with pytest.raises(RuntimeError, match="supports at most 512 elements"):
        copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
