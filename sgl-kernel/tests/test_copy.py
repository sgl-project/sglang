import pytest
import sgl_kernel
import torch
from sgl_kernel.elementwise import copy_to_gpu_no_ce


@pytest.mark.parametrize("size", [64, 72])
def test_copy_to_gpu_no_ce(size):
    tensor_cpu = torch.randint(0, 1000000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    assert torch.all(tensor_cpu.cuda() == tensor_gpu)


if __name__ == "__main__":
    pytest.main([__file__])
