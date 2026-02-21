import pytest
import torch

from sglang.jit_kernel.copy import copy_to_gpu_no_ce


@pytest.mark.parametrize("size", [64, 72])
def test_copy_to_gpu_no_ce(size):
    tensor_cpu = torch.randint(0, 1_000_000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty(size, dtype=torch.int32, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    assert torch.all(tensor_cpu.cuda() == tensor_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
