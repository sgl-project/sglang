import pytest
import torch
from sgl_kernel.elementwise import copy_to_gpu_no_ce as aot_copy_to_gpu_no_ce

from sglang.jit_kernel.copy import copy_to_gpu_no_ce as jit_copy_to_gpu_no_ce


@pytest.mark.parametrize("size", [64, 72])
def test_copy_to_gpu_no_ce(size):
    tensor_cpu = torch.randint(0, 1_000_000, (size,), dtype=torch.int32, device="cpu")

    # JIT output
    jit_out = torch.empty(size, dtype=torch.int32, device="cuda")
    jit_copy_to_gpu_no_ce(tensor_cpu, jit_out)

    # AOT output (reference)
    aot_out = torch.empty(size, dtype=torch.int32, device="cuda")
    aot_copy_to_gpu_no_ce(tensor_cpu, aot_out)

    assert torch.all(jit_out == aot_out), f"JIT and AOT outputs differ for size={size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
