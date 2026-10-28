import pytest
import torch

from sglang.kernels.ops.memory.gpu_tensor_hash import gpu_tensor_hash
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_small_tensor_hash_is_unchanged():
    tensor = torch.arange(1 << 20, dtype=torch.int32, device="cuda")
    assert gpu_tensor_hash(tensor) == 0x217789F197E070FE


def test_tensor_over_8_gib():
    nbytes = (8 << 30) + 4 * 8192  # one tile past 2**31 uint32 words
    if torch.cuda.mem_get_info()[0] < nbytes + (1 << 30):
        pytest.skip("needs about 9 GiB of free GPU memory")
    tensor = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    before = gpu_tensor_hash(tensor)
    tensor[-1] = 1
    assert gpu_tensor_hash(tensor) != before


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
