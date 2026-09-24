"""L1 preference must retain default occupancy and leave other devices untouched."""

import pathlib
import sys

import pytest
import torch

from sglang.kernels.jit.utils import load_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_launch_carveout():
    """A MaxL1-only policy loses occupancy for shared-memory-heavy kernels."""
    current = torch.cuda.current_device()
    untouched = [
        device
        for device in range(torch.cuda.device_count())
        if device != current and not torch._C._cuda_hasPrimaryContext(device)
    ]
    cases = [(smem, mode) for smem in (0, 32768) for mode in range(3)]
    module = load_jit(
        "test_launch_carveout",
        cuda_files=[str(pathlib.Path(__file__).with_name("launch_carveout.cuh"))],
        cuda_wrappers=[
            (f"run_{smem}_{mode}", f"check_launch_carveout<{smem}, {mode}>")
            for smem, mode in cases
        ],
    )
    expected = torch.arange(255, -1, -1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(expected)
    for smem, mode in cases:
        run = module[f"run_{smem}_{mode}"]
        for _ in range(2):
            output.fill_(-1)
            run(output)
            assert torch.equal(output, expected)
    for device in untouched:
        assert not torch._C._cuda_hasPrimaryContext(device)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
