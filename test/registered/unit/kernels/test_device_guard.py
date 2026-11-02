"""``device_guard`` enters a tensor's CUDA device only when it is not the current one."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import device_guard
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_current_device_enters_no_guard(monkeypatch):
    x = torch.empty(1, device=f"cuda:{torch.cuda.current_device()}")

    def entered(device):
        raise AssertionError(f"guard entered for {device}")

    monkeypatch.setattr(torch.cuda, "device", entered)
    with device_guard(x):
        assert torch.cuda.current_device() == x.get_device()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_other_device_is_entered_and_restored():
    torch.cuda.set_device(0)
    x = torch.empty(1, device="cuda:1")
    with device_guard(x):
        assert torch.cuda.current_device() == 1
    assert torch.cuda.current_device() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
