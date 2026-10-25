import os
import select
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pynvml
import pytest
import torch

from sglang.test.test_utils import _collect_busy_gpu_reports


@pytest.mark.parametrize("external_gib", [0, 3])
def test_gpu_idle_discounts_reserved_memory_by_uuid(external_gib):
    """A container PID mismatch must not hide a separate GPU memory holder."""
    nvml = Mock()
    nvml.nvmlDeviceGetUUID.return_value = "GPU-visible"
    nvml.nvmlDeviceGetMemoryInfo.return_value.used = (4 + external_gib) << 30
    nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [
        SimpleNamespace(pid=os.getpid() + 1000000, usedGpuMemory=4 << 30),
        SimpleNamespace(pid=os.getpid() + 2000000, usedGpuMemory=external_gib << 30),
    ]
    with (
        patch.object(torch.cuda, "is_initialized", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=2),
        patch.object(
            torch.cuda,
            "get_device_properties",
            side_effect=[
                SimpleNamespace(uuid="other"),
                SimpleNamespace(uuid="visible"),
            ],
        ),
        patch.object(torch.cuda, "memory_reserved", side_effect=[9 << 30, 4 << 30]),
    ):
        reports = _collect_busy_gpu_reports(nvml, [5])
    assert bool(reports) == bool(external_gib)


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(),
    reason="requires NVIDIA GPU",
)
def test_gpu_idle_with_real_cached_allocation_and_other_process():
    """Retained CUDA cache is allowed; another process still blocks startup."""
    pynvml.nvmlInit()
    process = None
    try:
        allocation = torch.empty(3 << 30, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        del allocation
        assert torch.cuda.memory_reserved() >= 3 << 30
        handle = pynvml.nvmlDeviceGetHandleByUUID(
            "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        )
        index = pynvml.nvmlDeviceGetIndex(handle)
        original_query = pynvml.nvmlDeviceGetComputeRunningProcesses

        def host_pid_query(handle):
            return [
                SimpleNamespace(
                    pid=proc.pid + 1000000, usedGpuMemory=proc.usedGpuMemory
                )
                for proc in original_query(handle)
            ]

        with patch.object(
            pynvml, "nvmlDeviceGetComputeRunningProcesses", side_effect=host_pid_query
        ):
            assert _collect_busy_gpu_reports(pynvml, [index]) == []
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys, torch; "
                    "allocation = torch.empty(3 << 30, dtype=torch.uint8, device='cuda'); "
                    "torch.cuda.synchronize(); print('ready', flush=True); sys.stdin.read()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert select.select([process.stdout], [], [], 60)[0], (
                "CUDA child timed out"
            )
            assert process.stdout.readline().strip() == "ready"
            assert _collect_busy_gpu_reports(pynvml, [index])
    finally:
        if process is not None:
            process.terminate()
            process.communicate(timeout=30)
        torch.cuda.empty_cache()
        pynvml.nvmlShutdown()
