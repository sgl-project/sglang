"""Reserve ROCm runtime memory outside torch's allocator."""

import torch

from sglang.srt.environ import envs


def reserve_runtime_memory(*, device: str, gpu_id: int) -> None:
    # HSA allocates a new hardware queue's scratch outside torch and aborts the process
    # (HSA_STATUS_ERROR_OUT_OF_RESOURCES) when none is free, so cap torch's share.
    reserve_gb = envs.SGLANG_ROCM_RUNTIME_RESERVE_GB.get()
    if not (device == "cuda" and reserve_gb > 0):
        return
    total = torch.cuda.get_device_properties(gpu_id).total_memory
    torch.cuda.set_per_process_memory_fraction(
        1.0 - reserve_gb * (1 << 30) / total, gpu_id
    )
