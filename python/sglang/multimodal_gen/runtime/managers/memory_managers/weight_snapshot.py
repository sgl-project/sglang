"""CPU weight ownership while a snapshot-offloaded component runs on device."""

import torch
from torch import nn


def weight_snapshot(module: nn.Module) -> dict[str, torch.Tensor] | None:
    return module.__dict__.get("_offload_weight_snapshot")


def capture_weight_snapshot(module: nn.Module) -> None:
    # CPU tensors keep their existing storage, including mmap and tied views
    module._offload_weight_snapshot = {
        name: parameter.detach().to("cpu")
        for name, parameter in module.named_parameters()
    }


def restore_weight_snapshot(module: nn.Module) -> bool:
    """Restore CPU parameters before offload or mutation, preserving live buffers."""
    snapshot = weight_snapshot(module)
    if snapshot is None:
        return False
    # drain prefetch reads of the host weights before a writer can mutate them
    torch.get_device_module().synchronize()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.data = snapshot[name]
        # buffers may change during forward and must not be restored from a snapshot
        module.to("cpu")
    del module._offload_weight_snapshot
    return True
