"""CPU weight ownership while a snapshot-offloaded component runs on device."""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
    host_copies_would_not_fit,
)


def weight_snapshot(module: nn.Module) -> dict[str, torch.Tensor] | None:
    return module.__dict__.get("_offload_weight_snapshot")


def capture_weight_snapshot(
    module: nn.Module,
    *,
    pin_budget: HostPinBudget | None = None,
    component_name: str = "",
) -> None:
    parameters = dict(module.named_parameters())
    snapshot = {
        name: parameter.detach().to("cpu") for name, parameter in parameters.items()
    }
    if pin_budget is not None:
        # pin each storage once, preserving tied views and releasing the original
        # CPU storage as we go rather than staging another complete model copy
        storage_names: dict[int, list[str]] = {}
        for name, tensor in snapshot.items():
            storage_names.setdefault(tensor.untyped_storage().data_ptr(), []).append(
                name
            )
        for names in storage_names.values():
            storage = snapshot[names[0]].untyped_storage()
            size = storage.nbytes()
            if storage.is_pinned() or not size or size > pin_budget.spendable_bytes:
                continue
            if host_copies_would_not_fit(size):
                continue
            if not pin_budget.request(component_name=component_name, weight_bytes=size):
                continue
            try:
                pinned_storage = storage.pin_memory()
            except Exception:
                pin_budget.release(size)
                raise
            # the lease outlives strategy rebuilds, snapshots and LoRA backups;
            # only the last tensor releasing this storage returns its allowance
            pin_budget.track_storage(pinned_storage)
            for name in names:
                tensor = snapshot[name]
                pinned = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
                    pinned_storage,
                    tensor.storage_offset(),
                    tensor.shape,
                    tensor.stride(),
                )
                snapshot[name] = pinned
                if parameters[name].device.type == "cpu":
                    parameters[name].data = pinned
    module._offload_weight_snapshot = snapshot


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
