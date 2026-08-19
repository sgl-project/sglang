"""Runtime strategies used by the component residency manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
        ComponentUse,
        ResidencyState,
    )


def _module_to_local_device(
    module: nn.Module, *, dtype: torch.dtype | None = None
) -> None:
    device = get_local_torch_device()
    tensor = _module_reference_tensor(module)
    if tensor is not None and tensor.device == device:
        if dtype is None or tensor.dtype == dtype:
            return
    if dtype is None:
        module.to(device, non_blocking=True)
    else:
        module.to(device, dtype=dtype, non_blocking=True)


def _module_reference_tensor(module: nn.Module) -> torch.Tensor | None:
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return tensor


def _module_ready_on_local_device(
    module: nn.Module, *, dtype: torch.dtype | None = None
) -> bool:
    tensor = _module_reference_tensor(module)
    if tensor is None:
        return True
    if tensor.device != get_local_torch_device():
        return False
    return dtype is None or tensor.dtype == dtype


def is_fsdp_managed_module(module: nn.Module) -> bool:
    return isinstance(module, FSDPModule)


class ComponentResidencyStrategy:
    """Controls one component's device placement around declared use intervals."""

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        pass

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        pass

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        pass

    def finish_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
        *,
        preferred: bool,
    ) -> None:
        if not preferred:
            self.finish_use(module, use, state)

    def prefetch_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> bool:
        self.prepare_for_use(module, use, state)
        return True


class ResidentStrategy(ComponentResidencyStrategy):
    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if is_fsdp_managed_module(module):
            return
        _module_to_local_device(module, dtype=use.target_dtype)


class ComponentOffloadStrategy(ComponentResidencyStrategy):
    """Move a complete component between CPU and device around each use."""

    def __init__(self) -> None:
        self._prefetch_stream: object | None = None
        self._ready_events: dict[str, object] = {}

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        _module_to_local_device(module, dtype=use.target_dtype)

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        ready_event = self._ready_events.get(use.component_name)
        if ready_event is None or not current_platform.is_cuda():
            return
        torch.get_device_module().current_stream().wait_event(ready_event)

    def prefetch_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> bool:
        if not current_platform.is_cuda():
            self.prepare_for_use(module, use, state)
            return True
        if _module_ready_on_local_device(module, dtype=use.target_dtype):
            return True
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.get_device_module().Stream(
                device=get_local_torch_device()
            )
        with torch.get_device_module().stream(self._prefetch_stream):
            _module_to_local_device(module, dtype=use.target_dtype)
            event = torch.get_device_module().Event()
            event.record(self._prefetch_stream)
        self._ready_events[use.component_name] = event
        return True

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.wait_for_use(module, use, state)
        tensor = _module_reference_tensor(module)
        if tensor is not None and tensor.device.type != "cpu":
            module.to("cpu", non_blocking=True)
        self._ready_events.pop(use.component_name, None)

    def finish_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
        *,
        preferred: bool,
    ) -> None:
        if preferred and state.batch_is_warmup:
            self.prepare_for_use(module, use, state)
            self.wait_for_use(module, use, state)
            return
        self.finish_use(module, use, state)


class LayerwiseOffloadStrategy(ComponentResidencyStrategy):
    """Run the lifecycle of an already configured layerwise component."""

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if isinstance(module, LayerwiseOffloadableModuleMixin):
            module.prepare_for_next_req()

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            return
        for manager in module.layerwise_offload_managers:
            manager.release_all()

    def finish_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
        *,
        preferred: bool,
    ) -> None:
        if preferred:
            self.prepare_for_use(module, use, state)
        else:
            self.finish_use(module, use, state)
