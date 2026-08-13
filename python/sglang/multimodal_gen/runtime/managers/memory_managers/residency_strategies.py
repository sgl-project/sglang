"""Runtime strategies for pipeline component residency."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    is_fsdp_managed_module,
)
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


class ResidencyStrategy:
    """Controls component placement around each declared pipeline use."""

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.enter(module)

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
        self.exit(module)

    def prepare_after_request(
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
        if preferred:
            self.prepare_for_use(module, use, state)
            self.wait_for_use(module, use, state)
        else:
            self.finish_use(module, use, state)

    def prefetch_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> bool:
        self.prepare_for_use(module, use, state)
        return True

    def enter(self, module: nn.Module) -> None:
        pass

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        pass


class ResidentStrategy(ResidencyStrategy):
    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if is_fsdp_managed_module(module):
            return
        _module_to_local_device(module, dtype=use.target_dtype)


class ComponentOffloadStrategy(ResidencyStrategy):
    """Move a complete non-FSDP component between CPU and the local device.

    FSDP-managed modules keep their load-time placement semantics; this strategy
    must not issue an additional module-wide ``to()`` on top of FSDP hooks.
    """

    def __init__(self) -> None:
        self._prefetch_stream: object | None = None
        self._ready_events: dict[str, object] = {}

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if is_fsdp_managed_module(module):
            return
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
        if is_fsdp_managed_module(module):
            return False
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

    def enter(self, module: nn.Module) -> None:
        if is_fsdp_managed_module(module):
            return
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            _module_to_local_device(module)

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        if is_fsdp_managed_module(module):
            return
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cuda":
            module.to("cpu", non_blocking=True)

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.wait_for_use(module, use, state)
        self.exit(module)
        self._ready_events.pop(use.component_name, None)

    def prepare_after_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.prefetch_for_use(module, use, state)

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
        if not preferred:
            self.finish_use(module, use, state)


class LayerwiseOffloadStrategy(ResidencyStrategy):
    """Delegate component placement to its layerwise offload managers."""

    def enter(self, module: nn.Module) -> None:
        cast(LayerwiseOffloadableModuleMixin, module).prepare_for_next_req()

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        layerwise_module = cast(LayerwiseOffloadableModuleMixin, module)
        for manager in layerwise_module.layerwise_offload_managers:
            manager.release_all()

    def prepare_after_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.prepare_for_use(module, use, state)
