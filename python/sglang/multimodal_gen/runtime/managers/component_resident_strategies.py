"""
Basic Component Resident Strategy Utilities for defining usage of components, to let ComponentResidencyManager to coordinate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.component_manager import (
        ComponentUse,
        ResidencyState,
    )

logger = init_logger(__name__)


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


class ComponentResidencyStrategy:
    """Baseclass for describing how a component should be treated (regarding where its weights locates)

    e.g., a LayerwiseOffloadStrategy would override:
        enter: to prefetch some layers before DiT is used, and
        exits: to release GPU weight snapshot after DiT is used
    to achieve desired behavior

    """

    name = "resident"

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        """hook called"""
        self.enter(module)

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        """Wait for the preparation to be ready, only applicable for async device syncs"""
        pass

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        """Finish a specific component use"""
        self.exit(module)

    def prepare_after_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        """Called after a request is finished, to prepare for the upcoming request"""
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


class ResidentStrategy(ComponentResidencyStrategy):
    name = "resident"

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if use.target_dtype is not None:
            _module_to_local_device(module, dtype=use.target_dtype)


class SnapshotModuleResidency:
    """Reusable snapshot-based module residency primitive.

    This helper only knows how to:
    - keep CPU parameter/buffer snapshots,
    - prefetch a module (H2D) to the local device on a CUDA side stream
    - release a module by rebinding tensors to those snapshots,
    - track and wait for readiness events.

    It deliberately does not know about pipeline stages, phases, or model-specific
    ordering. Strategy subclasses decide when each primitive is called.
    """

    def __init__(self, *, pin_cpu_memory: bool, enable_async_prefetch: bool) -> None:
        self.pin_cpu_memory = pin_cpu_memory
        self.enable_async_prefetch = enable_async_prefetch
        self._cpu_param_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self._cpu_buffer_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self._prefetch_stream: object | None = None
        self._ready_events: dict[str, object] = {}

    @staticmethod
    def is_on_gpu(module: nn.Module | None) -> bool:
        if module is None:
            return False
        param = next(module.parameters(), None)
        return param is not None and param.device.type == "cuda"

    def is_ready(self, component_name: str) -> bool:
        return component_name in self._ready_events

    def wait_ready(self, component_name: str) -> None:
        """wait for the (H2D) stream to be ready"""
        ready_event = self._ready_events.get(component_name)
        if ready_event is None or not current_platform.is_cuda():
            return
        torch.get_device_module().current_stream().wait_event(ready_event)

    def record_ready(self, component_name: str, module: nn.Module | None) -> None:
        if not current_platform.is_cuda():
            self._ready_events.pop(component_name, None)
            return
        if not self.is_on_gpu(module):
            self._ready_events.pop(component_name, None)
            return
        event = torch.get_device_module().Event()
        event.record(torch.get_device_module().current_stream())
        self._ready_events[component_name] = event

    @staticmethod
    def _clone_cpu_tensor_snapshot(
        tensor: torch.Tensor, *, pin_memory: bool
    ) -> torch.Tensor:
        snapshot = tensor.detach()
        if snapshot.device.type == "cpu":
            if pin_memory and not snapshot.is_pinned():
                return snapshot.pin_memory()
            return snapshot

        cpu_tensor = snapshot.to("cpu")
        if pin_memory:
            return cpu_tensor.pin_memory()
        return cpu_tensor

    def _should_pin_memory(self) -> bool:
        return bool(self.pin_cpu_memory and torch.get_device_module().is_available())

    def capture(self, component_name: str, module: nn.Module) -> None:
        """Capture a CPU snapshot for a component"""
        if component_name in self._cpu_param_snapshots:
            return

        pin_memory = self._should_pin_memory()
        self._cpu_param_snapshots[component_name] = {
            name: self._clone_cpu_tensor_snapshot(param.data, pin_memory=pin_memory)
            for name, param in module.named_parameters()
        }
        self._cpu_buffer_snapshots[component_name] = {
            name: self._clone_cpu_tensor_snapshot(buffer.data, pin_memory=pin_memory)
            for name, buffer in module.named_buffers()
        }

    def release_to_snapshot(
        self,
        component_name: str,
        module: nn.Module,
        *,
        copy_runtime_buffers: bool = False,
    ) -> None:
        """Release CUDA storages by rebinding tensors to cached CPU snapshots.

        This does not call `module.to("cpu")`. Instead, parameter and buffer
        storages are rebound to pre-captured CPU tensors so CUDA storages can be
        released by the allocator without an explicit D2H transfer.
        """
        param_snapshots = self._cpu_param_snapshots.get(component_name)
        buffer_snapshots = self._cpu_buffer_snapshots.get(component_name)
        if param_snapshots is None or buffer_snapshots is None:
            module.to("cpu")
            self._ready_events.pop(component_name, None)
            return

        pin_memory = self._should_pin_memory()
        for name, param in module.named_parameters():
            snapshot = param_snapshots.get(name)
            if snapshot is None:
                snapshot = self._clone_cpu_tensor_snapshot(
                    param.data, pin_memory=pin_memory
                )
                param_snapshots[name] = snapshot
            param.data = snapshot

        for name, buffer in module.named_buffers():
            snapshot = buffer_snapshots.get(name)
            if snapshot is None:
                snapshot = self._clone_cpu_tensor_snapshot(
                    buffer.data, pin_memory=pin_memory
                )
                buffer_snapshots[name] = snapshot
            if copy_runtime_buffers:
                # Preserve runtime-updated buffers (e.g., lazily built caches) when
                # releasing back to CPU snapshots.
                if buffer.device.type == "cuda":
                    snapshot.copy_(
                        buffer.detach().to(device="cpu", dtype=snapshot.dtype)
                    )
                elif buffer.device.type == "cpu":
                    snapshot.copy_(buffer.detach().to(dtype=snapshot.dtype))
            buffer.data = snapshot

        self._ready_events.pop(component_name, None)

    def _supports_async_prefetch(self) -> bool:
        return self.enable_async_prefetch and current_platform.is_cuda()

    def _get_prefetch_stream(self):
        """returns a stream is async-prefetch is enabled"""
        if not self._supports_async_prefetch():
            return None
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.get_device_module().Stream(
                device=get_local_torch_device()
            )
        return self._prefetch_stream

    def prefetch_to_device(self, component_name: str, module: nn.Module | None) -> None:
        if module is None:
            self._ready_events.pop(component_name, None)
            return
        prefetch_stream = self._get_prefetch_stream()
        if prefetch_stream is None:
            # if the async prefetching is disabled
            module.to(get_local_torch_device(), non_blocking=True)
            self.record_ready(component_name, module)
            return
        with torch.get_device_module().stream(prefetch_stream):
            module.to(get_local_torch_device(), non_blocking=True)
            event = torch.get_device_module().Event()
            event.record(prefetch_stream)
        self._ready_events[component_name] = event


class SnapshotStrategy(ComponentResidencyStrategy):
    """Snapshot residency: async H2D before use and light snapshot release after use."""

    name = "snapshot"

    def __init__(
        self,
        *,
        pin_cpu_memory: bool,
        enable_async_prefetch: bool,
        copy_runtime_buffers_on_release: bool = False,
    ) -> None:
        self._snapshot_residency = SnapshotModuleResidency(
            pin_cpu_memory=pin_cpu_memory,
            enable_async_prefetch=enable_async_prefetch,
        )
        self._copy_runtime_buffers_on_release = copy_runtime_buffers_on_release

    def capture(self, component_name: str, module: nn.Module) -> None:
        self._snapshot_residency.capture(component_name, module)

    def is_ready(self, component_name: str) -> bool:
        return self._snapshot_residency.is_ready(component_name)

    def record_ready(self, component_name: str, module: nn.Module | None) -> None:
        self._snapshot_residency.record_ready(component_name, module)

    def prefetch_component(self, component_name: str, module: nn.Module | None) -> None:
        if SnapshotModuleResidency.is_on_gpu(module):
            self._snapshot_residency.record_ready(component_name, module)
            return
        self._snapshot_residency.prefetch_to_device(component_name, module)

    def wait_component_ready(self, component_name: str) -> None:
        self._snapshot_residency.wait_ready(component_name)

    def release_component(self, component_name: str, module: nn.Module) -> None:
        self._snapshot_residency.release_to_snapshot(
            component_name,
            module,
            copy_runtime_buffers=self._copy_runtime_buffers_on_release,
        )

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.prefetch_component(use.component_name, module)

    def wait_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.wait_component_ready(use.component_name)

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.release_component(use.component_name, module)

    def prepare_after_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.prepare_for_use(module, use, state)


class VanillaD2HStrategy(ComponentResidencyStrategy):
    """A strategy that performs native torch D2H and H2D for a component"""

    name = "vanilla"

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

    def enter(self, module: nn.Module) -> None:
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            _module_to_local_device(module)

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
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


class LayerwiseOffloadStrategy(ComponentResidencyStrategy):
    """A wrapper around LayerwiseOffloadManager to fit in a ComponentResidencyStrategy"""

    name = "layerwise"

    def enter(self, module: nn.Module) -> None:
        if isinstance(module, OffloadableDiTMixin):
            module.prepare_for_next_req()

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        if not isinstance(module, OffloadableDiTMixin):
            return
        for manager in module.layerwise_offload_managers:
            manager.release_all()

    def prepare_after_request(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.prepare_for_use(module, use, state)
