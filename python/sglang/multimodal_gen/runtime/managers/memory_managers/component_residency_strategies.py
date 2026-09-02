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

    def supports_auto_residency(self) -> bool:
        """Whether runtime placement may change underneath this strategy."""
        return False

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
        try:
            _module_to_local_device(module, dtype=use.target_dtype)
        except BaseException:
            self._restore_cpu_after_failed_transfer(module)
            raise

    def _restore_cpu_after_failed_transfer(
        self, module: nn.Module, *, stream: object | None = None
    ) -> None:
        """Undo a partial ``Module.to(device)`` after allocation failure."""
        if stream is None:
            module.to("cpu", non_blocking=False)
        else:
            with torch.get_device_module().stream(stream):
                module.to("cpu", non_blocking=False)
            stream.synchronize()
        if current_platform.is_cuda():
            torch.get_device_module().empty_cache()

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
        try:
            with torch.get_device_module().stream(self._prefetch_stream):
                _module_to_local_device(module, dtype=use.target_dtype)
                event = torch.get_device_module().Event()
                event.record(self._prefetch_stream)
        except BaseException:
            self._ready_events.pop(use.component_name, None)
            self._restore_cpu_after_failed_transfer(
                module, stream=self._prefetch_stream
            )
            raise
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
            # A non-blocking device->host move lands in pinned host memory the
            # size of the component. On a shared pool that pins a second copy
            # of the weights next to the device copy still being read from
            # -- a 57 GiB DiT took 43 GiB of shared memory in under a minute
            # and exhausted a GB10. Take the synchronous, pageable path there.
            module.to(
                "cpu",
                non_blocking=not current_platform.device_shares_host_memory(),
            )
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
            # MPS layerwise components retain checkpoint-backed CPU weights and
            # synchronously materialize one layer at a time. Moving the whole
            # module here would defeat that bounded-residency contract.
            if current_platform.is_mps():
                if module.mps_stream_non_layer_weights:
                    return
                _module_to_local_device(module, dtype=use.target_dtype)
                return
            module.restore_non_layer_weights()
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
        # The layers are gone; the rest of this component is dead weight on the
        # device until it is used again, and the stage that follows may be the
        # one that needs the room.
        module.park_non_layer_weights()
        if current_platform.is_mps():
            torch.mps.synchronize()
            module.restore_mps_cpu_non_layer_weights()
            torch.mps.empty_cache()
        elif (
            current_platform.is_cuda() and current_platform.device_shares_host_memory()
        ):
            # The stage's streamed layer windows are freed but still reserved
            # by the caching allocator. On a shared pool that reserve is host
            # memory the next stage's mapping needs as page cache; hand it back.
            empty_cache = getattr(torch.get_device_module(), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()
            # And this component's own pages are now the least valuable in the
            # cache until its next stage; say so before the next phase evicts.
            for manager in module.layerwise_offload_managers:
                advise_cold = getattr(manager, "advise_mapped_pages_cold", None)
                if advise_cold is not None:
                    advise_cold()

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
