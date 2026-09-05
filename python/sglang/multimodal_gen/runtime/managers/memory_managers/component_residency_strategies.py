"""Runtime strategies used by the component residency manager."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import MappedRegions
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    to_local_tensor,
    wrap_for_target,
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


def _pinned_like(host: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    # Stride-preserving: a plain torch.empty would force contiguous and
    # silently drop channels_last_3d VAE layouts (see
    # memory_occupation_controller._module_to_pinned_cpu).
    pinned = torch.empty_strided(
        size=host.shape,
        stride=host.stride(),
        dtype=dtype if dtype is not None else host.dtype,
        device="cpu",
        pin_memory=True,
    )
    pinned.copy_(host)
    return pinned


class ComponentHostStore:
    """Host-resident weights of one component-offloaded module.

    The layerwise pattern at component granularity: while at rest the
    module's parameters hold a shared (1,) device placeholder, the real
    weights live here -- checkpoint-mmap views kept as-is, pinned copies for
    as many tensors as the pin budget grants, existing pageable storage
    otherwise -- and a swap binds device copies or the placeholder back.
    Weights are immutable during inference; the writers (weight refit, LoRA
    merge/unmerge) go through update_host_weights or begin/end_host_update.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        pin_budget: HostPinBudget,
        component_name: str,
    ) -> None:
        self.loaded = False
        self._module = module
        self._device = device
        self._targets: dict[str, torch.Tensor] = {}
        self._host: dict[str, torch.Tensor] = {}
        self._mapped: set[str] = set()
        self._placeholders: dict[torch.dtype, torch.Tensor] = {}
        with torch.inference_mode(False), torch.no_grad():
            self._capture(module, pin_budget=pin_budget, component_name=component_name)
            self.release()
        # seeded like retarget_dtype stamps it: from the floating weights
        self.reference_dtype: torch.dtype | None = next(
            (h.dtype for h in self._host.values() if h.is_floating_point()), None
        )

    def _capture(
        self,
        module: nn.Module,
        *,
        pin_budget: HostPinBudget,
        component_name: str,
    ) -> None:
        regions = MappedRegions()
        for name, target in chain(module.named_parameters(), module.named_buffers()):
            local = to_local_tensor(target)
            # a module captured warm pays one D2H here
            host = (local if local.device.type == "cpu" else local.to("cpu")).detach()
            self._targets[name] = target
            self._host[name] = host
            if regions.holds(host):
                self._mapped.add(name)

        # Two kinds must keep their storage and are never pinned: a mapped
        # view (a pinned copy is the committed memory the mapping avoids) and
        # a storage shared by several tensors (copies would sever the tie).
        storage_users: dict[int, int] = {}
        for host in self._host.values():
            if host.numel() > 0:
                pointer = host.untyped_storage().data_ptr()
                storage_users[pointer] = storage_users.get(pointer, 0) + 1
        chosen: list[str] = []
        chosen_bytes = 0
        spendable = pin_budget.spendable_bytes
        for name, host in self._host.items():
            if (
                host.numel() == 0
                or host.is_pinned()
                or name in self._mapped
                or storage_users[host.untyped_storage().data_ptr()] > 1
            ):
                continue
            nbytes = host.numel() * host.element_size()
            if chosen_bytes + nbytes <= spendable:
                chosen.append(name)
                chosen_bytes += nbytes
        if (
            chosen
            and torch.get_device_module().is_available()
            and pin_budget.request(
                component_name=component_name, weight_bytes=chosen_bytes
            )
        ):
            for name in chosen:
                self._host[name] = _pinned_like(self._host[name])

    def _placeholder(self, target: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        placeholder = self._placeholders.get(dtype)
        if placeholder is None:
            placeholder = torch.empty((1,), dtype=dtype, device=self._device)
            self._placeholders[dtype] = placeholder
        return wrap_for_target(target, placeholder)

    def load(self) -> None:
        """Bind device copies of the host store; runs under the caller's
        stream context, asynchronously where the source is pinned."""
        with torch.inference_mode(False), torch.no_grad():
            for name, target in self._targets.items():
                host = self._host[name]
                device_tensor = torch.empty_strided(
                    size=host.shape,
                    stride=host.stride(),
                    dtype=host.dtype,
                    device=self._device,
                )
                device_tensor.copy_(host, non_blocking=host.is_pinned())
                target.data = wrap_for_target(target, device_tensor)
        self.loaded = True

    def release(self) -> None:
        """Bind the shared placeholders; the host store keeps the weights."""
        with torch.inference_mode(False), torch.no_grad():
            for name, target in self._targets.items():
                target.data = self._placeholder(target, self._host[name].dtype)
        self.loaded = False

    def retarget_dtype(self, dtype: torch.dtype) -> None:
        """Convert the floating host weights to `dtype`, once; later swaps
        are plain byte copies again."""
        element_size = torch.empty((), dtype=dtype).element_size()
        with torch.inference_mode(False), torch.no_grad():
            for name, host in self._host.items():
                if not host.is_floating_point() or host.dtype == dtype:
                    continue
                if host.is_pinned() and element_size <= host.element_size():
                    # a wider dtype would outgrow the pin-budget booking
                    self._host[name] = _pinned_like(host, dtype=dtype)
                else:
                    self._host[name] = host.to(dtype)
                    self._mapped.discard(name)
        self.reference_dtype = dtype

    def update_host_weights(self, weight_dict: dict) -> set:
        """Write new weights into the host store; the layerwise
        update_cpu_weights contract at component granularity."""
        updated: set[str] = set()
        with torch.inference_mode(False), torch.no_grad():
            for name, loaded_weight in weight_dict.items():
                host = self._host.get(name)
                if host is None:
                    continue
                local = to_local_tensor(loaded_weight)
                if tuple(host.shape) != tuple(local.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"expected={tuple(host.shape)}, "
                        f"loaded={tuple(local.shape)}"
                    )
                if name in self._mapped:
                    # the mapping is a read-only view of the checkpoint; own
                    # the storage from here on
                    self._host[name] = (
                        local.detach().to(device="cpu", dtype=host.dtype).contiguous()
                    )
                    self._mapped.discard(name)
                else:
                    host.copy_(local)
                if self.loaded:
                    to_local_tensor(self._targets[name]).copy_(local)
                updated.add(name)
        return updated

    def begin_host_update(self) -> None:
        """Bind the host store into the module so in-place weight updates
        (LoRA merge/unmerge) write it directly."""
        with torch.inference_mode(False), torch.no_grad():
            if self.loaded:
                # adopt device-side mutations before dropping the device copy
                for name, target in self._targets.items():
                    current = to_local_tensor(target)
                    if name in self._mapped:
                        self._host[name] = current.detach().to("cpu")
                        self._mapped.discard(name)
                    else:
                        self._host[name].copy_(current)
            for name, target in self._targets.items():
                target.data = wrap_for_target(target, self._host[name])
        self.loaded = False

    def end_host_update(self) -> None:
        """Adopt tensors an update replaced and park on the placeholders.

        Targets are re-resolved from the module: an update may have replaced
        whole submodules (LoRA layer conversion), not just tensor storage.
        """
        with torch.inference_mode(False), torch.no_grad():
            self._targets = dict(
                chain(self._module.named_parameters(), self._module.named_buffers())
            )
            for name, target in self._targets.items():
                local = to_local_tensor(target)
                if local.device.type != "cpu":
                    continue
                if name not in self._host or local is not self._host[name]:
                    self._host[name] = local.detach()
                    self._mapped.discard(name)
            for name in list(self._host):
                if name not in self._targets:
                    del self._host[name]
                    self._mapped.discard(name)
        self.release()

    def iter_cpu_weights(self):
        yield from self._host.items()


def component_offload_host_store(module: nn.Module) -> ComponentHostStore | None:
    """The module's host store, or None when it is not component-offloaded."""
    store = getattr(module, "component_offload_host_store", None)
    return store if isinstance(store, ComponentHostStore) else None


class ComponentOffloadStrategy(ComponentResidencyStrategy):
    """Swap a complete component between its host store and the device.

    Swap-in copies from the retained host weights (asynchronously when they
    are pinned); swap-out rebinds the parameters to a shared device
    placeholder instead of copying device weights back to the host, so there
    is no D2H traffic and a checkpoint-mapped component stays mapped across
    uses.
    """

    def __init__(
        self,
        *,
        component_name: str,
        pin_budget: HostPinBudget,
    ) -> None:
        self._component_name = component_name
        self._prefetch_stream: object | None = None
        self._ready_events: dict[str, object] = {}
        self._pin_budget = pin_budget
        self._store: ComponentHostStore | None = None

    def _ensure_store(self, module: nn.Module) -> ComponentHostStore:
        if self._store is None:
            # a rebuilt strategy (cache refresh) must adopt the module's
            # existing store: the module rests on placeholders by then, and a
            # fresh capture would take those as the weights
            existing = component_offload_host_store(module)
            if existing is not None:
                self._store = existing
                return existing
            self._store = ComponentHostStore(
                module,
                device=get_local_torch_device(),
                pin_budget=self._pin_budget,
                component_name=self._component_name,
            )
            # the module attribute is how code with no strategy reference
            # finds the store
            module.component_offload_host_store = self._store
        return self._store

    def prepare_for_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        if get_local_torch_device().type == "cpu":
            return
        self._swap_in(module, use, prefetch=False)

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
        self._swap_in(module, use, prefetch=True)
        return True

    def _swap_in(self, module: nn.Module, use: ComponentUse, *, prefetch: bool) -> None:
        store = self._ensure_store(module)
        needs_retarget = (
            use.target_dtype is not None
            and store.reference_dtype is not None
            and store.reference_dtype != use.target_dtype
        )
        if store.loaded and not needs_retarget:
            return
        if needs_retarget:
            if store.loaded:
                store.release()
            store.retarget_dtype(use.target_dtype)

        if not prefetch:
            store.load()
            return
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.get_device_module().Stream(
                device=get_local_torch_device()
            )
        # allocator blocks freed by compute must not be reused for the
        # incoming copies before that work drains (as in prefetch_layer)
        self._prefetch_stream.wait_stream(torch.get_device_module().current_stream())
        with torch.get_device_module().stream(self._prefetch_stream):
            store.load()
            event = torch.get_device_module().Event()
            event.record(self._prefetch_stream)
            self._ready_events[use.component_name] = event

    def finish_use(
        self,
        module: nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self._ready_events.pop(use.component_name, None)
        if self._store is None or not self._store.loaded:
            return
        if self._prefetch_stream is not None:
            # subsumes the ready event, and device tensors allocated on the
            # prefetch stream must not be reused before it drains (as in
            # release_all)
            torch.get_device_module().current_stream().wait_stream(
                self._prefetch_stream
            )
        self._store.release()

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
