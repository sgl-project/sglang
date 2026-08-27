from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol, Sequence

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
    ComponentResidencyStrategy,
    LayerwiseOffloadStrategy,
    ResidentStrategy,
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
    is_resident_layerwise_module,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import DiffusionNvtxHooks

logger = init_logger(__name__)


@dataclass(slots=True)
class ComponentUse:
    """One ordered stage access to a pipeline component."""

    stage_name: str
    component_name: str
    phase: str | None = None
    preferred_ready_after_request: bool = False
    allow_prefetch: bool = True
    memory_intensive: bool = False
    target_dtype: torch.dtype | None = None
    keep_ready_after_warmup: bool = False
    start_at_stage_entry: bool = True


@dataclass(slots=True)
class ResidencyState:
    """Request-local state shared with component strategies."""

    stages: Sequence["ComponentResidencyStage"] = ()
    stage_index: int = -1
    stage_name: str | None = None
    next_stage_name: str | None = None
    current_use: ComponentUse | None = None
    future_uses: tuple[ComponentUse, ...] = ()
    batch_is_warmup: bool = False


class ResidencyBatch(Protocol):
    is_warmup: bool


class ComponentResidencyStage(Protocol):
    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]: ...


class ComponentResidencyPipeline(Protocol):
    modules: Mapping[str, object]
    _stage_name_mapping: Mapping[str, ComponentResidencyStage]
    component_residency_strategies: MutableMapping[str, "ComponentResidencyStrategy"]


def build_component_residency_strategy(
    component_name: str,
    module: nn.Module,
    server_args: ServerArgs,
) -> ComponentResidencyStrategy:
    residency_mode = server_args.residency_mode(component_name)
    if is_layerwise_offloaded_module(module):
        return LayerwiseOffloadStrategy()
    if residency_mode == LAYERWISE_OFFLOAD:
        raise ComponentResidencyError(
            f"Component {component_name!r} resolved to layerwise-offload, but its "
            "loaded module did not enable layerwise offload"
        )
    if residency_mode == COMPONENT_OFFLOAD and is_fsdp_managed_module(module):
        raise ComponentResidencyError(
            f"Component {component_name!r} resolved to component-offload, but it "
            "was loaded as an FSDP-managed module"
        )
    if (
        not current_platform.is_mps()
        and not is_fsdp_managed_module(module)
        and residency_mode == COMPONENT_OFFLOAD
    ):
        return ComponentOffloadStrategy()
    return ResidentStrategy()


class ComponentResidencyManager:
    """Coordinate component placement over a sequential request timeline."""

    def __init__(
        self, pipeline: ComponentResidencyPipeline, server_args: ServerArgs
    ) -> None:
        self.pipeline = pipeline
        self.server_args = server_args
        self.state = ResidencyState()
        self._stage_names_by_id: dict[int, str] = {}
        self._stage_uses_by_index: list[tuple[ComponentUse, ...]] = []
        self._ordered_uses: tuple[ComponentUse, ...] = ()
        self._current_use_index: int = -1
        self._active_use: ComponentUse | None = None
        self._active_use_module: nn.Module | None = None
        self._active_nvtx_key: tuple[str, str, str | None] | None = None
        self._nvtx_hooks_by_use_key: dict[
            tuple[str, str, str | None], tuple[int, DiffusionNvtxHooks]
        ] = {}
        self._prefetched_use_keys: set[tuple[str, str, str | None]] = set()
        self._custom_strategies: dict[str, ComponentResidencyStrategy] = dict(
            pipeline.component_residency_strategies
        )
        self._strategy_cache: dict[
            str, tuple[nn.Module, ComponentResidencyStrategy]
        ] = {}
        self._uses_seen: dict[str, ComponentUse] = {}
        self._modules_seen: dict[str, nn.Module] = {}

    def refresh_pipeline(self, pipeline: ComponentResidencyPipeline) -> None:
        custom_strategies = dict(pipeline.component_residency_strategies)
        if pipeline is not self.pipeline:
            self._remove_nvtx_hooks()
            self._strategy_cache.clear()
            self._active_use = None
            self._active_use_module = None
            self._uses_seen.clear()
            self._modules_seen.clear()
            self._prefetched_use_keys.clear()
        elif custom_strategies != self._custom_strategies:
            self._strategy_cache.clear()
        self.pipeline = pipeline
        self._custom_strategies = custom_strategies
        self._stage_names_by_id = {
            id(stage): name for name, stage in pipeline._stage_name_mapping.items()
        }

    def refresh_server_args(self, server_args: ServerArgs) -> None:
        if server_args is not self.server_args:
            self._strategy_cache.clear()
        self.server_args = server_args

    def begin_request(
        self,
        stages: Sequence[ComponentResidencyStage],
        batch: ResidencyBatch | list[ResidencyBatch],
        server_args: ServerArgs,
    ) -> None:
        self.refresh_server_args(server_args)
        self.state = ResidencyState(
            stages=stages,
            batch_is_warmup=self._is_warmup_batch(batch),
        )
        self._active_use = None
        self._active_use_module = None
        self._disable_active_nvtx()
        self._current_use_index = -1
        self._prefetched_use_keys.clear()
        self._uses_seen.clear()
        self._modules_seen.clear()
        self._stage_uses_by_index = [
            tuple(stage.component_uses(server_args, self.stage_name(stage)))
            for stage in stages
        ]
        self._ordered_uses = tuple(
            use for uses in self._stage_uses_by_index for use in uses
        )

    @staticmethod
    def _is_warmup_batch(batch: ResidencyBatch | list[ResidencyBatch]) -> bool:
        if isinstance(batch, list):
            return bool(batch) and all(item.is_warmup for item in batch)
        return batch.is_warmup

    def before_stage(
        self,
        stage: ComponentResidencyStage,
        stage_index: int,
        batch: ResidencyBatch,
        server_args: ServerArgs,
    ) -> None:
        self.state.stage_index = stage_index
        self.state.stage_name = self.stage_name(stage)
        self.state.next_stage_name = self._next_stage_name(stage_index)

    def begin_stage(self) -> None:
        """Prepare a stage that declares one uninterrupted component use."""
        stage_uses = self._stage_uses_by_index[self.state.stage_index]
        if len(stage_uses) == 1 and stage_uses[0].start_at_stage_entry:
            self.begin_use(stage_uses[0])

    def end_stage(self) -> None:
        """Close the component interval owned by the current stage."""
        if self._active_use is None:
            return
        if self._active_use.stage_name != self.state.stage_name:
            return
        if self.state.future_uses and self._same_use(
            self._active_use, self.state.future_uses[0]
        ):
            return
        self.finish_active_use()

    def begin_use(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """Begin one sequential component use interval.

        Repeated calls for the same component/phase extend the active interval.
        """
        if self._active_use is not None and self._same_use(self._active_use, use):
            previous_use = self._active_use
            if self._use_key(self._active_use) != self._use_key(use):
                self._mark_current_use(use)
                self._active_use = use
                self.state.current_use = use
            active_module = module
            if active_module is None:
                active_module = self._active_use_module
            if active_module is None:
                active_module = self.get_module(use.component_name)
            module_changed = (
                self._active_use_module is not None
                and active_module is not self._active_use_module
            )
            if module_changed:
                self._disable_active_nvtx()
                self._finish_use(
                    previous_use,
                    module=self._active_use_module,
                    keep_on_warmup=False,
                    force=True,
                )
            if active_module is not None and (
                self._active_use_module is None
                or module_changed
                or use.target_dtype != previous_use.target_dtype
            ):
                active_module = self._prepare_forward_use(use, module=active_module)
                self._active_use = use
                self._active_use_module = active_module
                self.state.current_use = use
            self._enable_nvtx_for_use(use, active_module)
            return
        if self._active_use is not None:
            self._disable_active_nvtx()
            self._finish_use(
                self._active_use,
                module=self._active_use_module,
                keep_on_warmup=self._active_use.keep_ready_after_warmup,
            )
            self._active_use = None
            self._active_use_module = None
            self.state.current_use = None
        self._mark_current_use(use)
        module = self._prepare_forward_use(use, module=module)
        self._active_use = use
        self._active_use_module = module
        self._enable_nvtx_for_use(use, module)
        self._prefetch_next_memory_intensive_use()

    def end_use(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """End one sequential component use interval."""
        if self._active_use is None or not self._same_use(self._active_use, use):
            return
        self._disable_active_nvtx()
        self._finish_use(
            self._active_use,
            module=(
                self._active_use_module
                if self._active_use_module is not None
                else module
            ),
            keep_on_warmup=self._active_use.keep_ready_after_warmup,
        )
        self._active_use = None
        self._active_use_module = None
        self.state.current_use = None
        self._prefetch_next_memory_intensive_use()

    @contextmanager
    def use_component(
        self, use: ComponentUse, module: nn.Module | None = None
    ) -> Iterator[nn.Module | None]:
        self.begin_use(use, module=module)
        try:
            yield module if module is not None else self.get_module(use.component_name)
        finally:
            self.end_use(use, module=module)

    def ensure_ready(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """Prepare a shared component and wait without making it the active use."""
        self._prepare_forward_use(use, module=module)

    def remove_nvtx_hooks_for_module(self, module: nn.Module | None) -> None:
        """Detach NVTX hooks before a component object is deleted or replaced."""
        if module is None:
            return
        module_id = id(module)
        for key, (registered_id, hooks) in list(self._nvtx_hooks_by_use_key.items()):
            if registered_id != module_id:
                continue
            if self._active_nvtx_key == key:
                hooks.set_enabled(False)
                self._active_nvtx_key = None
            hooks.remove_hooks()
            del self._nvtx_hooks_by_use_key[key]

    def forget_module(self, module: nn.Module | None) -> None:
        """Drop manager-owned references before a component is deleted."""
        if module is None:
            return
        self.remove_nvtx_hooks_for_module(module)
        forgotten_component_names: set[str] = set()
        if self._active_use_module is module:
            forgotten_component_names.add(self._active_use.component_name)
            self._active_use = None
            self._active_use_module = None
            self.state.current_use = None
        for component_name, (cached_module, _) in list(self._strategy_cache.items()):
            if cached_module is module:
                del self._strategy_cache[component_name]
                forgotten_component_names.add(component_name)
        for component_name, seen_module in list(self._modules_seen.items()):
            if seen_module is module:
                del self._modules_seen[component_name]
                forgotten_component_names.add(component_name)
        for component_name in forgotten_component_names:
            self._uses_seen.pop(component_name, None)
        self._prefetched_use_keys = {
            key
            for key in self._prefetched_use_keys
            if key[1] not in forgotten_component_names
        }

    def finish_active_use(self, *, prefetch_next: bool = True) -> None:
        """Finish the currently active sequential use, if any."""
        if self._active_use is None:
            return
        active_use = self._active_use
        self._disable_active_nvtx()
        self._finish_use(
            active_use,
            module=self._active_use_module,
            keep_on_warmup=active_use.keep_ready_after_warmup,
        )
        self._active_use = None
        self._active_use_module = None
        self.state.current_use = None
        if prefetch_next:
            self._prefetch_next_memory_intensive_use()

    def _prepare_forward_use(
        self, use: ComponentUse, module: nn.Module | None = None
    ) -> nn.Module | None:
        """Prepare a component that is about to run and wait until it is ready."""
        if module is None:
            module = self.get_module(use.component_name)
        if module is None:
            return None
        strategy = self.strategy_for(use.component_name, module)
        self._uses_seen[use.component_name] = use
        self._modules_seen[use.component_name] = module
        self.state.current_use = use
        strategy.prepare_for_use(module, use, self.state)
        strategy.wait_for_use(module, use, self.state)
        return module

    def _enable_nvtx_for_use(
        self, use: ComponentUse, module: nn.Module | None = None
    ) -> None:
        if (
            not self.server_args.enable_layerwise_nvtx_marker
            or self.state.batch_is_warmup
            or not isinstance(module, nn.Module)
        ):
            self._disable_active_nvtx()
            return

        key = self._use_key(use)
        if self._active_nvtx_key != key:
            self._disable_active_nvtx()

        module_id = id(module)
        existing = self._nvtx_hooks_by_use_key.get(key)
        if existing is None or existing[0] != module_id:
            if existing is not None:
                existing[1].remove_hooks()
                self._nvtx_hooks_by_use_key.pop(key, None)
            hooks = DiffusionNvtxHooks()
            prefix = self._nvtx_prefix_for_use(use)
            total = hooks.register_hooks(module, prefix=prefix)
            if total == 0:
                return
            logger.debug(
                "[component_residency] Registered NVTX hooks for %s on %d submodules",
                prefix,
                total,
            )
            self._nvtx_hooks_by_use_key[key] = (module_id, hooks)
        else:
            hooks = existing[1]

        hooks.set_enabled(True)
        self._active_nvtx_key = key

    def _disable_active_nvtx(self) -> None:
        if self._active_nvtx_key is None:
            return
        existing = self._nvtx_hooks_by_use_key.get(self._active_nvtx_key)
        if existing is not None:
            existing[1].set_enabled(False)
        self._active_nvtx_key = None

    def _remove_nvtx_hooks(self) -> None:
        self._disable_active_nvtx()
        for _, hooks in self._nvtx_hooks_by_use_key.values():
            hooks.remove_hooks()
        self._nvtx_hooks_by_use_key.clear()

    @staticmethod
    def _nvtx_prefix_for_use(use: ComponentUse) -> str:
        parts = [use.stage_name, use.component_name]
        if use.phase is not None and use.phase != use.component_name:
            parts.append(use.phase)
        return ".".join(parts)

    def _prefetch_use(self, use: ComponentUse) -> None:
        """Prepare a future memory-intensive component without waiting."""
        if not use.allow_prefetch:
            return
        module = self.get_module(use.component_name)
        if module is None:
            return
        strategy = self.strategy_for(use.component_name, module)
        if (
            isinstance(strategy, ComponentOffloadStrategy)
            and self._active_use is not None
        ):
            return
        if is_resident_layerwise_module(module):
            return

        self._uses_seen[use.component_name] = use
        self._modules_seen[use.component_name] = module
        if strategy.prefetch_for_use(module, use, self.state):
            self._prefetched_use_keys.add(self._use_key(use))

    def _finish_use(
        self,
        use: ComponentUse,
        *,
        module: nn.Module | None = None,
        keep_on_warmup: bool,
        force: bool = False,
    ) -> None:
        if module is None:
            module = self._modules_seen.get(use.component_name)
        if module is None:
            module = self.get_module(use.component_name)
        if module is None:
            return
        if not force:
            should_keep = (
                keep_on_warmup and self.state.batch_is_warmup
            ) or self._should_keep_after_use(use)
            if should_keep:
                return
        strategy = self.strategy_for(use.component_name, module)
        was_on_supported_device = self._module_on_supported_device(module)
        strategy.finish_use(module, use, self.state)
        self._empty_cache_after_large_release(
            use, strategy, module, was_on_supported_device
        )

    def finish_request(self) -> None:
        self.finish_active_use(prefetch_next=False)
        preferred_uses = self._preferred_request_end_uses()
        for component_name, use in list(self._uses_seen.items()):
            module = self._modules_seen.get(component_name)
            if module is None:
                module = self.get_module(component_name)
            if module is None:
                continue
            if self.state.batch_is_warmup and use.keep_ready_after_warmup:
                continue
            preferred = component_name in preferred_uses
            if is_resident_layerwise_module(module):
                preferred = False
            keep_single_dit = self._should_keep_single_dit(component_name, module)
            if not preferred and keep_single_dit:
                continue
            preferred = preferred and (
                not self._is_single_dit_component(component_name) or keep_single_dit
            )
            strategy = self.strategy_for(component_name, module)
            was_on_supported_device = self._module_on_supported_device(module)
            strategy.finish_request(module, use, self.state, preferred=preferred)
            self._empty_cache_after_large_release(
                use, strategy, module, was_on_supported_device
            )

    def stage_name(self, stage: ComponentResidencyStage) -> str:
        return self._stage_names_by_id.get(id(stage), stage.__class__.__name__)

    def component_name_for_module(self, module: nn.Module | None, default: str) -> str:
        if module is None:
            return default
        for name, candidate in self.pipeline.modules.items():
            if candidate is module:
                return name
        return default

    def get_module(self, component_name: str) -> nn.Module | None:
        module = self.pipeline.modules.get(component_name)
        return module if isinstance(module, nn.Module) else None

    def strategy_for(
        self, component_name: str, module: nn.Module
    ) -> ComponentResidencyStrategy:
        cached = self._strategy_cache.get(component_name)
        if cached is not None and cached[0] is module:
            return cached[1]
        custom_strategy = self._custom_strategies.get(component_name)
        if custom_strategy is None:
            strategy = build_component_residency_strategy(
                component_name,
                module,
                self.server_args,
            )
        else:
            strategy = custom_strategy
        self._strategy_cache[component_name] = (module, strategy)
        return strategy

    def _next_stage_name(self, stage_index: int) -> str | None:
        next_index = stage_index + 1
        if next_index < 0 or next_index >= len(self.state.stages):
            return None
        return self.stage_name(self.state.stages[next_index])

    def _mark_current_use(self, use: ComponentUse) -> None:
        index = self._locate_use_index(use)
        if index is None:
            self._current_use_index = len(self._ordered_uses)
            self.state.future_uses = ()
            return
        self._current_use_index = index
        self.state.future_uses = self._ordered_uses[index + 1 :]

    def _locate_use_index(self, use: ComponentUse) -> int | None:
        for index in range(self._current_use_index + 1, len(self._ordered_uses)):
            if self._same_use(self._ordered_uses[index], use):
                return index
        return None

    def _prefetch_next_memory_intensive_use(self) -> None:
        for use in self._ordered_uses[self._current_use_index + 1 :]:
            if not use.memory_intensive:
                continue
            if self._use_key(use) in self._prefetched_use_keys:
                return
            self._prefetch_use(use)
            return

    def _should_keep_after_use(self, use: ComponentUse) -> bool:
        if self.state.future_uses and self._same_use(use, self.state.future_uses[0]):
            return True
        module = self.get_module(use.component_name)
        if module is not None and self._should_keep_single_dit(
            use.component_name, module
        ):
            return True
        return False

    def _should_keep_single_dit(self, component_name: str, module: nn.Module) -> bool:
        if not self._is_single_dit_component(component_name):
            return False
        return isinstance(self.strategy_for(component_name, module), ResidentStrategy)

    def _is_single_dit_component(self, component_name: str) -> bool:
        modules = self.pipeline.modules
        return (component_name == "transformer" and "transformer_2" not in modules) or (
            component_name == "video_dit" and "video_dit_2" not in modules
        )

    def _preferred_request_end_use(self) -> ComponentUse | None:
        for uses in self._stage_uses_by_index:
            for use in uses:
                if use.preferred_ready_after_request:
                    return use
        for uses in self._stage_uses_by_index:
            if uses:
                return uses[0]
        return None

    def _preferred_request_end_uses(self) -> dict[str, ComponentUse]:
        preferred_uses: dict[str, ComponentUse] = {}
        for uses in self._stage_uses_by_index:
            for use in uses:
                if use.preferred_ready_after_request:
                    preferred_uses[use.component_name] = use
        for use in self._uses_seen.values():
            if use.preferred_ready_after_request:
                preferred_uses[use.component_name] = use
        if preferred_uses:
            return preferred_uses
        preferred_use = self._preferred_request_end_use()
        if preferred_use is None:
            return {}
        return {preferred_use.component_name: preferred_use}

    @staticmethod
    def _same_use(lhs: ComponentUse, rhs: ComponentUse) -> bool:
        return lhs.component_name == rhs.component_name and lhs.phase == rhs.phase

    @staticmethod
    def _use_key(use: ComponentUse) -> tuple[str, str, str | None]:
        return (use.stage_name, use.component_name, use.phase)

    def _module_device(self, module: nn.Module | None) -> str | None:
        if module is None:
            return None
        param = next(module.parameters(), None)
        if param is not None:
            return param.device.type
        buffer = next(module.buffers(), None)
        return buffer.device.type if buffer is not None else None

    def _module_on_supported_device(self, module: nn.Module | None) -> bool:
        is_supported_platform = (
            current_platform.is_cuda()
            or current_platform.is_rocm()
            or current_platform.is_npu()
        )
        return is_supported_platform and current_platform.is_device_type(
            self._module_device(module)
        )

    def _empty_cache_after_large_release(
        self,
        use: ComponentUse,
        strategy: ComponentResidencyStrategy,
        module: nn.Module,
        was_on_supported_device: bool,
    ) -> None:
        if not use.memory_intensive:
            return
        released_device_storage = (
            was_on_supported_device and not self._module_on_supported_device(module)
        )
        released_layerwise_storage = isinstance(strategy, LayerwiseOffloadStrategy)
        should_empty_component_cache = (
            released_device_storage and not current_platform.is_npu()
        )
        if not (should_empty_component_cache or released_layerwise_storage):
            return
        if not torch.get_device_module().is_available():
            return
        torch.get_device_module().empty_cache()


_GLOBAL_COMPONENT_RESIDENCY_MANAGER: ComponentResidencyManager | None = None


def get_global_component_residency_manager(
    pipeline: ComponentResidencyPipeline,
    server_args: ServerArgs,
) -> ComponentResidencyManager:
    global _GLOBAL_COMPONENT_RESIDENCY_MANAGER

    if _GLOBAL_COMPONENT_RESIDENCY_MANAGER is None:
        _GLOBAL_COMPONENT_RESIDENCY_MANAGER = ComponentResidencyManager(
            pipeline, server_args
        )
    else:
        _GLOBAL_COMPONENT_RESIDENCY_MANAGER.refresh_server_args(server_args)
    _GLOBAL_COMPONENT_RESIDENCY_MANAGER.refresh_pipeline(pipeline)

    return _GLOBAL_COMPONENT_RESIDENCY_MANAGER
