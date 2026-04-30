from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, MutableMapping, Protocol, Sequence, TypeVar

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.component_resident_strategies import (
    ComponentResidencyStrategy,
    LayerwiseOffloadStrategy,
    ResidentStrategy,
    VanillaD2HStrategy,
)
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")


@dataclass(slots=True)
class ComponentUse:
    """Describes one stage/use-site access to a pipeline component."""

    stage_name: str
    # Pipeline module key: transformer / video_dit / text_encoder / ...
    component_name: str
    # Model-specific phase for sequential components, e.g. stage1 or stage2.
    # TODO: Replace this with ordered timeline identity. In an all-sequential
    # pipeline, use-site identity should come from the declared ComponentUse
    # order instead of a per-use `phase` field.
    phase: str | None = None
    # Whether the manager may prepare this component for the next request.
    preferred_ready_after_request: bool = False
    # Whether cross-stage prefetch may prepare this use before the use-site.
    allow_prefetch: bool = True
    # Whether this use is expensive enough that earlier timeline prefetch matters.
    # TODO: Replace this boolean hint with a budget-aware lookahead planner:
    # estimate memory/load cost and reuse distance, keep small and early-request
    # components resident within budget, prefetch as soon as VRAM slack appears,
    # and release completed components only when the budget requires it.
    memory_intensive: bool = False
    # Optional module dtype required by this use-site.
    target_dtype: torch.dtype | None = None
    # Some components are intentionally kept ready between warmup and the first
    # real request to avoid measuring a cold H2D in the user-visible request.
    keep_ready_after_warmup: bool = False


@dataclass(slots=True)
class ResidencyState:
    """
    Necessary internal runtime info of ComponentResidencyManager
    """

    stages: Sequence["ComponentResidencyStage"] = ()
    stage_index: int = -1
    stage_name: str | None = None
    next_stage_name: str | None = None
    current_use: ComponentUse | None = None
    # the ComponentUses of the preceding stages
    future_uses: tuple[ComponentUse, ...] = ()
    batch_is_warmup: bool = False
    manager_mode: str = "static"
    trace_enabled: bool = False


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


def build_dit_residency_strategy(
    module: nn.Module,
    server_args: ServerArgs,
) -> ComponentResidencyStrategy:
    if (
        isinstance(module, OffloadableDiTMixin)
        and module.layerwise_offload_managers
        and any(manager.enabled for manager in module.layerwise_offload_managers)
    ):
        # only if dit_layerwise_offload is enabled
        return LayerwiseOffloadStrategy()
    if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
        # handles offload by vanalla D2H
        return VanillaD2HStrategy()
    return ResidentStrategy()


def is_fsdp_managed_module(module: nn.Module) -> bool:
    return module.__class__.__name__.startswith("FSDP")


def build_component_residency_strategy(
    component_name: str,
    module: nn.Module,
    server_args: ServerArgs,
) -> ComponentResidencyStrategy:
    if component_name in {
        "transformer",
        "transformer_2",
        "video_dit",
        "video_dit_2",
        "audio_dit",
        "dual_tower_bridge",
    }:
        return build_dit_residency_strategy(module, server_args)

    if component_name.startswith("text_encoder") or component_name.endswith(
        "text_encoder"
    ):
        if (
            server_args.text_encoder_cpu_offload
            and not server_args.use_fsdp_inference
            and not is_fsdp_managed_module(module)
        ):
            return VanillaD2HStrategy()
        return ResidentStrategy()

    if component_name == "image_encoder":
        if server_args.image_encoder_cpu_offload and not server_args.use_fsdp_inference:
            return VanillaD2HStrategy()
        return ResidentStrategy()

    if component_name in {
        "vae",
        "video_vae",
        "audio_vae",
        "vocoder",
        "spatial_upsampler",
        "condition_image_encoder",
    }:
        if server_args.vae_cpu_offload and not server_args.use_fsdp_inference:
            return VanillaD2HStrategy()
        return ResidentStrategy()

    return ResidentStrategy()


class ComponentResidencyManager:
    """Executor-owned component lifecycle coordinator. Provide hooks for a PipelineExecutor

    Hooks are called around executor progress:
        before request: collect a flat ordered ComponentUse timeline.
        before stage: update current/next stage context only.
        begin use: finish previous active use, prepare current use, wait until ready.
        end use: finish or keep current use, then prefetch the next heavy timeline use.
        finish request: finish active use and schedule preferred next-request prefetch.

    The manager instance is global and rebound to the active pipeline before request execution.
    This manager is designed only for sequential execution order for now
    """

    def __init__(
        self, pipeline: ComponentResidencyPipeline, server_args: ServerArgs
    ) -> None:
        self.pipeline = pipeline
        self.server_args = server_args
        self.state = ResidencyState(trace_enabled=False)
        self._stage_names_by_id: dict[int, str] = {}
        self._stage_uses_by_index: list[tuple[ComponentUse, ...]] = []
        self._ordered_uses: tuple[ComponentUse, ...] = ()
        self._current_use_index: int = -1
        self._active_use: ComponentUse | None = None
        self._active_use_module: nn.Module | None = None
        self._prefetched_use_keys: set[tuple[str, str, str | None]] = set()
        self._custom_strategies: dict[str, ComponentResidencyStrategy] = dict(
            pipeline.component_residency_strategies
        )
        self._uses_seen: dict[str, ComponentUse] = {}

    @property
    def enabled(self) -> bool:
        return True

    def refresh_pipeline(self, pipeline: ComponentResidencyPipeline) -> None:
        custom_strategies = dict(pipeline.component_residency_strategies)
        if pipeline is not self.pipeline:
            self.strategy_for.cache_clear()
            self._should_keep_single_dit.cache_clear()
            self._active_use = None
            self._active_use_module = None
            self._uses_seen.clear()
            self._prefetched_use_keys.clear()
        elif custom_strategies != self._custom_strategies:
            self.strategy_for.cache_clear()
        self.pipeline = pipeline
        self._custom_strategies = custom_strategies
        self._stage_names_by_id = {
            id(stage): name for name, stage in pipeline._stage_name_mapping.items()
        }

    def refresh_server_args(self, server_args: ServerArgs) -> None:
        if server_args is not self.server_args:
            self.strategy_for.cache_clear()
        self.server_args = server_args

    def register_strategy(
        self, component_name: str, strategy: ComponentResidencyStrategy
    ) -> None:
        self.pipeline.component_residency_strategies[component_name] = strategy
        self._custom_strategies[component_name] = strategy
        self.strategy_for.cache_clear()

    def begin_request(
        self,
        stages: Sequence[ComponentResidencyStage],
        batch: ResidencyBatch,
        server_args: ServerArgs,
    ) -> None:
        """A hook called before processing an actual request"""
        self.refresh_server_args(server_args)
        self.state = ResidencyState(
            stages=stages, batch_is_warmup=batch.is_warmup, trace_enabled=False
        )
        self._active_use = None
        self._active_use_module = None
        self._current_use_index = -1
        self._prefetched_use_keys.clear()
        self._uses_seen.clear()
        if self.enabled:
            self._stage_uses_by_index = [
                tuple(stage.component_uses(server_args, self.stage_name(stage)))
                for stage in stages
            ]
            self._ordered_uses = tuple(
                use for uses in self._stage_uses_by_index for use in uses
            )
        else:
            self._stage_uses_by_index = []
            self._ordered_uses = ()
        self._trace(
            "request_start",
            detail=f"stages={len(stages)} uses={len(self._ordered_uses)}",
        )

    def before_stage(
        self,
        stage: ComponentResidencyStage,
        stage_index: int,
        batch: ResidencyBatch,
        server_args: ServerArgs,
    ) -> None:
        """called after stage starts"""
        if not self.enabled:
            return
        # update state before entering the stage
        self.state.stage_index = stage_index
        self.state.stage_name = self.stage_name(stage)
        self.state.next_stage_name = self._next_stage_name(stage_index)
        self._trace("stage_enter", detail=f"index={stage_index}")

    def after_stage(self, stage_index: int) -> None:
        """called after stage exits"""
        if not self.enabled:
            return
        self._trace("stage_exit", detail=f"index={stage_index}")

    def before_use(self, use: ComponentUse) -> None:
        """component use-site starts"""
        if not self.enabled:
            return
        self.begin_use(use)

    def begin_use(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """Begin one sequential component use interval. this is idempotent

        1. Finish the previous active use if this is a different timeline use.
        2. Prepare the current component.
        3. Wait until the current component is ready, then prefetch the next heavy use.
        """
        if self._active_use is not None and self._same_use(self._active_use, use):
            return
        if self._active_use is not None:
            # finish previous active use
            self._finish_use(
                self._active_use,
                module=self._active_use_module,
                keep_on_warmup=self._active_use.keep_ready_after_warmup,
            )
            self._active_use = None
            self._active_use_module = None
            self.state.current_use = None
        self._mark_current_use(use)
        self._prepare_forward_use(use, module=module)
        self._active_use = use
        self._active_use_module = module
        self._prefetch_next_memory_intensive_use()

    def end_use(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """End one sequential component use interval.

        1. Finish or keep the current component.
        2. Clear it as the active use.
        3. Prefetch the next memory-intensive use without waiting.
        """
        if self._active_use is None or not self._same_use(self._active_use, use):
            return
        self._finish_use(
            self._active_use,
            module=self._active_use_module or module,
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

    def call_component(
        self,
        use: ComponentUse,
        module: Callable[..., _T],
        *args,
        **kwargs,
    ) -> _T:
        with self.use_component(use):
            return module(*args, **kwargs)

    def prefetch_use(self, use: ComponentUse) -> None:
        """Prepare a future use without blocking the current use."""
        if not self.enabled:
            return
        self._prefetch_use(use)

    def ensure_ready(self, use: ComponentUse, module: nn.Module | None = None) -> None:
        """Prepare a shared component and wait without making it the active use."""
        if not self.enabled:
            return
        self._prepare_forward_use(use, module=module)

    def prefetch_checkpoint(self, anchor: ComponentUse | None = None) -> None:
        """Give the manager a timeline overlap point.

        1. Locate the anchor or current use in the ordered timeline.
        2. Find the next prefetchable memory-intensive use.
        3. Prepare it opportunistically without waiting.
        """
        if not self.enabled:
            return
        if anchor is not None:
            self._mark_current_use(anchor)
        self._prefetch_next_memory_intensive_use()

    def finish_active_use(self, *, prefetch_next: bool = True) -> None:
        """Finish the currently active sequential use, if any."""
        if self._active_use is None:
            return
        active_use = self._active_use
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
    ) -> None:
        """Prepare a component that is about to run and wait until it is ready."""
        module = module or self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        strategy = self.strategy_for(use.component_name, module)
        self._uses_seen[use.component_name] = use
        self.state.current_use = use
        self._trace("prepare", use, strategy, module)
        strategy.prepare_for_use(module, use, self.state)
        self._trace("wait", use, strategy, module)
        strategy.wait_for_use(module, use, self.state)

    def _prefetch_use(self, use: ComponentUse) -> None:
        """Prepare a future component opportunistically without waiting.

        This is called when the component is memory-intensive so it may takes a long time to prefetch.

        manager will perform the prefetch at some checkpoints, if necessary
        """
        if not use.allow_prefetch:
            return
        module = self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        strategy = self.strategy_for(use.component_name, module)
        if isinstance(strategy, VanillaD2HStrategy) and self._active_use is not None:
            # Avoid making two vanilla-offloaded heavy components resident before
            # a budget-aware planner can prove the overlap is safe.
            self._trace("prefetch_skip_active_vanilla", use, strategy, module)
            return

        self._uses_seen[use.component_name] = use
        self._trace("prefetch", use, strategy, module)
        if strategy.prefetch_for_use(module, use, self.state):
            self._prefetched_use_keys.add(self._use_key(use))

    def after_use(self, use: ComponentUse) -> None:
        if not self.enabled:
            return
        self.end_use(use)

    def _finish_use(
        self,
        use: ComponentUse,
        *,
        module: nn.Module | None = None,
        keep_on_warmup: bool,
    ) -> None:
        """finish a specific use by keeping them resident or call finish_use hook"""
        module = module or self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        should_keep = (
            keep_on_warmup and self.state.batch_is_warmup
        ) or self._should_keep_after_use(use)
        if should_keep:
            self._trace(
                "keep",
                use,
                self.strategy_for(use.component_name, module),
                module,
            )
            return
        strategy = self.strategy_for(use.component_name, module)
        self._trace("finish", use, strategy, module)
        strategy.finish_use(module, use, self.state)

    def finish_request(self) -> None:
        if not self.enabled and not self._uses_seen and self._active_use is None:
            return
        # 1. Close the currently active sequential use.
        self.finish_active_use(prefetch_next=False)
        # 2. Pick components that should be ready for the next request.
        preferred_uses = self._preferred_request_end_uses()
        # 3. Finish everything else, or prepare preferred uses for request tail.
        for component_name, use in list(self._uses_seen.items()):
            module = self.get_module(component_name)
            if module is None:
                continue
            if self.state.batch_is_warmup and use.keep_ready_after_warmup:
                self._trace(
                    "request_keep_warmup",
                    use,
                    self.strategy_for(component_name, module),
                    module,
                )
                continue
            preferred = component_name in preferred_uses
            if not preferred and self._should_keep_single_dit(component_name):
                self._trace(
                    "keep",
                    use,
                    self.strategy_for(component_name, module),
                    module,
                    detail="single_dit",
                )
                continue
            strategy = self.strategy_for(component_name, module)
            if preferred and not self.state.batch_is_warmup:
                self._trace("request_prefetch", use, strategy, module)
                strategy.prepare_after_request(module, use, self.state)
            else:
                action = "request_resident" if preferred else "request_finish"
                self._trace(action, use, strategy, module)
                strategy.finish_request(module, use, self.state, preferred=preferred)
        self._trace("request_end")

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

    @lru_cache(maxsize=None)
    def strategy_for(
        self, component_name: str, module: nn.Module
    ) -> ComponentResidencyStrategy:
        """Return the pre-registered strategy for a specific component"""
        custom_strategy = self._custom_strategies.get(component_name)
        if custom_strategy is not None:
            return custom_strategy
        return build_component_residency_strategy(
            component_name, module, self.server_args
        )

    def _stage_uses(self, stage_index: int) -> tuple[ComponentUse, ...]:
        """Returns the ComponentUse(s) of a specific stage"""
        if stage_index < 0 or stage_index >= len(self._stage_uses_by_index):
            return ()
        return self._stage_uses_by_index[stage_index]

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
        for index, candidate in enumerate(self._ordered_uses):
            if self._same_use(candidate, use):
                return index
        return None

    def _prefetch_next_memory_intensive_use(self) -> None:
        for use in self._ordered_uses[self._current_use_index + 1 :]:
            if not use.memory_intensive:
                continue
            if self._use_key(use) in self._prefetched_use_keys:
                return
            self.prefetch_use(use)
            return

    def _should_keep_after_use(self, use: ComponentUse) -> bool:
        future_component_names = {
            future.component_name for future in self.state.future_uses
        }
        if use.component_name in future_component_names:
            return True
        if self._should_keep_single_dit(use.component_name):
            return True
        return False

    @lru_cache(maxsize=None)
    def _should_keep_single_dit(self, component_name: str) -> bool:
        modules = self.pipeline.modules
        return (component_name == "transformer" and "transformer_2" not in modules) or (
            component_name == "video_dit" and "video_dit_2" not in modules
        )

    def _preferred_request_end_use(self) -> ComponentUse | None:
        """Returns a ComponentUse preferred to be resident after a request finishes, to prepare for next request"""
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

    def _trace(
        self,
        action: str,
        use: ComponentUse | None = None,
        strategy: ComponentResidencyStrategy | None = None,
        module: nn.Module | None = None,
        *,
        component_name: str | None = None,
        detail: str = "",
    ) -> None:
        if not self.state.trace_enabled:
            return
        if use is not None:
            component_name = use.component_name
        device = self._module_device(module)
        logger.info(
            "[component_residency] action=%s stage=%s next_stage=%s component=%s "
            "strategy=%s phase=%s device=%s warmup=%s mode=%s %s",
            action,
            self.state.stage_name,
            self.state.next_stage_name,
            component_name,
            strategy.name if strategy is not None else None,
            use.phase if use is not None else None,
            device,
            self.state.batch_is_warmup,
            self.state.manager_mode,
            detail,
        )

    def _module_device(self, module: nn.Module | None) -> str | None:
        if module is None:
            return None
        param = next(module.parameters(), None)
        if param is not None:
            return param.device.type
        buffer = next(module.buffers(), None)
        return buffer.device.type if buffer is not None else None


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
