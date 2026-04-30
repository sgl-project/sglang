from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, MutableMapping, Protocol, Sequence

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# local handoff slots for sequential active-use handoff
DIT_HANDOFF_SLOT = "dit"
MOVA_VIDEO_DIT_HANDOFF_SLOT = "mova_video_dit"


@dataclass(slots=True)
class ComponentUse:
    """Describes one stage/use-site access to a pipeline component."""

    # Logical stage that declares or triggers this use.
    stage_name: str
    # Pipeline module key: transformer / video_dit / text_encoder / ...
    component_name: str
    # Model-specific phase for sequential components, e.g. stage1 or stage2.
    phase: str | None = None
    # Whether the manager may prepare this component for the next request.
    preferred_ready_after_request: bool = False
    # Whether cross-stage prefetch may prepare this use before the use-site.
    allow_prefetch: bool = True


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
    # the ComponentUses from the following stages
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
        """Wait for the preparation to be ready"""
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
        del module, use, state

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

    def enter(self, module: nn.Module) -> None:
        pass

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        pass


class ResidentStrategy(ComponentResidencyStrategy):
    name = "resident"


class StageManagedStrategy(ComponentResidencyStrategy):
    """No-op strategy for components with existing stage-local device lifecycle."""

    name = "stage_managed"


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

    def release_to_snapshot(self, component_name: str, module: nn.Module) -> None:
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
            # Preserve runtime-updated buffers (e.g., lazily built caches) when
            # releasing back to CPU snapshots.
            if buffer.device.type == "cuda":
                snapshot.copy_(buffer.detach().to(device="cpu", dtype=snapshot.dtype))
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


class VanillaD2HStrategy(ComponentResidencyStrategy):
    """A strategy that performs native torch D2H and H2D"""

    name = "vanilla"

    def enter(self, module: nn.Module) -> None:
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            module.to(get_local_torch_device(), non_blocking=True)

    def exit(self, module: nn.Module, next_module: nn.Module | None = None) -> None:
        del next_module
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cuda":
            module.to("cpu")

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
        del next_module
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

    if component_name.startswith("text_encoder"):
        if (
            server_args.text_encoder_cpu_offload
            and not server_args.use_fsdp_inference
            and not is_fsdp_managed_module(module)
        ):
            return VanillaD2HStrategy()
        return ResidentStrategy()

    if component_name == "image_encoder":
        return StageManagedStrategy()

    if component_name in {
        "vae",
        "video_vae",
        "audio_vae",
        "vocoder",
        "spatial_upsampler",
    }:
        return StageManagedStrategy()

    return ResidentStrategy()


class ComponentResidencyManager:
    """Executor-owned component lifecycle coordinator. Provide hooks for a PipelineExecutor

    Hooks are called around executor progress:
        before request: collect stage-declared uses and reset request state.
        before stage: update current/next stage context only.
        before usage: make the required component ready at its use-site.
        after usage: finish or keep the component after a use-site.
        after stage: finish declared stage uses and optionally prefetch the next stage.
        finish request: finish active handoff slots and schedule preferred next-request prefetch.

    The manager instance is global and rebound to the active pipeline before request execution.
    """

    def __init__(
        self, pipeline: ComponentResidencyPipeline, server_args: ServerArgs
    ) -> None:
        self.pipeline = pipeline
        self.server_args = server_args
        self.state = ResidencyState(
            manager_mode=server_args.component_residency_manager,
            trace_enabled=server_args.component_residency_trace,
        )
        self._stage_names_by_id: dict[int, str] = {}
        self._stage_uses_by_index: list[tuple[ComponentUse, ...]] = []
        self._custom_strategies: dict[str, ComponentResidencyStrategy] = dict(
            pipeline.component_residency_strategies
        )
        # marks the active use of a handoff slot
        # a handoff slot is a local handoff domain for sequential components,
        # e.g. transformer_1 and transformer_2.
        # if a `switch_use` is called within a same handoff slot, manager will try to:
        # 1. finish prev active use
        # 2. prepare/wait for next use
        # 3. make active
        # while for cross-handoff-slot components, manager doesn't handle them internally now
        self._active_uses_by_handoff_slot: dict[str, ComponentUse] = {}
        self._uses_seen: dict[str, ComponentUse] = {}

    @property
    def enabled(self) -> bool:
        return self.server_args.component_residency_manager != "disabled"

    def refresh_pipeline(self, pipeline: ComponentResidencyPipeline) -> None:
        custom_strategies = dict(pipeline.component_residency_strategies)
        if pipeline is not self.pipeline:
            self.strategy_for.cache_clear()
            self._active_uses_by_handoff_slot.clear()
            self._uses_seen.clear()
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
            stages=stages,
            batch_is_warmup=batch.is_warmup,
            manager_mode=server_args.component_residency_manager,
            trace_enabled=server_args.component_residency_trace,
        )
        self._active_uses_by_handoff_slot.clear()
        self._uses_seen.clear()
        if self.enabled:
            self._stage_uses_by_index = [
                tuple(stage.component_uses(server_args, self.stage_name(stage)))
                for stage in stages
            ]
        else:
            self._stage_uses_by_index = []
        self._trace("request_start", detail=f"stages={len(stages)}")

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
        del batch, server_args
        # update state before entering the stage
        self.state.stage_index = stage_index
        self.state.stage_name = self.stage_name(stage)
        self.state.next_stage_name = self._next_stage_name(stage_index)
        self.state.future_uses = self._future_uses(stage_index + 1)
        self._trace("stage_enter", detail=f"index={stage_index}")

    def after_stage(self, stage_index: int) -> None:
        """called after stage exits"""
        if not self.enabled:
            return
        for use in self._stage_uses(stage_index):
            self.after_use(use)
        self._trace("stage_exit", detail=f"index={stage_index}")
        # Automatic cross-stage prefetch only looks at the next future stage
        # that declares component uses. Explicit sequential component handoff
        # inside a stage is handled by use-site hooks such as switch_use().
        self.prefetch_next_stage_uses(stage_index + 1)

    def before_use(self, use: ComponentUse) -> None:
        """component use-site starts"""
        if not self.enabled:
            return
        self._prepare_forward_use(use)

    def prefetch_use(self, use: ComponentUse) -> None:
        """prepare for next stage by prefetching"""
        if not self.enabled:
            return
        self._prefetch_use(use)

    def switch_use(self, use: ComponentUse, handoff_slot: str | None = None) -> None:
        """Trigger an explicit intra-stage use-site change. (e.g., dual-dit in denoising stage)

        This path always enforces readiness; disabled mode only disables
        cross-stage scheduling, not required intra-stage component switching.
        """
        key = handoff_slot or use.component_name
        prev_active_use = self._active_uses_by_handoff_slot.get(key)
        if prev_active_use is not None and self._same_use(prev_active_use, use):
            return
        if prev_active_use is not None:
            # finish the previously active use
            self._finish_use(prev_active_use, keep_on_warmup=False)
        # prepare for the upcoming use
        self._prepare_forward_use(use)
        self._active_uses_by_handoff_slot[key] = use

    def finish_handoff_slot(self, handoff_slot: str) -> None:
        """Finish the current explicit intra-stage use-site for a handoff slot."""
        active_use = self._active_uses_by_handoff_slot.pop(handoff_slot, None)
        if active_use is not None:
            self._finish_use(active_use, keep_on_warmup=False)

    def _prepare_forward_use(self, use: ComponentUse) -> None:
        """Prepare a component that is about to run and wait until it is ready."""
        module = self.get_module(use.component_name)
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
        """Prepare a future component opportunistically without waiting."""
        if not use.allow_prefetch:
            return
        module = self.get_module(use.component_name)
        if module is None:
            self._trace("skip_missing", use)
            return
        strategy = self.strategy_for(use.component_name, module)
        if isinstance(strategy, VanillaD2HStrategy):
            self._trace("prefetch_skip", use, strategy, module)
            return

        self._uses_seen[use.component_name] = use
        self._trace("prefetch", use, strategy, module)
        strategy.prepare_for_use(module, use, self.state)

    def after_use(self, use: ComponentUse) -> None:
        if not self.enabled:
            return
        self._finish_use(use, keep_on_warmup=True)

    def _finish_use(self, use: ComponentUse, *, keep_on_warmup: bool) -> None:
        """finish a specific use by keeping them resident or call finish_use hook"""
        module = self.get_module(use.component_name)
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
        if (
            not self.enabled
            and not self._uses_seen
            and not self._active_uses_by_handoff_slot
        ):
            return
        for handoff_slot in tuple(self._active_uses_by_handoff_slot):
            self.finish_handoff_slot(handoff_slot)
        preferred_uses = self._preferred_request_end_uses()
        for component_name, use in list(self._uses_seen.items()):
            module = self.get_module(component_name)
            if module is None:
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

    def prefetch_next_stage_uses(self, start_index: int) -> None:
        """Prefetch uses declared by the next future stage with component needs.

        TODO: this is temporary and conservative. Broader lookahead requires a
        real dependency graph, otherwise prefetch can over-reserve VRAM.
        """
        if not self.enabled:
            return
        for index in range(start_index, len(self._stage_uses_by_index)):
            uses = self._stage_uses(index)
            if not uses:
                continue
            for use in uses:
                self.prefetch_use(use)
            return

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
        """Return the strategy for a specific component"""
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

    def _future_uses(self, start_stage_index: int) -> tuple[ComponentUse, ...]:
        """Returns the component uses in the future stages"""
        uses: list[ComponentUse] = []
        for index in range(start_stage_index, len(self._stage_uses_by_index)):
            uses.extend(self._stage_uses_by_index[index])
        return tuple(uses)

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
