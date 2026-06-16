# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingContext,
        DenoisingStepState,
    )


class ContinuousBatchingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DenoisingBatchKey:
    pipeline_config_type: str
    model_phase: str
    current_model_type: str
    latent_shape: tuple[int, ...]
    latent_dtype: str
    latent_device: str
    timestep_shape: tuple[int, ...]
    timestep_dtype: str
    timestep_device: str
    raw_latent_shape: tuple[int, ...] | None
    image_latent_shape: tuple[int, ...] | None
    cfg_branch_count: int
    cfg_branch_names: tuple[str, ...]
    do_classifier_free_guidance: bool
    scheduler_type: str
    scheduler_order: int | None
    attention_backend: str | None
    attention_metadata_type: str | None
    attention_metadata_signature: Any
    tp_size: int
    sp_degree: int
    ulysses_degree: int
    ring_degree: int
    cfg_parallel_degree: int
    enable_cfg_parallel: bool
    target_dtype: str


@dataclass(slots=True)
class DenoisingRequestState:
    req: Req
    identity: bytes | None
    denoising_context: DenoisingContext
    request_id: str = ""
    step_batch_key: DenoisingBatchKey | None = None
    step_index: int = 0
    current_step: DenoisingStepState | None = None
    output_batch: OutputBatch | None = None
    error: str | None = None
    response_group_id: str | None = None
    response_index: int = 0
    response_group_size: int = 1
    created_time_s: float = field(default_factory=time.monotonic)
    completed_time_s: float | None = None
    queue_wait_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(getattr(self.req, "request_id", ""))

    @property
    def is_complete(self) -> bool:
        return self.output_batch is not None or self.error is not None

    @property
    def num_timesteps(self) -> int:
        extra = getattr(self.denoising_context, "extra", None)
        if extra is None:
            extra = {}
            self.denoising_context.extra = extra
        timesteps_cpu = extra.get("timesteps_cpu")
        if timesteps_cpu is None:
            timesteps_cpu = self.denoising_context.timesteps.cpu()
            extra["timesteps_cpu"] = timesteps_cpu
        return int(timesteps_cpu.shape[0])

    def set_error_output(self, error: str) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self.error = error
        self.output_batch = OutputBatch(error=error)
        self.completed_time_s = time.monotonic()


@dataclass(slots=True)
class ContinuousResponseGroup:
    identity: bytes | None
    reqs: list[Req]
    outputs: list[Any | None] = field(init=False)
    is_warmup: bool = field(init=False)

    def __post_init__(self) -> None:
        self.outputs = [None] * len(self.reqs)
        self.is_warmup = any(req.is_warmup for req in self.reqs)

    def set_member_output(self, index: int, output_batch: Any) -> None:
        if index < 0 or index >= len(self.outputs):
            raise IndexError(f"response group index {index} out of range")
        self.outputs[index] = output_batch

    @property
    def has_all_outputs(self) -> bool:
        return all(output is not None for output in self.outputs)

    def merge_outputs(self, worker: Any) -> Any:
        if not self.has_all_outputs:
            raise RuntimeError("cannot merge an incomplete continuous response group")
        return worker._merge_expanded_output_batches(self.outputs)


def validate_continuous_batching_config(server_args: Any) -> None:
    pipeline_config = server_args.pipeline_config
    supports = getattr(pipeline_config, "supports_continuous_batching", None)
    if not callable(supports) or not supports():
        raise ContinuousBatchingError(
            "continuous batching is only supported by pipelines that explicitly "
            "opt in via supports_continuous_batching()"
        )

    backend = getattr(server_args, "backend", None)
    backend_value = str(getattr(backend, "value", backend)).lower()
    if backend_value == "diffusers":
        raise ContinuousBatchingError(
            "continuous batching requires the native composed pipeline backend; "
            "use --backend sglang"
        )

    if pipeline_config.task_type != ModelTaskType.T2I:
        raise ContinuousBatchingError(
            "continuous batching currently only supports text-to-image pipelines"
        )

    disagg_role = getattr(server_args, "disagg_role", RoleType.MONOLITHIC)
    disagg_role_value = getattr(disagg_role, "value", disagg_role)
    if disagg_role_value != RoleType.MONOLITHIC.value:
        raise ContinuousBatchingError(
            "continuous batching does not support disaggregated serving yet"
        )

    if getattr(server_args, "comfyui_mode", False):
        raise ContinuousBatchingError(
            "continuous batching does not support ComfyUI noise-pred requests"
        )

    if envs.SGLANG_CACHE_DIT_ENABLED or getattr(server_args, "cache_dit_config", None):
        raise ContinuousBatchingError(
            "continuous batching currently does not support Cache-DiT"
        )

    if getattr(server_args, "lora_path", None):
        raise ContinuousBatchingError(
            "continuous batching currently does not support startup LoRA adapters"
        )

    if getattr(server_args, "dit_cpu_offload", False):
        raise ContinuousBatchingError(
            "continuous batching requires a GPU-resident DiT; disable --dit-cpu-offload"
        )

    if getattr(server_args, "dit_layerwise_offload", False) or getattr(
        server_args, "is_dit_layerwise_offload_selected", False
    ):
        raise ContinuousBatchingError(
            "continuous batching requires a GPU-resident DiT; disable "
            "--dit-layerwise-offload/DiT layerwise offload components"
        )


def validate_continuous_batching_request(req: Req, server_args: Any) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

    if not isinstance(req, Req):
        raise ContinuousBatchingError(
            f"continuous batching only handles generation Req objects, got {type(req)}"
        )

    pipeline_config = server_args.pipeline_config
    if pipeline_config.task_type != ModelTaskType.T2I:
        raise ContinuousBatchingError(
            "continuous batching currently only supports text-to-image pipelines"
        )

    if not isinstance(getattr(req, "prompt", None), str):
        raise ContinuousBatchingError(
            "continuous batching currently only supports a single text prompt per request"
        )

    if req.image_path is not None:
        raise ContinuousBatchingError(
            "continuous batching currently does not support image-conditioned requests"
        )

    if getattr(req, "rollout", False):
        raise ContinuousBatchingError(
            "continuous batching currently does not support rollout requests"
        )

    if getattr(req, "enable_teacache", False):
        raise ContinuousBatchingError(
            "continuous batching currently does not support TeaCache requests"
        )

    if getattr(req, "return_raw_frames", False):
        raise ContinuousBatchingError(
            "continuous batching currently does not support raw-frame streaming responses"
        )


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _shape_suffix_of(value: Any) -> tuple[int, ...] | None:
    shape = _shape_of(value)
    if shape is None:
        return None
    return shape[1:]


def _shape_suffix_from_sequence(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = tuple(int(dim) for dim in value)
    return shape[1:]


def _device_of(value: Any) -> str:
    device = getattr(value, "device", None)
    return str(device) if device is not None else "unknown"


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def _attention_backend_name(stage: Any) -> str | None:
    backend = getattr(stage, "attn_backend", None)
    if backend is None:
        return None
    get_enum = getattr(backend, "get_enum", None)
    enum_value = get_enum() if callable(get_enum) else None
    if isinstance(enum_value, AttentionBackendEnum):
        return enum_value.name
    return type(backend).__name__


def _to_hashable_signature(value: Any) -> Any:
    import torch

    def convert(item: Any) -> Any:
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, torch.Tensor):
            return ("tensor", tuple(item.shape), str(item.dtype), str(item.device))
        if is_dataclass(item):
            return (
                type(item).__name__,
                tuple(
                    (field.name, convert(getattr(item, field.name)))
                    for field in fields(item)
                ),
            )
        if isinstance(item, dict):
            return tuple(
                sorted((str(key), convert(value)) for key, value in item.items())
            )
        if isinstance(item, (list, tuple)):
            return tuple(convert(value) for value in item)
        return type(item).__name__

    return convert(value)


def build_denoising_batch_key(
    state: DenoisingRequestState,
    denoising_stage: Any,
    server_args: Any,
) -> DenoisingBatchKey:
    """Build the compatibility key used to group denoising steps."""
    req = state.req
    ctx = state.denoising_context
    step = state.current_step
    if step is None:
        raise ValueError("state.current_step must be prepared before key building")

    scheduler = ctx.scheduler
    cfg_policy = ctx.cfg_policy
    branches = tuple(branch.name for branch in getattr(cfg_policy, "branches", ()))
    attn_metadata = getattr(step, "attn_metadata", None)
    raw_latent_shape = _shape_suffix_from_sequence(
        getattr(req, "raw_latent_shape", None)
    )

    return DenoisingBatchKey(
        pipeline_config_type=type(server_args.pipeline_config).__name__,
        model_phase=(
            "transformer_2"
            if getattr(step, "current_model", None)
            is getattr(denoising_stage, "transformer_2", None)
            else "transformer"
        ),
        current_model_type=type(getattr(step, "current_model", None)).__name__,
        latent_shape=_shape_suffix_of(ctx.latents) or (),
        latent_dtype=_dtype_of(ctx.latents),
        latent_device=_device_of(ctx.latents),
        timestep_shape=_shape_of(step.t_device) or (),
        timestep_dtype=_dtype_of(step.t_device),
        timestep_device=_device_of(step.t_device),
        raw_latent_shape=raw_latent_shape,
        image_latent_shape=_shape_suffix_of(getattr(req, "image_latent", None)),
        cfg_branch_count=len(branches),
        cfg_branch_names=branches,
        do_classifier_free_guidance=bool(req.do_classifier_free_guidance),
        scheduler_type=type(scheduler).__name__,
        scheduler_order=getattr(scheduler, "order", None),
        attention_backend=_attention_backend_name(denoising_stage),
        attention_metadata_type=(
            type(attn_metadata).__name__ if attn_metadata is not None else None
        ),
        attention_metadata_signature=_to_hashable_signature(attn_metadata),
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        sp_degree=int(getattr(server_args, "sp_degree", 1) or 1),
        ulysses_degree=int(getattr(server_args, "ulysses_degree", 1) or 1),
        ring_degree=int(getattr(server_args, "ring_degree", 1) or 1),
        cfg_parallel_degree=int(getattr(server_args, "cfg_parallel_degree", 1) or 1),
        enable_cfg_parallel=bool(getattr(server_args, "enable_cfg_parallel", False)),
        target_dtype=str(getattr(ctx, "target_dtype", "")),
    )


class ContinuousDenoisingCoordinator:
    """Coordinates request-local denoising state for continuous batching."""

    def __init__(
        self,
        *,
        worker: Any,
        server_args: Any,
        batching_max_size: int,
    ) -> None:
        self.worker = worker
        self.server_args = server_args
        self.batching_max_size = max(1, int(batching_max_size))
        self.pipeline = worker.pipeline
        self.denoising_stage = self.pipeline.get_denoising_stage()

    def prepare_request_state(
        self,
        *,
        identity: bytes | None,
        req: Req,
        response_group_id: str | None = None,
        response_index: int = 0,
        response_group_size: int = 1,
    ) -> DenoisingRequestState:
        validate_continuous_batching_request(req, self.server_args)

        prepared_req = self.pipeline.run_stages_before_denoising(req, self.server_args)
        ctx = self.denoising_stage.prepare_denoising_context(
            prepared_req,
            self.server_args,
            open_progress_bar=True,
        )
        try:
            state = DenoisingRequestState(
                req=prepared_req,
                identity=identity,
                denoising_context=ctx,
                request_id=str(getattr(prepared_req, "request_id", "")),
                response_group_id=response_group_id,
                response_index=response_index,
                response_group_size=response_group_size,
            )
            if not self.prepare_next_denoising_step(state):
                raise ContinuousBatchingError(
                    "continuous batching requires at least one denoising timestep"
                )
            return state
        except Exception:
            self._cleanup_denoising_context(ctx)
            raise

    def prepare_next_denoising_step(self, state: DenoisingRequestState) -> bool:
        if state.step_index >= state.num_timesteps:
            state.current_step = None
            state.step_batch_key = None
            return False
        state.current_step = self.denoising_stage.prepare_denoising_step_state(
            state.denoising_context,
            state.req,
            self.server_args,
            state.step_index,
        )
        state.step_batch_key = build_denoising_batch_key(
            state,
            self.denoising_stage,
            self.server_args,
        )
        return True

    def select_next_step_batch(
        self,
        active_states: list[DenoisingRequestState],
    ) -> list[DenoisingRequestState]:
        first_key = None
        for state in active_states:
            if state.is_complete:
                continue
            if state.current_step is None:
                self.prepare_next_denoising_step(state)
            first_key = state.step_batch_key
            break
        if first_key is None:
            return []

        selected = [
            state
            for state in active_states
            if not state.is_complete and state.step_batch_key == first_key
        ]
        selected = selected[: self.batching_max_size]
        can_pack_selected = self.denoising_stage.can_run_steps_in_one_forward_pass(
            selected,
            self.server_args,
        )
        if len(selected) > 1 and not can_pack_selected:
            return selected[:1]
        return selected

    def run_selected_steps_and_advance_requests(
        self,
        states: list[DenoisingRequestState],
    ) -> list[DenoisingRequestState]:
        """Run one denoising step for selected requests and finish completed ones."""
        if not states:
            return []

        self.denoising_stage.run_denoising_steps_for_requests(
            states,
            self.server_args,
        )

        completed: list[DenoisingRequestState] = []
        for state in states:
            state.step_index += 1
            if state.step_index >= state.num_timesteps:
                try:
                    self._finish_request_and_build_output(state)
                except Exception as e:
                    logger.error(
                        "Error completing continuous batching request %s: %s",
                        state.request_id,
                        e,
                        exc_info=True,
                    )
                    self._cleanup_denoising_context(state.denoising_context)
                    state.set_error_output(
                        f"continuous batching completion failed: {e}"
                    )
                completed.append(state)
            else:
                self.prepare_next_denoising_step(state)
        return completed

    def _finish_denoising_context(self, state: DenoisingRequestState) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self.denoising_stage.finish_denoising_context(
            state.denoising_context,
            state.req,
            self.server_args,
        )
        state.output_batch = OutputBatch()
        state.completed_time_s = time.monotonic()

    def _finish_request_and_build_output(self, state: DenoisingRequestState) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self._finish_denoising_context(state)
        result = self.pipeline.run_stages_after_denoising(state.req, self.server_args)
        output_batch = self.worker._to_output_batch(result)
        if not isinstance(output_batch, OutputBatch):
            output_batch = OutputBatch(
                error=f"Unexpected output batch: {type(output_batch).__name__}"
            )
        self._prepare_output_for_return(state, output_batch)
        state.output_batch = output_batch

    def fail_request(self, state: DenoisingRequestState, error: str) -> None:
        self._cleanup_denoising_context(state.denoising_context)
        state.set_error_output(error)

    def _cleanup_denoising_context(self, ctx: DenoisingContext) -> None:
        """Clean denoising resources without masking the original failure."""
        try:
            self.denoising_stage.cleanup_denoising_context(ctx)
        except Exception:
            logger.debug(
                "Ignoring continuous batching cleanup failure",
                exc_info=True,
            )

    def _prepare_output_for_return(
        self,
        state: DenoisingRequestState,
        output_batch: OutputBatch,
    ) -> None:
        import torch

        req = state.req
        self.worker._record_output_peak_memory(output_batch)
        if output_batch.metrics is not None:
            output_batch.metrics.total_duration_ms = (
                time.monotonic() - state.created_time_s
            ) * 1000.0

        if req.save_output and req.return_file_paths_only:
            self.worker._save_output_paths(req, output_batch)
            output_batch.output = None
            output_batch.audio = None
            output_batch.audio_sample_rate = None
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

        self.worker._materialize_frame_outputs_for_return(output_batch, req)
