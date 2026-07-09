# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen import envs
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
    enable_sequence_shard: bool | None
    did_sp_shard_latents: bool
    sp_video_start_frame: int
    tp_size: int
    sp_degree: int
    ulysses_degree: int
    ring_degree: int
    cfg_parallel_degree: int
    enable_cfg_parallel: bool
    target_dtype: str
    # True when the request may share a forward with other resolutions via
    # sequence packing; the latent/raw shapes are then resolution-free.
    varlen_packed: bool = False


@dataclass(slots=True)
class DenoisingRequestState:
    req: Req
    identity: bytes | None
    denoising_context: DenoisingContext
    request_id: str = ""
    step_batch_key: Any = None
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
    # Step-invariant compatibility key computed at admission.
    static_batch_key: DenoisingBatchKey | None = None
    packable: bool = True
    # Cached step-invariant attention metadata.
    static_attn_metadata: Any = None
    attn_metadata_is_static: bool = False
    # Deep-copied raw request for mid-flight export/import.
    raw_req_snapshot: Req | None = None
    # Per-request TeaCache state snapshot.
    teacache_state: dict[str, Any] | None = None

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

    @property
    def remaining_steps(self) -> int:
        return max(0, self.num_timesteps - self.step_index)

    def set_error_output(self, error: str) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self.error = error
        self.output_batch = OutputBatch(error=error)
        self.completed_time_s = time.monotonic()


_TEACACHE_SCALAR_ATTRS = ("cnt", "enable_teacache", "is_cfg_negative")
_TEACACHE_PREFIX_ATTRS = ("previous_", "accumulated_rel_l1")


class TeaCacheStateIsolator:
    """Capture/restore per-request TeaCache state on a shared DiT model."""

    @staticmethod
    def state_attrs(model: Any) -> tuple[str, ...]:
        names = []
        for name in vars(model):
            if name in _TEACACHE_SCALAR_ATTRS or name.startswith(
                _TEACACHE_PREFIX_ATTRS
            ):
                names.append(name)
        return tuple(names)

    @classmethod
    def capture(cls, model: Any) -> dict[str, Any]:
        return {name: getattr(model, name) for name in cls.state_attrs(model)}

    @classmethod
    def install(cls, model: Any, snapshot: dict[str, Any] | None) -> None:
        if snapshot is not None:
            for name, value in snapshot.items():
                setattr(model, name, value)
        else:
            reset = getattr(model, "reset_teacache_state", None)
            if callable(reset):
                reset()

    @staticmethod
    def model_has_teacache(model: Any) -> bool:
        return callable(getattr(model, "reset_teacache_state", None))


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

    if getattr(req, "enable_teacache", False) and not getattr(
        server_args, "cb_allow_step_caches", True
    ):
        raise ContinuousBatchingError(
            "continuous batching TeaCache support is disabled "
            "(--cb-allow-step-caches false)"
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


def _request_is_varlen_packable(state: DenoisingRequestState, server_args: Any) -> bool:
    """Whether this request may pack with other resolutions (varlen)."""
    if not getattr(server_args, "cb_varlen_packing", False):
        return False
    if not getattr(server_args.pipeline_config, "supports_varlen_step_packing", False):
        return False
    for attr in ("sp_degree", "ulysses_degree", "ring_degree"):
        if int(getattr(server_args, attr, 1) or 1) != 1:
            return False
    step = state.current_step
    model = getattr(step, "current_model", None) if step is not None else None
    if not getattr(model, "supports_varlen_step_packing", False):
        return False
    if getattr(model, "zero_cond_t", False):
        return False
    return state.denoising_context.latents.ndim == 3


def build_denoising_batch_key(
    state: DenoisingRequestState,
    denoising_stage: Any,
    server_args: Any,
) -> DenoisingBatchKey:
    """Build the step-invariant compatibility key for one request."""
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
    latent_shape = _shape_suffix_of(ctx.latents) or ()
    varlen_packed = _request_is_varlen_packable(state, server_args)
    if varlen_packed:
        # Drop the sequence dim so different resolutions share one group.
        latent_shape = latent_shape[-1:]
        raw_latent_shape = None

    return DenoisingBatchKey(
        pipeline_config_type=type(server_args.pipeline_config).__name__,
        model_phase="",
        current_model_type=type(getattr(step, "current_model", None)).__name__,
        latent_shape=latent_shape,
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
        enable_sequence_shard=getattr(req, "enable_sequence_shard", None),
        did_sp_shard_latents=bool(getattr(req, "did_sp_shard_latents", False)),
        sp_video_start_frame=int(getattr(req, "sp_video_start_frame", 0) or 0),
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        sp_degree=int(getattr(server_args, "sp_degree", 1) or 1),
        ulysses_degree=int(getattr(server_args, "ulysses_degree", 1) or 1),
        ring_degree=int(getattr(server_args, "ring_degree", 1) or 1),
        cfg_parallel_degree=int(getattr(server_args, "cfg_parallel_degree", 1) or 1),
        enable_cfg_parallel=bool(getattr(server_args, "enable_cfg_parallel", False)),
        target_dtype=str(getattr(ctx, "target_dtype", "")),
        varlen_packed=varlen_packed,
    )


# Backends whose metadata does not depend on step content.
_STEP_INVARIANT_ATTN_BACKENDS = frozenset({"FA"})


def _attn_metadata_is_step_invariant(denoising_stage: Any) -> bool:
    backend_name = _attention_backend_name(denoising_stage)
    if backend_name is None:
        return True
    return backend_name in _STEP_INVARIANT_ATTN_BACKENDS


class ContinuousDenoisingCoordinator:
    """Coordinates continuous denoising for a set of in-flight requests."""

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
        self.schedule_policy = str(
            getattr(server_args, "cb_schedule_policy", "largest") or "largest"
        )
        self.allow_step_caches = bool(
            getattr(server_args, "cb_allow_step_caches", True)
        )
        # Reused while membership and model phase stay the same.
        self._packed_group: Any = None
        # Avoid shape ping-pong between mixed compatibility groups.
        self._last_selected_key: Any = None
        self._attn_metadata_static = _attn_metadata_is_step_invariant(
            self.denoising_stage
        )

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
        raw_req_snapshot = self._snapshot_raw_request(req)

        prepared_req = self.pipeline.run_stages_before_denoising(req, self.server_args)
        return self.prepare_request_state_from_prepared(
            identity=identity,
            prepared_req=prepared_req,
            raw_req_snapshot=raw_req_snapshot,
            response_group_id=response_group_id,
            response_index=response_index,
            response_group_size=response_group_size,
        )

    def prepare_request_state_from_prepared(
        self,
        *,
        identity: bytes | None,
        prepared_req: Req,
        raw_req_snapshot: Req | None = None,
        response_group_id: str | None = None,
        response_index: int = 0,
        response_group_size: int = 1,
    ) -> DenoisingRequestState:
        """Build state from a request whose pre-denoise stages already ran."""
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
                raw_req_snapshot=raw_req_snapshot,
            )
            if not self.prepare_next_denoising_step(state):
                raise ContinuousBatchingError(
                    "continuous batching requires at least one denoising timestep"
                )
            state.static_batch_key = build_denoising_batch_key(
                state,
                self.denoising_stage,
                self.server_args,
            )
            state.step_batch_key = (
                state.static_batch_key,
                self._model_phase(state),
            )
            state.attn_metadata_is_static = self._attn_metadata_static or (
                state.current_step is not None
                and state.current_step.attn_metadata is None
            )
            if state.attn_metadata_is_static and state.current_step is not None:
                state.static_attn_metadata = state.current_step.attn_metadata
            state.packable = self._request_is_packable(state)
            return state
        except Exception:
            self._cleanup_denoising_context(ctx)
            raise

    @staticmethod
    def _snapshot_raw_request(req: Req) -> Req | None:
        """Deep-copy the raw request for mid-flight export/import."""
        try:
            return copy.deepcopy(req)
        except Exception:
            return None

    def _model_phase(self, state: DenoisingRequestState) -> str:
        step = state.current_step
        if step is None:
            return ""
        transformer_2 = getattr(self.denoising_stage, "transformer_2", None)
        if transformer_2 is not None and step.current_model is transformer_2:
            return "transformer_2"
        return "transformer"

    def _request_is_packable(self, state: DenoisingRequestState) -> bool:
        if state.req.enable_teacache:
            # TeaCache is per-request and must run unpacked.
            return False
        if not state.attn_metadata_is_static:
            # Sparse/video backends have step-dependent metadata; keep separate.
            return False
        return True

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
            attn_metadata_override=(
                state.static_attn_metadata
                if state.attn_metadata_is_static and state.step_index > 0
                else None
            ),
            use_attn_metadata_override=(
                state.attn_metadata_is_static and state.step_index > 0
            ),
        )
        if state.static_batch_key is not None:
            state.step_batch_key = (
                state.static_batch_key,
                self._model_phase(state),
            )
        return True

    def _group_incomplete_states(
        self,
        active_states: list[DenoisingRequestState],
    ) -> dict[Any, list[DenoisingRequestState]]:
        groups: dict[Any, list[DenoisingRequestState]] = {}
        for state in active_states:
            if state.is_complete:
                continue
            if state.current_step is None:
                self.prepare_next_denoising_step(state)
            if state.current_step is None:
                continue
            groups.setdefault(state.step_batch_key, []).append(state)
        return groups

    def select_next_step_batch(
        self,
        active_states: list[DenoisingRequestState],
    ) -> list[DenoisingRequestState]:
        """Select the next compatible group to step.

        Policies: rotate (round-robin), largest (sticky largest group),
        srpt (shortest remaining steps first).
        """
        groups = self._group_incomplete_states(active_states)
        if not groups:
            return []

        policy = self.schedule_policy
        if policy == "rotate":
            for state in active_states:
                if not state.is_complete and state.current_step is not None:
                    selected_key = state.step_batch_key
                    break
            else:
                return []
        elif policy == "srpt":
            selected_key = min(
                groups,
                key=lambda key: (
                    min(state.remaining_steps for state in groups[key]),
                    -len(groups[key]),
                ),
            )
        else:  # largest (default), with stickiness
            selected_key = max(groups, key=lambda key: len(groups[key]))
            sticky_key = self._last_selected_key
            if sticky_key in groups and len(groups[sticky_key]) >= len(
                groups[selected_key]
            ):
                selected_key = sticky_key
        self._last_selected_key = selected_key

        selected = groups[selected_key][: self.batching_max_size]
        if len(selected) > 1 and not all(state.packable for state in selected):
            return selected[:1]
        return selected

    def _swap_in_teacache_state(
        self, state: DenoisingRequestState
    ) -> tuple[Any, dict[str, Any]] | None:
        """Install this request's TeaCache state on the active model."""
        if not (self.allow_step_caches and state.req.enable_teacache):
            return None
        step = state.current_step
        model = getattr(step, "current_model", None) if step is not None else None
        if model is None or not TeaCacheStateIsolator.model_has_teacache(model):
            return None
        previous = TeaCacheStateIsolator.capture(model)
        TeaCacheStateIsolator.install(model, state.teacache_state)
        return model, previous

    def _swap_out_teacache_state(
        self,
        state: DenoisingRequestState,
        swap: tuple[Any, dict[str, Any]] | None,
    ) -> None:
        if swap is None:
            return
        model, previous = swap
        state.teacache_state = TeaCacheStateIsolator.capture(model)
        TeaCacheStateIsolator.install(model, previous)

    def run_selected_steps_and_advance_requests(
        self,
        states: list[DenoisingRequestState],
    ) -> list[DenoisingRequestState]:
        """Run one denoising step for selected requests and finish completed ones."""
        if not states:
            return []

        teacache_swap = None
        if len(states) == 1:
            teacache_swap = self._swap_in_teacache_state(states[0])
        try:
            self._packed_group = self.denoising_stage.run_denoising_steps_for_requests(
                states,
                self.server_args,
                packed_group=self._packed_group,
            )
        finally:
            if teacache_swap is not None:
                self._swap_out_teacache_state(states[0], teacache_swap)

        completed: list[DenoisingRequestState] = []
        for state in states:
            state.step_index += 1
            if state.step_index >= state.num_timesteps:
                # Detach finished latents from the shared packed buffer.
                latents = state.denoising_context.latents
                if latents is not None and latents._base is not None:
                    state.denoising_context.latents = latents.clone()
                self._packed_group = None
                completed.append(state)
            else:
                self.prepare_next_denoising_step(state)
        return completed

    def finalize_completed_request(
        self, state: DenoisingRequestState
    ) -> DenoisingRequestState:
        """Run post-denoise stages and build output for a finished request."""
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
            state.set_error_output(f"continuous batching completion failed: {e}")
        return state

    def export_request_state(self, state: DenoisingRequestState) -> dict[str, Any]:
        """Serialize a mid-flight request for migration or drain-resume."""
        if state.raw_req_snapshot is None:
            raise ContinuousBatchingError(
                f"request {state.request_id} has no exportable snapshot"
            )
        ctx = state.denoising_context
        if "generator" in (ctx.extra_step_kwargs or {}):
            raise ContinuousBatchingError(
                "stochastic schedulers are not exportable mid-flight"
            )
        return {
            "version": 1,
            "request_id": state.request_id,
            "raw_req": state.raw_req_snapshot,
            "latents": ctx.latents.detach().to("cpu"),
            "step_index": int(state.step_index),
            "scheduler_step_index": getattr(ctx.scheduler, "_step_index", None),
            "response_group_id": state.response_group_id,
            "response_index": state.response_index,
            "response_group_size": state.response_group_size,
        }

    def import_request_state(
        self,
        payload: dict[str, Any],
        *,
        identity: bytes | None,
    ) -> DenoisingRequestState:
        """Resume an exported mid-flight request on this worker."""
        state = self.prepare_request_state(
            identity=identity,
            req=payload["raw_req"],
            response_group_id=payload.get("response_group_id"),
            response_index=int(payload.get("response_index", 0)),
            response_group_size=int(payload.get("response_group_size", 1)),
        )
        ctx = state.denoising_context
        latents = payload["latents"].to(
            device=ctx.latents.device, dtype=ctx.latents.dtype
        )
        ctx.latents = latents
        state.step_index = int(payload["step_index"])
        scheduler_step_index = payload.get("scheduler_step_index")
        if scheduler_step_index is not None:
            ctx.scheduler._step_index = int(scheduler_step_index)
        if not self.prepare_next_denoising_step(state):
            raise ContinuousBatchingError(
                f"imported request {state.request_id} has no remaining steps"
            )
        return state

    def export_states_to_dir(
        self,
        states: list[DenoisingRequestState],
        export_dir: str,
    ) -> list[str]:
        """Drain in-flight requests to disk for later resume."""
        import torch

        os.makedirs(export_dir, exist_ok=True)
        exported = []
        for state in states:
            if state.is_complete:
                continue
            try:
                payload = self.export_request_state(state)
            except ContinuousBatchingError as e:
                logger.warning(
                    "Skipping drain export for request %s: %s",
                    state.request_id,
                    e,
                )
                continue
            path = os.path.join(export_dir, f"{state.request_id}.cbstate.pt")
            torch.save(payload, path)
            exported.append(path)
        return exported

    def import_states_from_dir(
        self,
        export_dir: str,
    ) -> list[DenoisingRequestState]:
        """Resume drained requests from disk."""
        import torch

        if not export_dir or not os.path.isdir(export_dir):
            return []
        resumed = []
        for name in sorted(os.listdir(export_dir)):
            if not name.endswith(".cbstate.pt"):
                continue
            path = os.path.join(export_dir, name)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = self.import_request_state(payload, identity=None)
            except Exception as e:
                logger.error(
                    "Failed to resume drained request from %s: %s",
                    path,
                    e,
                    exc_info=True,
                )
                continue
            os.remove(path)
            resumed.append(state)
        return resumed

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
        """Clean denoising resources; ignore cleanup errors."""
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
