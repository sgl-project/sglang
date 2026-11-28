"""Rollout HTTP API (``POST /rollout/generate``)."""

from __future__ import annotations

import dataclasses
from typing import Any

import msgspec
import torch
from fastapi import APIRouter, HTTPException, Response

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    resolve_sampling_params_cls,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
    RolloutResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/rollout", tags=["rollout"])


def _extract_single_sample_tensor(
    obj: Any, sample_idx: int, batch_size: int, *, current_key: str | None = None
) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            return obj[sample_idx].contiguous()
        return obj
    if isinstance(obj, dict):
        return {
            k: _extract_single_sample_tensor(v, sample_idx, batch_size, current_key=k)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if current_key == "img_shapes" and len(obj) == batch_size:
            return [obj[sample_idx]]
        return [
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        )
    return obj


def _slice_rollout_trajectory_for_sample(
    rtd: RolloutTrajectoryData | None,
    sample_idx: int,
    batch_size: int,
) -> RolloutTrajectoryData | None:
    if rtd is None:
        return None
    log_probs = rtd.rollout_log_probs
    if (
        isinstance(log_probs, torch.Tensor)
        and log_probs.dim() >= 1
        and log_probs.shape[0] == batch_size
    ):
        log_probs = log_probs[sample_idx].contiguous()
    debug_tensors = None
    if rtd.rollout_debug_tensors:
        rd = rtd.rollout_debug_tensors
        debug_tensors = RolloutDebugTensors(
            rollout_variance_noises=_extract_single_sample_tensor(
                rd.rollout_variance_noises, sample_idx, batch_size
            ),
            rollout_prev_sample_means=_extract_single_sample_tensor(
                rd.rollout_prev_sample_means, sample_idx, batch_size
            ),
            rollout_noise_std_devs=_extract_single_sample_tensor(
                rd.rollout_noise_std_devs, sample_idx, batch_size
            ),
            rollout_model_outputs=_extract_single_sample_tensor(
                rd.rollout_model_outputs, sample_idx, batch_size
            ),
        )
    denoising_env = None
    if rtd.denoising_env:
        env = rtd.denoising_env
        denoising_env = RolloutDenoisingEnv(
            image_kwargs=(
                _extract_single_sample_tensor(env.image_kwargs, sample_idx, batch_size)
                if env.image_kwargs
                else None
            ),
            pos_cond_kwargs=(
                _extract_single_sample_tensor(
                    env.pos_cond_kwargs, sample_idx, batch_size
                )
                if env.pos_cond_kwargs
                else None
            ),
            neg_cond_kwargs=(
                _extract_single_sample_tensor(
                    env.neg_cond_kwargs, sample_idx, batch_size
                )
                if env.neg_cond_kwargs
                else None
            ),
            guidance=(
                _extract_single_sample_tensor(env.guidance, sample_idx, batch_size)
                if env.guidance is not None
                else None
            ),
        )
    dit_trajectory = None
    if rtd.dit_trajectory:
        dit = rtd.dit_trajectory
        dit_trajectory = RolloutDitTrajectory(
            latents=_extract_single_sample_tensor(dit.latents, sample_idx, batch_size),
            timesteps=dit.timesteps,
            sigmas=dit.sigmas,
        )
    return RolloutTrajectoryData(
        rollout_log_probs=log_probs,
        rollout_debug_tensors=debug_tensors,
        denoising_env=denoising_env,
        dit_trajectory=dit_trajectory,
    )


def _serialize_rollout_trajectory(
    rtd: RolloutTrajectoryData | None,
    *,
    serialized_dit_timesteps: dict | None = None,
    serialized_dit_sigmas: dict | None = None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Return order: rollout_log_probs, rollout_debug_tensors, denoising_env, dit_trajectory."""
    if rtd is None:
        return None, None, None, None
    serialized_log_probs = _maybe_serialize(rtd.rollout_log_probs)
    serialized_debug_tensors = None
    if rtd.rollout_debug_tensors:
        rd = rtd.rollout_debug_tensors
        serialized_debug_tensors = {
            "rollout_variance_noises": _maybe_serialize(rd.rollout_variance_noises),
            "rollout_prev_sample_means": _maybe_serialize(rd.rollout_prev_sample_means),
            "rollout_noise_std_devs": _maybe_serialize(rd.rollout_noise_std_devs),
            "rollout_model_outputs": _maybe_serialize(rd.rollout_model_outputs),
        }
    serialized_denoising_env = None
    if rtd.denoising_env:
        env = rtd.denoising_env
        serialized_denoising_env = {
            "image_kwargs": (
                _maybe_serialize(env.image_kwargs) if env.image_kwargs else None
            ),
            "pos_cond_kwargs": (
                _maybe_serialize(env.pos_cond_kwargs) if env.pos_cond_kwargs else None
            ),
            "neg_cond_kwargs": (
                _maybe_serialize(env.neg_cond_kwargs) if env.neg_cond_kwargs else None
            ),
            "guidance": (
                _maybe_serialize(env.guidance) if env.guidance is not None else None
            ),
        }
    serialized_dit_trajectory = None
    if rtd.dit_trajectory:
        dit = rtd.dit_trajectory
        serialized_dit_trajectory = {
            "latents": (
                _maybe_serialize(dit.latents) if dit.latents is not None else None
            ),
            "timesteps": serialized_dit_timesteps,
            "sigmas": serialized_dit_sigmas,
        }
    return (
        serialized_log_probs,
        serialized_debug_tensors,
        serialized_denoising_env,
        serialized_dit_trajectory,
    )


def _build_response(
    request_id: str, prompt: str, seed: int, rollout: bool, result: OutputBatch
) -> list[RolloutResponse]:
    """
    rollout: bool - set to False when evaluating the model
    """
    batch_size = result.output.shape[0]
    inference_time_s = (
        result.metrics.total_duration_s
        if result.metrics and result.metrics.total_duration_s > 0
        else None
    )
    peak_memory_mb = result.peak_memory_mb if result.peak_memory_mb > 0 else None
    rollout_trajectory_data = result.rollout_trajectory_data
    if rollout:
        assert rollout_trajectory_data is not None, (
            "rollout_trajectory_data must be present when rollout=True"
        )

    serialized_dit_timesteps = None
    serialized_dit_sigmas = None
    if rollout and rollout_trajectory_data and rollout_trajectory_data.dit_trajectory:
        serialized_dit_timesteps = _maybe_serialize(
            rollout_trajectory_data.dit_trajectory.timesteps
        )
        serialized_dit_sigmas = _maybe_serialize(
            rollout_trajectory_data.dit_trajectory.sigmas
        )

    responses: list[RolloutResponse] = []
    for sample_idx in range(batch_size):
        out_i = result.output[sample_idx]
        if isinstance(out_i, torch.Tensor):
            out_i = out_i.contiguous()
        serialized_generated_output = _maybe_serialize(out_i)
        if not rollout:
            responses.append(
                RolloutResponse(
                    request_id=request_id,
                    prompt=prompt,
                    seed=seed,
                    generated_output=serialized_generated_output,
                    inference_time_s=inference_time_s,
                    peak_memory_mb=peak_memory_mb,
                )
            )
            continue
        per_sample_trajectory = _slice_rollout_trajectory_for_sample(
            result.rollout_trajectory_data, sample_idx, batch_size
        )
        (
            serialized_log_probs,
            serialized_debug_tensors,
            serialized_denoising_env,
            serialized_dit_trajectory,
        ) = _serialize_rollout_trajectory(
            per_sample_trajectory,
            serialized_dit_timesteps=serialized_dit_timesteps,
            serialized_dit_sigmas=serialized_dit_sigmas,
        )
        responses.append(
            RolloutResponse(
                request_id=request_id,
                prompt=prompt,
                seed=seed,
                generated_output=serialized_generated_output,
                rollout_log_probs=serialized_log_probs,
                rollout_debug_tensors=serialized_debug_tensors,
                denoising_env=serialized_denoising_env,
                dit_trajectory=serialized_dit_trajectory,
                inference_time_s=inference_time_s,
                peak_memory_mb=peak_memory_mb,
            )
        )
    return responses


# Fields a remote /rollout/generate client must never override through
# `extra_sampling_params`. These control where the server reads/writes files,
# whether generated output is persisted, or identify/route the request
# itself. Without this filter, an unauthenticated client could inject e.g.
# `perf_dump_path` and make GPUWorker create directories and overwrite
# arbitrary files (PerformanceLogger.dump_benchmark_report does os.makedirs
# + an atomic os.replace at the supplied path), or flip `save_output` back
# on after this endpoint forced it off. The rejected-vs-ignored choice is
# deliberate: invalid request params fail loudly with HTTP 400 everywhere
# else in this codebase (see _reject_remote_model_paths), so silently
# dropping a field would hide client mistakes instead of surfacing them.
_EXTRA_SAMPLING_PARAMS_BLOCKED_FIELDS = frozenset(
    {
        # file input paths (image_path already has a typed request field)
        "image_path",
        "video_path",
        "prompt_path",
        # file output paths
        "output_path",
        "output_file_name",
        "perf_dump_path",
        # model file paths (remote URLs are rejected later, but a client
        # must not pick server-side files either)
        "frame_interpolation_model_path",
        "upscaling_model_path",
        # output persistence control (this endpoint forces save_output=False)
        "save_output",
        # request/server identity and routing (server-derived, not client-set)
        "request_id",
        "data_type",
        "task_type",
        # internal switch guarding the protected-fields merge mechanism
        "no_override_protected_fields",
        # untyped nested dict passed straight to the diffusers scheduler;
        # not a documented rollout knob and not examinable as a flat field
        "diffusers_kwargs",
    }
)


def _allowed_extra_sampling_params() -> frozenset[str]:
    """Sampling knobs a rollout client may override via extra_sampling_params.

    The active model's SamplingParams subclass when the server is running
    (so model-specific knobs such as LTX's video_*_scale stay usable),
    falling back to the base class when server args are not set (unit tests,
    or before the server starts). Blocked fields can never be overridden,
    whichever class is active; any other `*_path`/`*_paths` field is blocked
    too, so a subclass-declared path field is covered as well.
    """
    cls: type[SamplingParams] = SamplingParams
    try:
        server_args = get_global_server_args()
    except ValueError:
        # No global server args (unit tests, or before the server starts):
        # fall back to the base class.
        server_args = None
    if server_args is not None:
        try:
            cls = resolve_sampling_params_cls(server_args)
        except (ValueError, AttributeError, TypeError):
            logger.debug(
                "Could not resolve the active SamplingParams subclass; "
                "falling back to the base-class allowlist.",
                exc_info=True,
            )
    return frozenset(
        field.name
        for field in dataclasses.fields(cls)
        if field.init
        and field.name not in _EXTRA_SAMPLING_PARAMS_BLOCKED_FIELDS
        and not field.name.endswith(("_path", "_paths"))
    )


def _build_sampling_kwargs(request: RolloutRequest) -> dict:
    sampling_kwargs: dict = dict(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        generator_device=request.generator_device,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        num_outputs_per_prompt=request.num_outputs_per_prompt,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        num_frames=request.num_frames,
        fps=request.fps,
        image_path=request.image_path,
        rollout=request.rollout,
        rollout_sde_type=request.rollout_sde_type,
        rollout_noise_level=request.rollout_noise_level,
        rollout_log_prob_no_const=request.rollout_log_prob_no_const,
        rollout_debug_mode=request.rollout_debug_mode,
        rollout_return_denoising_env=request.rollout_return_denoising_env,
        rollout_return_dit_trajectory=request.rollout_return_dit_trajectory,
        rollout_sde_step_indices=request.rollout_sde_step_indices,
        rollout_return_step_indices=request.rollout_return_step_indices,
        suppress_logs=request.suppress_logs,
        save_output=False,
        return_trajectory_latents=False,
        return_trajectory_decoded=False,
    )
    if request.extra_sampling_params:
        # The allowlist is about keys, not values: even a null value for a
        # blocked field is rejected, because a null would otherwise be
        # stripped by the None filter below and the downstream default
        # (e.g. save_output=True in build_sampling_params) would silently
        # re-enable the behaviour this endpoint forces off.
        rejected = sorted(
            key
            for key in request.extra_sampling_params
            if key not in _allowed_extra_sampling_params()
        )
        if rejected:
            raise HTTPException(
                status_code=400,
                detail=(
                    "extra_sampling_params may only contain SamplingParams "
                    f"sampling knobs; rejected field(s): {', '.join(rejected)}. "
                    "Path and output-control fields (e.g. perf_dump_path, "
                    "output_path, save_output, image_path, video_path) "
                    "cannot be set by a client."
                ),
            )
        sampling_kwargs.update(request.extra_sampling_params)
        sampling_kwargs["rollout"] = request.rollout
    return {k: v for k, v in sampling_kwargs.items() if v is not None}


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "model": list[RolloutResponse],
            "content": {"application/msgpack": {}},
        }
    },
)
async def rollout_generate(request: RolloutRequest):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    sampling_kwargs = _build_sampling_kwargs(request)
    try:
        sampling_params = build_sampling_params(request_id, **sampling_kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid sampling params: {exc}"
        ) from exc
    pipeline_request = prepare_request(
        server_args=server_args, sampling_params=sampling_params
    )
    try:
        output_batch: OutputBatch = await async_scheduler_client.forward(
            pipeline_request
        )
    except Exception as exc:
        logger.error("Rollout generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc
    if output_batch.error:
        raise HTTPException(status_code=500, detail=output_batch.error)
    seed = request.seed if request.seed is not None else sampling_params.seed
    rollout_responses = _build_response(
        request_id, request.prompt, seed, request.rollout, output_batch
    )
    payload = [r.model_dump() for r in rollout_responses]
    return Response(
        content=msgspec.msgpack.encode(payload),
        media_type="application/msgpack",
    )
