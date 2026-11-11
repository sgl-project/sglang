"""Rollout HTTP API (``POST /rollout/generate``)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import torch
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
    RolloutResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.output_store import (
    rollout_output_store,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.request_timing import (
    STAGES_HEADER,
    TIMING_HEADER,
    RequestStamps,
    stages_header,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
    _quantize_video_uint8,
    msgpack_encode_spliced,
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

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

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
            latent_step_indices=dit.latent_step_indices,
        )
    return RolloutTrajectoryData(
        rollout_log_probs=log_probs,
        rollout_debug_tensors=debug_tensors,
        denoising_env=denoising_env,
        dit_trajectory=dit_trajectory,
    )


def _store_leaves(
    obj: Any,
    sample_idx: int,
    batch_size: int,
    *,
    path: str,
    stored: dict[str, torch.Tensor],
    current_key: str | None = None,
) -> Any:
    """``_extract_single_sample_tensor`` with every tensor put in ``stored`` and a marker in its place."""
    if isinstance(obj, torch.Tensor):
        # One field per sample of a batched tensor, so a reader fetches only its samples.
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            path, obj = f"{path}.{sample_idx}", obj[sample_idx]
        stored[path] = obj
        return {"__store__": path}
    if isinstance(obj, dict):
        return {
            k: _store_leaves(
                v,
                sample_idx,
                batch_size,
                path=f"{path}.{k}",
                stored=stored,
                current_key=k,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        if current_key == "img_shapes" and len(obj) == batch_size:
            return [obj[sample_idx]]
        return [
            _store_leaves(
                v,
                sample_idx,
                batch_size,
                path=f"{path}.{i}",
                stored=stored,
                current_key=current_key,
            )
            for i, v in enumerate(obj)
        ]
    return _maybe_serialize(obj)


def _store_steps(
    tensor: torch.Tensor | None,
    sample_idx: int,
    *,
    path: str,
    stored: dict[str, torch.Tensor],
    steps: list[int] | None = None,
) -> list[dict] | None:
    """One field per step of a ``[B, steps, ...]`` tensor's sample, named by its original step number."""
    if tensor is None:
        return None
    if steps is None:
        steps = range(tensor.shape[1])
    markers = []
    for position, step in enumerate(steps):
        field = f"{path}.{sample_idx}.{step}"
        stored[field] = tensor[sample_idx, position]
        markers.append({"__store__": field})
    return markers


def _serialize_rollout_trajectory_via_store(
    rtd: RolloutTrajectoryData,
    sample_idx: int,
    batch_size: int,
    *,
    stored: dict[str, torch.Tensor],
    serialized_dit_timesteps: dict | None,
    serialized_dit_sigmas: dict | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """``_serialize_rollout_trajectory`` of the batched trajectory, its large tensors going to ``stored``."""
    log_probs = rtd.rollout_log_probs
    if (
        isinstance(log_probs, torch.Tensor)
        and log_probs.dim() >= 1
        and log_probs.shape[0] == batch_size
    ):
        log_probs = log_probs[sample_idx].contiguous()
    serialized_debug_tensors = None
    if rtd.rollout_debug_tensors:
        serialized_debug_tensors = {
            name: _store_steps(
                tensor,
                sample_idx,
                path=f"rollout_debug_tensors.{name}",
                stored=stored,
            )
            for name, tensor in vars(rtd.rollout_debug_tensors).items()
        }
    serialized_denoising_env = None
    if rtd.denoising_env:
        env = rtd.denoising_env
        serialized_denoising_env = {
            name: (
                _store_leaves(
                    kwargs,
                    sample_idx,
                    batch_size,
                    path=f"denoising_env.{name}",
                    stored=stored,
                )
                if kwargs
                else None
            )
            for name, kwargs in (
                ("image_kwargs", env.image_kwargs),
                ("pos_cond_kwargs", env.pos_cond_kwargs),
                ("neg_cond_kwargs", env.neg_cond_kwargs),
            )
        }
        serialized_denoising_env["guidance"] = (
            _maybe_serialize(
                _extract_single_sample_tensor(env.guidance, sample_idx, batch_size)
            )
            if env.guidance is not None
            else None
        )
    serialized_dit_trajectory = None
    if rtd.dit_trajectory:
        dit = rtd.dit_trajectory
        serialized_dit_trajectory = {
            "latents": _store_steps(
                dit.latents,
                sample_idx,
                path="dit_trajectory.latents",
                stored=stored,
                steps=(
                    dit.latent_step_indices.tolist()
                    if dit.latent_step_indices is not None
                    else None
                ),
            ),
            "timesteps": serialized_dit_timesteps,
            "sigmas": serialized_dit_sigmas,
            "latent_step_indices": _maybe_serialize(dit.latent_step_indices),
        }
    return (
        _maybe_serialize(log_probs),
        serialized_debug_tensors,
        serialized_denoising_env,
        serialized_dit_trajectory,
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
            "latent_step_indices": (
                _maybe_serialize(dit.latent_step_indices)
                if dit.latent_step_indices is not None
                else None
            ),
        }
    return (
        serialized_log_probs,
        serialized_debug_tensors,
        serialized_denoising_env,
        serialized_dit_trajectory,
    )


def _build_response(
    request_id: str,
    prompt: str,
    seed: int,
    rollout: bool,
    result: OutputBatch,
    video_dtype: str | None = None,
    stored: dict[str, torch.Tensor] | None = None,
) -> list[RolloutResponse]:
    """
    rollout: bool - set to False when evaluating the model
    stored: collects the batched tensors to put in the output store, or None to inline them
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
            if video_dtype == "uint8":
                out_i = _quantize_video_uint8(out_i)
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
        if stored is None:
            per_sample_trajectory = _slice_rollout_trajectory_for_sample(
                result.rollout_trajectory_data, sample_idx, batch_size
            )
            serialized = _serialize_rollout_trajectory(
                per_sample_trajectory,
                serialized_dit_timesteps=serialized_dit_timesteps,
                serialized_dit_sigmas=serialized_dit_sigmas,
            )
        else:
            serialized = _serialize_rollout_trajectory_via_store(
                result.rollout_trajectory_data,
                sample_idx,
                batch_size,
                stored=stored,
                serialized_dit_timesteps=serialized_dit_timesteps,
                serialized_dit_sigmas=serialized_dit_sigmas,
            )
        (
            serialized_log_probs,
            serialized_debug_tensors,
            serialized_denoising_env,
            serialized_dit_trajectory,
        ) = serialized
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
        rollout_stop_step_index=request.rollout_stop_step_index,
        suppress_logs=request.suppress_logs,
        save_output=False,
        return_trajectory_latents=False,
        return_trajectory_decoded=False,
    )
    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)
        sampling_kwargs["rollout"] = request.rollout
    return {k: v for k, v in sampling_kwargs.items() if v is not None}


def _stream_parts(parts: list[bytes], ref: StoreRef | None):
    sent = False
    try:
        yield from parts
        sent = True
    finally:
        if ref is not None and not sent:
            rollout_output_store.delete(ref)


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
    stamps = RequestStamps()
    stamps.mark("srv_recv")
    request_id = generate_request_id()
    stamps.request_id = request_id
    server_args = get_global_server_args()
    if request.rollout_return_outputs_via_store and not rollout_output_store.enabled:
        raise HTTPException(
            status_code=400,
            detail="rollout_return_outputs_via_store needs a server started with "
            "--output-store-backend",
        )
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
        with stamps.span("forward"):
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

    def _serialize_response() -> tuple[list[bytes], StoreRef | None]:
        stored = {} if request.rollout_return_outputs_via_store else None
        with stamps.span("build"):
            rollout_responses = _build_response(
                request_id,
                request.prompt,
                request.seed,
                request.rollout,
                output_batch,
                video_dtype=request.rollout_video_dtype,
                stored=stored,
            )
        ref = None
        if stored:
            with stamps.span("store"):
                ref = rollout_output_store.put(stored)
            for rollout_response in rollout_responses:
                rollout_response.output_store_ref = ref.to_builtins()
        try:
            with stamps.span("dump"):
                payload = [r.model_dump() for r in rollout_responses]
            with stamps.span("msgpack"):
                return msgpack_encode_spliced(payload), ref
        except BaseException:
            if ref is not None:
                rollout_output_store.delete(ref)
            raise

    # Off the event loop: building the response copies hundreds of MB
    parts, ref = await asyncio.to_thread(_serialize_response)

    # Timing rides a header: the last marks postdate the encoded body. The
    # explicit content-length keeps uvicorn on identity framing, not chunked.
    headers = {
        TIMING_HEADER: stamps.to_header(),
        "content-length": str(sum(len(p) for p in parts)),
    }
    stages = stages_header(output_batch.metrics)
    if stages:
        headers[STAGES_HEADER] = stages
    return StreamingResponse(
        _stream_parts(parts, ref),
        media_type="application/msgpack",
        headers=headers,
    )
