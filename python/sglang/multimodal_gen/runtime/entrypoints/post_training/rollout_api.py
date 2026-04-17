"""Rollout HTTP API (``POST /rollout/generate``)."""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
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


def _extract_single_sample_tensor(obj: Any, sample_idx: int, batch_size: int) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            return obj[sample_idx].contiguous()
        return obj
    if isinstance(obj, dict):
        return {
            k: _extract_single_sample_tensor(v, sample_idx, batch_size)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_extract_single_sample_tensor(v, sample_idx, batch_size) for v in obj]
    if isinstance(obj, tuple):
        return tuple(
            _extract_single_sample_tensor(v, sample_idx, batch_size) for v in obj
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
        assert (
            rollout_trajectory_data is not None
        ), "rollout_trajectory_data must be present when rollout=True"

    serialized_dit_timesteps = None
    if rollout and rollout_trajectory_data and rollout_trajectory_data.dit_trajectory:
        serialized_dit_timesteps = _maybe_serialize(
            rollout_trajectory_data.dit_trajectory.timesteps
        )

    responses: list[RolloutResponse] = []
    for sample_idx in range(batch_size):
        out_i = result.output[sample_idx].contiguous()
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


@router.post("/generate", response_model=list[RolloutResponse])
async def rollout_generate(request: RolloutRequest):
    request_id = generate_request_id()
    server_args = get_global_server_args()
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
        suppress_logs=request.suppress_logs,
        save_output=False,
        return_trajectory_latents=False,
        return_trajectory_decoded=False,
    )
    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)
        sampling_kwargs["rollout"] = request.rollout
    sampling_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}
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
    rollout_responses = _build_response(
        request_id, request.prompt, request.seed, request.rollout, output_batch
    )
    return ORJSONResponse(content=[r.model_dump() for r in rollout_responses])
