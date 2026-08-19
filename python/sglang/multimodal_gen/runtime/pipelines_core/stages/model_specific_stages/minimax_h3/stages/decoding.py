# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from collections.abc import Mapping
from contextlib import nullcontext

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_world_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_enabled,
    resolve_decode_precision,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    ActiveTargetCompiledCallable,
)


def _required_tensor(value, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must be a torch.Tensor")
    return value


@functools.lru_cache(maxsize=None)
def _cached_decode_mean_std(
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device/dtype-keyed mean/std tensors, built once per distinct combination.

    mean_values/std_values come from the loaded arch_config and are fixed
    for the process lifetime, so the same (values, device, dtype) always
    reconstructs an identical tensor; cache it instead of rebuilding it on
    every decode call.
    """
    mean = torch.as_tensor(mean_values, device=device, dtype=dtype)
    std = torch.as_tensor(std_values, device=device, dtype=dtype)
    return mean, std


def _reverse_normalize_latents(
    latents: torch.Tensor,
    *,
    mean_values,
    std_values,
    name: str,
) -> torch.Tensor:
    mean, std = _cached_decode_mean_std(
        tuple(mean_values), tuple(std_values), latents.device, latents.dtype
    )
    if mean.ndim != 1:
        raise ValueError(f"{name}.latents_mean must be 1-D, got {tuple(mean.shape)}")
    if std.ndim != 1:
        raise ValueError(f"{name}.latents_std must be 1-D, got {tuple(std.shape)}")
    if mean.shape != std.shape:
        raise ValueError(
            f"{name} latent normalization shape mismatch: "
            f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents must have a channel dimension")
    if int(latents.shape[1]) != int(mean.shape[0]):
        raise ValueError(
            f"{name} latent normalization channel mismatch: "
            f"latents.shape[1]={int(latents.shape[1])} mean_len={int(mean.shape[0])}"
        )
    view_shape = [1] * latents.ndim
    view_shape[1] = int(mean.shape[0])
    # Out of place on purpose. batch.latents / batch.audio_latents are
    # inference tensors allocated inside the denoising stage's InferenceMode,
    # while --vae-cpu-offload runs this stage under torch.inference_mode(False)
    # (PipelineExecutor._stage_needs_version_counters), where writing to one in
    # place raises. Keep the mul-then-add rounding order rather than addcmul so
    # the result does not shift with FMA contraction.
    return latents * std.view(*view_shape) + mean.view(*view_shape)


def _crop_to_target_canvas(batch: Req, frames: torch.Tensor) -> torch.Tensor:
    """Crop decoded frames [B,C,T,H,W] back to the target canvas.

    The visual VAE pads the latent grid to its tile multiples (padding lands
    at the bottom/right), so a non-tile-aligned geometry decodes larger than the
    requested canvas (e.g. 1344x768 for a 1280x704 target). Target dims come
    from the direct-mode denoise state (latent_h/w * 16); requests without
    that state keep the raw decode.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if state is None:
        return frames
    target_h = int(state["latent_h"]) * 16
    target_w = int(state["latent_w"]) * 16
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < target_h or w < target_w:
        raise ValueError(
            f"decoded frames {h}x{w} smaller than target canvas {target_h}x{target_w}"
        )
    if h == target_h and w == target_w:
        return frames
    return frames[..., :target_h, :target_w]


def _canonical_visual_video_frames(
    frames: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    if frames.ndim == 4:
        if int(frames.shape[0]) % batch_size != 0:
            raise ValueError(
                f"Decoded visual video shape {tuple(frames.shape)} is incompatible "
                f"with batch_size={batch_size}"
            )
        frames = frames.reshape(
            batch_size, int(frames.shape[0]) // batch_size, *frames.shape[1:]
        )
        frames = frames.transpose(1, 2)
    elif frames.ndim == 5:
        if int(frames.shape[0]) != batch_size:
            raise ValueError(
                f"Decoded visual video batch mismatch: frames.shape[0]={int(frames.shape[0])} "
                f"batch_size={batch_size}"
            )
    else:
        raise ValueError(
            f"Decoded visual video shape {tuple(frames.shape)} is not supported"
        )
    return frames


def _canonical_output_audio_waveform(
    audio_waveform: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    """Project audio-VAE-native ``[C, 1, L]`` audio to output ``[1, C, L]``.

    The audio VAE treats stereo channels as its decoder batch and returns
    ``[2, 1, samples]`` for MiniMax H3's one generated sample.  The generic output
    path instead selects generated samples along dimension zero.  Keep the audio VAE
    tensor unchanged for decoder artifacts, then make the singleton generated-
    sample dimension explicit only at the ``OutputBatch`` boundary.
    """
    if audio_waveform.ndim != 3:
        raise ValueError(
            "Decoded audio VAE waveform must be [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    if batch_size != 1:
        raise ValueError(
            "MiniMax H3 audio VAE output only supports one generated sample, "
            f"got visual batch_size={batch_size}"
        )
    if int(audio_waveform.shape[1]) != 1:
        raise ValueError(
            "Decoded audio VAE waveform must have shape [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    return audio_waveform.permute(1, 0, 2).contiguous()


_MINIMAX_H3_DECODER_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY = "minimax_h3_canonical_request"
_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY = "minimax_h3_resolved_plan"


def _minimax_h3_decoder_task(batch: Req) -> str | None:
    """Return the validated request task used for output-decoder routing.

    Debug requests have no canonical task and retain the
    generic decoder.
    """

    extra = getattr(batch, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    canonical = extra.get(_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if canonical is not None and not isinstance(canonical, Mapping):
        raise ValueError("minimax_h3_canonical_request must be a mapping")
    canonical_task = canonical.get("task") if isinstance(canonical, Mapping) else None
    resolved = extra.get(_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    resolved_task = getattr(resolved, "task", None) if resolved is not None else None
    if canonical_task is not None and resolved_task is not None:
        if str(canonical_task) != str(resolved_task):
            raise ValueError(
                "MiniMax H3 decoder task mismatch between canonical request and "
                "resolved plan"
            )
    task_value = resolved_task if resolved_task is not None else canonical_task
    if task_value is None:
        return None
    if not isinstance(task_value, str) or task_value not in _MINIMAX_H3_DECODER_TASKS:
        raise ValueError(f"unsupported MiniMax H3 decoder task {task_value!r}")
    return task_value


class MiniMaxH3DecodingStage(DecodingStage):
    def __init__(self, video_vae, audio_vae) -> None:
        super().__init__(vae=video_vae, component_name="video_vae")
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self._compiled_audio_vae_decode = ActiveTargetCompiledCallable()

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    @property
    def parallelism_type(self) -> StageParallelismType:
        # Every decode-group rank owns a subset of visual VAE tiles. The GPU
        # worker only materializes/saves the final OutputBatch on world rank 0.
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        video_vae_dtype = resolve_precision(
            server_args, "video_vae", precision_attr="vae_precision"
        )
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        uses = [
            ComponentUse(stage_name, "video_vae", target_dtype=video_vae_dtype),
        ]
        uses.append(ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype))
        return uses

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents",
            batch.audio_latents,
            [V.is_tensor, V.with_dims(3)],
        )
        return result

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio", batch.audio, [V.is_tensor, V.with_dims(3)])
        result.add_check("audio_sample_rate", batch.audio_sample_rate, V.positive_int)
        return result

    def _decode_audio(
        self,
        audio_latent: torch.Tensor,
        server_args: ServerArgs,
    ) -> dict:
        with self.use_declared_component(
            component_name="audio_vae",
            module=self.audio_vae,
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae
            if audio_vae.training:
                audio_vae.eval()
            audio_arch_config = server_args.pipeline_config.audio_vae_config.arch_config
            audio_decode_latent = _reverse_normalize_latents(
                audio_latent,
                mean_values=audio_arch_config.latents_mean,
                std_values=audio_arch_config.latents_std,
                name="audio_vae",
            )
            audio_vae_dtype = resolve_precision(
                server_args, "audio_vae", precision_attr="audio_vae_precision"
            )
            audio_autocast_enabled = (
                audio_latent.device.type == "cuda"
                and autocast_enabled(audio_vae_dtype, server_args.disable_autocast)
            )
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=audio_vae_dtype,
                    enabled=audio_autocast_enabled,
                )
                if audio_latent.is_cuda
                else nullcontext()
            )
            with autocast_context:
                audio_decode = self._get_vae_decode_fn(
                    audio_vae,
                    server_args,
                    decode_fn=audio_vae.decode,
                    compiled_callable=self._compiled_audio_vae_decode,
                )
                waveform = _required_tensor(
                    audio_decode(audio_decode_latent), "audio_vae.decode"
                )
            return {
                "waveform": waveform,
                "sample_rate": int(audio_vae.sample_rate),
            }

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        _minimax_h3_decoder_task(batch)
        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")
        if visual_latent.ndim != 5:
            raise ValueError("batch.latents must be [B, C, T, H, W]")
        if audio_latent.ndim != 3:
            raise ValueError(
                "batch.audio_latents must be [audio_channel, latent_dim, T]"
            )

        if self.video_vae is None:
            raise RuntimeError("MiniMax H3 tasks require the video_vae output decoder")
        with self.use_declared_component(
            component_name="video_vae",
            module=self.video_vae,
        ) as selected_video_vae:
            if selected_video_vae is None:
                raise RuntimeError("video_vae became unavailable during decode")
            self.video_vae = selected_video_vae
            if selected_video_vae.training:
                selected_video_vae.eval()
            visual_arch_config = server_args.pipeline_config.vae_config.arch_config
            visual_decode_latent = _reverse_normalize_latents(
                visual_latent,
                mean_values=visual_arch_config.latents_mean,
                std_values=visual_arch_config.latents_std,
                name="video_vae",
            )
            video_vae_dtype = resolve_decode_precision(server_args, "video_vae")
            visual_autocast_enabled = (
                visual_latent.device.type == "cuda"
                and autocast_enabled(video_vae_dtype, server_args.disable_autocast)
            )
            if visual_autocast_enabled:
                selected_video_vae.prepare_decoder_autocast_weights(video_vae_dtype)
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=video_vae_dtype,
                    enabled=visual_autocast_enabled,
                )
                if visual_latent.is_cuda
                else nullcontext()
            )
            with autocast_context:
                video_decode = self._get_vae_decode_fn(
                    selected_video_vae,
                    server_args,
                    decode_fn=selected_video_vae.decode_base,
                )
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    visual_frames = video_decode(visual_decode_latent)
                visual_frames = selected_video_vae.processor.revert_tensor(
                    visual_frames
                )
                visual_frames = _required_tensor(
                    visual_frames,
                    "video_vae.processor.revert_tensor",
                )
                visual_frames = _canonical_visual_video_frames(
                    visual_frames, batch_size=int(visual_latent.shape[0])
                )
                visual_frames = _crop_to_target_canvas(batch, visual_frames)
                if (
                    visual_frames.dtype != torch.float32
                    or not visual_frames.is_contiguous()
                ):
                    canonical_frames = torch.empty_like(
                        visual_frames,
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    )
                    canonical_frames.copy_(visual_frames)
                    visual_frames = canonical_frames

        # DP is currently rejected by ServerArgs validation, so the world group
        # is one request replica (TP/CFG/SP ranks), not a collection of
        # independent requests. Decode the non-sharded audio VAE once per
        # request and distribute its output to the ranks that decoded video.
        world_group = get_world_group() if model_parallel_is_initialized() else None
        is_audio_owner = world_group is None or world_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        if is_audio_owner:
            try:
                audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if world_group is not None:
            owner_error = world_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if world_group is not None:
            audio_payload = world_group.broadcast_tensor_dict(audio_payload, src=0)
        if not isinstance(audio_payload, dict):
            raise RuntimeError("MiniMax H3 audio decode produced no output payload")
        audio_waveform = _required_tensor(
            audio_payload.get("waveform"), "audio_vae.decode"
        )
        audio_sample_rate = int(audio_payload["sample_rate"])

        visual_frames = server_args.pipeline_config.post_decoding(
            visual_frames, server_args
        )
        output_audio_waveform = _canonical_output_audio_waveform(
            audio_waveform, batch_size=int(visual_frames.shape[0])
        )
        return OutputBatch(
            output=visual_frames,
            audio=output_audio_waveform,
            audio_sample_rate=audio_sample_rate,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )


__all__ = [
    "MiniMaxH3DecodingStage",
]
