# SPDX-License-Identifier: Apache-2.0
"""LTX-2.x audio-video (ComfyUI ``ltxav``) adapter for the DiT-forward contract.

ComfyUI ``model_base.LTXAV`` unpacks the nested AV latent before calling the
diffusion model, so ``x`` is ``[video, audio]`` and ``timestep`` is the
``(video, audio)`` pair from ``process_timestep``. The worker mirrors
``LTXAVModel.forward``; the in-DiT text connectors also run on the worker,
reached from ``preprocess_text_embeds``.
"""

from __future__ import annotations

from typing import Any

import torch

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError:  # pragma: no cover - reported by base.py
    SamplingParams = None
    prepare_request = None

# Must match runtime/.../ltx_2/comfyui_step.py LTXAV_OP_CONNECTORS.
LTXAV_OP_CONNECTORS = "connectors"
_LTXAV_EXTRA_PREFIX = "ltxav_"
# ComfyUI LTXVModel defaults.
VIDEO_SCALE_FACTORS = (8, 32, 32)


def split_av(x) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return x[0], x[1]
    raise TypeError(
        "LTX-AV integrated mode expects [video, audio] latents; video-only LTX "
        f"latents are not supported (got {type(x)!r})"
    )


def split_timestep(timestep) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(timestep, (list, tuple)) and len(timestep) == 2:
        return timestep[0], timestep[1]
    return timestep, timestep


class LTXAVAdapter(ComfyUIModelAdapter):
    model_types = ("ltxav",)
    pipeline_class_name = "LTX2Pipeline"

    def pack(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        frame_rate=25,
        transformer_options=None,
        keyframe_idxs=None,
        denoise_mask=None,
        **kwargs,
    ) -> PackedForward:
        video, audio = split_av(x)
        v_ts, a_ts = split_timestep(timestep)
        unsupported = {
            "attention_mask": attention_mask is not None,
            "keyframe guides (keyframe_idxs)": keyframe_idxs is not None
            and keyframe_idxs.shape[2] > 0,
            "reference audio": kwargs.get("ref_audio") is not None,
        }
        if any(unsupported.values()):
            raise NotImplementedError(
                "SGLang integrated LTX does not support "
                + ", ".join(name for name, used in unsupported.items() if used)
                + "; use the stock ComfyUI model loader for this workflow"
            )
        extra: dict[str, Any] = {
            "audio_latents": audio,
            "ltxav_audio_timestep": a_ts,
            "ltxav_frame_rate": float(frame_rate),
        }
        return PackedForward(
            latents=video,
            timesteps=v_ts,
            prompt_embeds=[context],
            prompt_seq_lens=[[int(context.shape[1])]],
            height=int(video.shape[-2]) * VIDEO_SCALE_FACTORS[1],
            width=int(video.shape[-1]) * VIDEO_SCALE_FACTORS[2],
            extra_req=extra,
        )

    def unpack(self, noise_pred, packed, x):
        video_x, audio_x = split_av(x)
        if not isinstance(noise_pred, (list, tuple)) or len(noise_pred) != 2:
            raise TypeError(
                f"LTX-AV unpack expects [video, audio] noise_pred, got {type(noise_pred)!r}"
            )
        return [
            noise_pred[0].to(device=video_x.device, dtype=video_x.dtype),
            noise_pred[1].to(device=audio_x.device, dtype=audio_x.dtype),
        ]

    def fill_req(self, req, packed: PackedForward) -> None:
        # ltxav_* fields travel to the worker stage in req.extra, not as Req fields.
        worker_extra = {
            key: packed.extra_req.pop(key)
            for key in list(packed.extra_req)
            if key.startswith(_LTXAV_EXTRA_PREFIX)
        }
        super().fill_req(req, packed)
        req.extra = {**(req.extra or {}), **worker_extra}


class LTXAVExecutor(SGLDiffusionExecutor):
    adapter_cls = LTXAVAdapter
    # Like ComfyUI, fold LoRA into the (FP8) weights once instead of adding a
    # low-rank delta per layer and step.
    lora_merge_mode = "merge"

    @staticmethod
    def should_suppress_logs(timestep):
        # model_base.LTXAV passes the (video, audio) timestep pair.
        return SGLDiffusionExecutor.should_suppress_logs(split_timestep(timestep)[0])

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        # model_base.LTXAV.process_timestep and context windows read these.
        from comfy.ldm.lightricks.symmetric_patchifier import (
            AudioPatchifier,
            SymmetricPatchifier,
        )

        self.patchifier = SymmetricPatchifier(1, start_end=True)
        self.a_patchifier = AudioPatchifier(1, start_end=True)
        self.vae_scale_factors = VIDEO_SCALE_FACTORS
        unet_config = config.unet_config
        self.causal_temporal_positioning = bool(
            unet_config.get("causal_temporal_positioning", False)
        )
        self._processed_context_dims = {
            int(unet_config.get("cross_attention_dim", 4096))
            + int(unet_config.get("audio_cross_attention_dim", 2048)),
            2 * int(unet_config.get("caption_channels", 3840)),
        }

    def preprocess_text_embeds(self, context, unprocessed=False):
        """ComfyUI ``LTXAVModel.preprocess_text_embeds``; connectors run on the worker."""
        if not unprocessed and context.shape[-1] in self._processed_context_dims:
            return context
        return self._run_connectors(context)

    def _run_connectors(self, context: torch.Tensor) -> torch.Tensor:
        ensure = self._ensure_runtime
        if ensure is not None:
            ensure(self)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            height=VIDEO_SCALE_FACTORS[1],
            width=VIDEO_SCALE_FACTORS[2],
            num_frames=1,
            num_inference_steps=1,
            save_output=False,
            suppress_logs=True,
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        # No comfyui_session_id: connector calls must not touch the step cache.
        req.latents = context
        req.prompt_embeds = [context]
        req.do_classifier_free_guidance = False
        req.extra = {**(req.extra or {}), "ltxav_op": LTXAV_OP_CONNECTORS}
        req.generator = [torch.Generator("cuda")]
        output = self.generator._send_to_scheduler_and_wait_for_response([req])
        out = output.noise_pred
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.to(device=context.device, dtype=context.dtype)
