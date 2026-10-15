# SPDX-License-Identifier: Apache-2.0
"""LTX-2.x audio-video (ComfyUI ``ltxav``) adapter for the DiT-forward contract.

ComfyUI ``model_base.LTXAV`` passes ``x`` as ``[video, audio]`` and
``timestep`` as the ``(video, audio)`` pair from ``process_timestep``. The
in-DiT text connectors also run on the worker, from ``preprocess_text_embeds``.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# ComfyUI LTXVModel defaults.
VIDEO_SCALE_FACTORS = (8, 32, 32)


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
        **kwargs,
    ) -> PackedForward:
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise TypeError(
                "LTX-AV integrated mode expects [video, audio] latents; video-only "
                f"LTX latents are not supported (got {type(x)!r})"
            )
        video, audio = x[0], x[1]
        v_ts, a_ts = split_timestep(timestep)
        opts = transformer_options or {}
        unsupported = {
            "attention_mask": attention_mask is not None,
            "keyframe guides (keyframe_idxs)": keyframe_idxs is not None
            and keyframe_idxs.shape[2] > 0,
            "reference audio": kwargs.get("ref_audio") is not None,
            # Set by the LTXV STG / modality-guidance nodes for their extra pass;
            # the worker does not apply them, so the pass would equal cond.
            "STG (stg_self_attn_blocks)": bool(opts.get("stg_self_attn_blocks")),
            "modality guidance (a2v/v2a cross-attention off)": not opts.get(
                "a2v_cross_attn", True
            )
            or not opts.get("v2a_cross_attn", True),
            "single-modality passes (run_vx/run_ax)": not opts.get("run_vx", True)
            or not opts.get("run_ax", True),
        }
        if any(unsupported.values()):
            raise NotImplementedError(
                "SGLang integrated LTX does not support "
                + ", ".join(name for name, used in unsupported.items() if used)
                + "; use the stock ComfyUI model loader for this workflow"
            )
        return PackedForward(
            latents=video,
            timesteps=v_ts,
            prompt_embeds=[context],
            height=int(video.shape[-2]) * VIDEO_SCALE_FACTORS[1],
            width=int(video.shape[-1]) * VIDEO_SCALE_FACTORS[2],
            extra_req={
                "audio_latents": audio,
                # Not Req fields; the worker stage reads them from Req.extra.
                "extra": {
                    "ltxav_audio_timestep": a_ts,
                    "ltxav_frame_rate": float(frame_rate),
                },
            },
        )

    def unpack(self, noise_pred, packed, x):
        return [noise_pred[0].to(x[0]), noise_pred[1].to(x[1])]


class LTXAVExecutor(SGLDiffusionExecutor):
    adapter_cls = LTXAVAdapter

    @staticmethod
    def should_suppress_logs(timestep):
        # With I2V the video timestep is per token; conditioned tokens sit below 1.
        video_ts = torch.as_tensor(split_timestep(timestep)[0])
        return SGLDiffusionExecutor.should_suppress_logs(video_ts.max())

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
        if self._ensure_runtime is not None:
            self._ensure_runtime(self)
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
        req.extra["ltxav_connectors"] = True
        req.generator = [torch.Generator("cuda")]
        output = self.generator._send_to_scheduler_and_wait_for_response([req])
        if output.error or output.noise_pred is None:
            raise RuntimeError(
                f"SGLD worker failed the LTX text connectors: {output.error}"
            )
        return output.noise_pred.to(device=context.device, dtype=context.dtype)
