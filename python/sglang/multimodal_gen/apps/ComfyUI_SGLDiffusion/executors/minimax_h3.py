# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 adapter for the ComfyUI DiT-forward contract."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

_logger = logging.getLogger(__name__)
_STEP_PROFILE = os.environ.get("SGLANG_H3_STEP_PROFILE", "0") == "1"

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    DEFAULT_SIGMA_SHIFT_AUDIO,
    DEFAULT_SIGMA_SHIFT_VIDEO,
    serialize_comfyui_layout,
    time_shift_sigma,
)

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


def _split_av(x) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return x[0], x[1]
    unbind = getattr(x, "unbind", None)
    if callable(unbind):
        parts = unbind()
        if len(parts) >= 2:
            return parts[0], parts[1]
    raise TypeError(
        "MiniMax H3 ComfyUI forward expects x to be [video, audio], "
        f"got {type(x)!r}"
    )


def _as_context(context: torch.Tensor) -> torch.Tensor:
    if context.ndim == 3:
        return context[0]
    return context


class MiniMaxH3Adapter(ComfyUIModelAdapter):
    model_types = ("minimax_h3",)
    pipeline_class_name = "MiniMaxH3Pipeline"

    def pack(self, x, timestep, context, **kwargs) -> PackedForward:
        video, audio_src = _split_av(x)
        # PackedLayout is a ComfyUI class; the SGLD worker cannot unpickle it.
        payload = dict(kwargs.get("minimax_payload") or {})
        layout = payload.pop("layout", None)
        transformer_options = kwargs.get("transformer_options") or {}
        scale = float(payload.get("audio_scale", 1.0))
        shift_v = float(
            transformer_options.get(
                "minimax_h3_sigma_shift_video", DEFAULT_SIGMA_SHIFT_VIDEO
            )
        )
        shift_a = float(
            transformer_options.get(
                "minimax_h3_sigma_shift_audio", DEFAULT_SIGMA_SHIFT_AUDIO
            )
        )
        sigma_v = (timestep.reshape(-1)[0] / 1000.0).to(torch.float32).clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
        carry = (sigma_a / sigma_v).to(audio_src.dtype)
        audio = audio_src * carry if scale != 1.0 else audio_src
        text = _as_context(context)
        sample_sigmas = transformer_options.get("sample_sigmas")
        extra_req: dict[str, Any] = {
            "audio_latents": audio,
            "h3_payload": payload,
            "h3_transformer_options": transformer_options,
            "h3_sample_sigmas": sample_sigmas,
            "h3_context": text,
        }
        serialized = serialize_comfyui_layout(layout)
        if serialized is not None:
            extra_req["h3_layout"] = serialized
        for name in ("denoise_mask", "audio_denoise_mask"):
            mask = kwargs.get(name)
            if mask is not None:
                extra_req[f"h3_{name}"] = mask
        if sample_sigmas is not None:
            extra_req["sigmas"] = sample_sigmas
        return PackedForward(
            latents=video,
            timesteps=timestep,
            prompt_embeds=[text],
            prompt_seq_lens=[[int(text.shape[0])]],
            height=int(video.shape[-2]),
            width=int(video.shape[-1]),
            extra_req=extra_req,
            unpack_ctx={
                "audio_scale": scale,
                "audio_src": audio_src,
                "carry": carry,
                "sigma_a": sigma_a,
            },
        )

    def unpack(self, noise_pred, packed, x):
        video_x, audio_x = _split_av(x)
        if isinstance(noise_pred, (list, tuple)):
            v_video, v_audio = noise_pred[0], noise_pred[1]
        else:
            raise TypeError(
                "MiniMax H3 unpack expects [video, audio] noise_pred, "
                f"got {type(noise_pred)!r}"
            )
        # ComfyUI MiniMaxH3._forward returns the negated DiT velocities.
        v_video = (-v_video).to(device=video_x.device, dtype=video_x.dtype)
        v_audio = (-v_audio).to(device=audio_x.device, dtype=audio_x.dtype)
        ctx = packed.unpack_ctx
        scale = float(ctx.get("audio_scale", 1.0))
        if scale != 1.0:
            audio_src = ctx["audio_src"]
            carry = ctx["carry"]
            sigma_a = ctx["sigma_a"]
            v_audio = (1.0 - scale) * (audio_src * carry) + (
                1.0 + (scale - 1.0) * sigma_a
            ).to(v_audio.dtype) * v_audio
        return [v_video, v_audio]

    def fill_req(self, req, packed: PackedForward) -> None:
        super().fill_req(req, packed)
        extra = dict(req.extra or {})
        for key in (
            "h3_payload",
            "h3_transformer_options",
            "h3_sample_sigmas",
            "h3_context",
            "h3_layout",
            "h3_denoise_mask",
            "h3_audio_denoise_mask",
        ):
            value = packed.extra_req.get(key)
            if value is not None:
                extra[key] = value
        req.extra = extra

    def drop_cached_fields(self, packed: PackedForward) -> None:
        super().drop_cached_fields(packed)
        packed.extra_req.pop("h3_payload", None)
        packed.extra_req.pop("h3_transformer_options", None)
        packed.extra_req.pop("h3_sample_sigmas", None)
        packed.extra_req.pop("h3_context", None)
        packed.extra_req.pop("h3_layout", None)
        packed.extra_req.pop("h3_denoise_mask", None)
        packed.extra_req.pop("h3_audio_denoise_mask", None)
        packed.extra_req.pop("sigmas", None)


class MiniMaxH3Executor(SGLDiffusionExecutor):
    adapter_cls = MiniMaxH3Adapter
    # ComfyUI MiniMaxH3.extra_conds reads this when a denoise_mask is present.
    patch_size = (1, 2, 2)

    def preprocess_text_embeds(self, text_states):
        """ComfyUI extra_conds runs this once per sample.

        Token refinement happens on the SGLD worker (the DiT lives there).
        Identity-pass here so PackedLayout still sees the correct token length.
        """
        return text_states

    def forward(self, x, timestep, context, **kwargs):
        t0 = time.perf_counter()
        video, _audio = _split_av(x)
        spatial = (
            int(video.shape[-3]),
            int(video.shape[-2]),
            int(video.shape[-1]),
        )
        self._begin_run_if_needed(timestep, kwargs, spatial=spatial)
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        if self._conditioning_sent:
            self.adapter.drop_cached_fields(packed)
        else:
            self._conditioning_sent = True
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

        t1 = time.perf_counter()
        sampling_kwargs = {
            "prompt": " ",
            "height": packed.height,
            "width": packed.width,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": self.should_suppress_logs(timestep),
        }
        enable_cache_dit = getattr(self, "enable_cache_dit", None)
        if enable_cache_dit is not None:
            sampling_kwargs["enable_cache_dit"] = bool(enable_cache_dit)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            **sampling_kwargs,
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = (
            f"{self.session_id}:{self._run_id}:{packed.height}x{packed.width}"
        )
        req.extra = extra
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]
        t2 = time.perf_counter()
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        t3 = time.perf_counter()
        out = self.adapter.unpack(output_batch.noise_pred, packed, x)
        t4 = time.perf_counter()
        if _STEP_PROFILE:
            _logger.info(
                "[h3-step-profile] parent pack=%.2fms prep=%.2fms rpc=%.2fms "
                "unpack=%.2fms total=%.2fms",
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                (t3 - t2) * 1e3,
                (t4 - t3) * 1e3,
                (t4 - t0) * 1e3,
            )
        return out
