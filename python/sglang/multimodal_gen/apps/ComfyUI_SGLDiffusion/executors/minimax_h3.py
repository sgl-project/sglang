# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 adapter for the ComfyUI DiT-forward contract."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    DEFAULT_SIGMA_SHIFT_AUDIO,
    DEFAULT_SIGMA_SHIFT_VIDEO,
    serialize_comfyui_layout,
    time_shift_sigma,
)

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# H3 pins loop/CFG fields with init=False; ComfyUI owns those.
_H3_PINNED_SAMPLING_FIELDS = frozenset(
    f.name for f in fields(MiniMaxH3SamplingParams) if not f.init
)


def drop_h3_pinned_sampling_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    for name in _H3_PINNED_SAMPLING_FIELDS:
        kwargs.pop(name, None)
    return kwargs


_WORKER_TRANSFORMER_OPTION_KEYS = (
    "minimax_h3_sigma_shift_video",
    "minimax_h3_sigma_shift_audio",
    "sample_sigmas",
)


def worker_transformer_options(opts: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only pickle-safe H3 keys; drop ComfyUI model_sampling etc."""
    src = opts or {}
    return {key: src[key] for key in _WORKER_TRANSFORMER_OPTION_KEYS if key in src}


def _split_av(x) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return x[0], x[1]
    unbind = getattr(x, "unbind", None)
    if callable(unbind):
        parts = unbind()
        if len(parts) >= 2:
            return parts[0], parts[1]
    raise TypeError(
        f"MiniMax H3 ComfyUI forward expects x to be [video, audio], got {type(x)!r}"
    )


def _as_context(context: torch.Tensor) -> torch.Tensor:
    if context.ndim == 3:
        return context[0]
    return context


def _cache_fingerprint(video, sample_sigmas, serialized_layout) -> dict[str, Any]:
    spatial = tuple(int(dim) for dim in video.shape[-3:])
    sigmas_fp = None
    if sample_sigmas is not None:
        values = (
            sample_sigmas.reshape(-1)
            if torch.is_tensor(sample_sigmas)
            else sample_sigmas
        )
        sigmas_fp = (float(values[0]), float(values[-1]), int(len(values)))
    layout_sig = None
    if isinstance(serialized_layout, dict):
        layout_sig = serialized_layout.get("signature")
    return {"spatial": spatial, "sigmas": sigmas_fp, "layout": layout_sig}


class MiniMaxH3Adapter(ComfyUIModelAdapter):
    model_types = ("minimax_h3",)
    pipeline_class_name = "MiniMaxH3Pipeline"

    def pack(self, x, timestep, context, **kwargs) -> PackedForward:
        video, audio_src = _split_av(x)
        if video.shape[0] != 1:
            # Same limit as ComfyUI's MiniMax H3 model: the checkpoint is
            # CFG-distilled and the packed sequence carries a single sample.
            raise ValueError(f"MiniMax H3 supports batch size 1, got {video.shape[0]}")
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
            "h3_transformer_options": worker_transformer_options(transformer_options),
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
        extra_req["comfyui_cache_fp"] = _cache_fingerprint(
            video, sample_sigmas, serialized
        )
        return PackedForward(
            latents=video,
            timesteps=timestep,
            prompt_embeds=[text],
            prompt_seq_lens=[[int(text.shape[0])]],
            height=int(video.shape[-2]),
            width=int(video.shape[-1]),
            extra_req=extra_req,
            unpack_ctx={
                "video_denoise_mask": kwargs.get("denoise_mask"),
                "audio_denoise_mask": kwargs.get("audio_denoise_mask"),
                "audio_scale": scale,
                "audio_src": audio_src,
                "carry": carry,
                "sigma_v": sigma_v,
                "sigma_a": sigma_a,
                "shift_v": shift_v,
                "shift_a": shift_a,
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
        # ComfyUI's carried-audio transformation below already incorporates
        # the sigma-a / sigma-v derivative. Applying time_shift_slope again
        # double-scales audio and changes the coupled denoising trajectory.
        v_video = (-v_video).to(device=video_x.device, dtype=video_x.dtype)
        v_audio = (-v_audio).to(device=audio_x.device, dtype=audio_x.dtype)
        ctx = packed.unpack_ctx
        for name, velocity in (
            ("video_denoise_mask", v_video),
            ("audio_denoise_mask", v_audio),
        ):
            mask = ctx.get(name)
            if mask is not None:
                velocity.mul_(mask.to(device=velocity.device, dtype=velocity.dtype))
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
            "comfyui_cache_fp",
            "comfyui_cond_key",
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
        # Keep comfyui_cache_fp / comfyui_cond_key; later steps still send them.


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

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        kwargs = super()._sampling_params_kwargs(packed, timestep)
        kwargs.update(self.request_options)
        drop_h3_pinned_sampling_fields(kwargs)
        enable_cache_dit = self.enable_cache_dit
        if enable_cache_dit is not None:
            kwargs["enable_cache_dit"] = bool(enable_cache_dit)
        cache_params = self.cache_dit_params
        if cache_params is not None:
            kwargs["cache_dit_params"] = cache_params
        return kwargs


class FastH3Adapter(MiniMaxH3Adapter):
    model_types = ("fast_h3",)
    pipeline_class_name = "FastH3Pipeline"


class VDNH3Adapter(MiniMaxH3Adapter):
    model_types = ("vdn_h3",)
    pipeline_class_name = "VDNH3Pipeline"


# FastVideo/FastVideo-FastH3-8-Step-V2 fastvideo_inference.json; SGLDH3DistilledSigmas
# cannot see the loaded model, the executor re-checks against its model_index.json.
FASTH3_DMD_DENOISING_STEPS = (999, 874, 749, 624, 500, 375, 250, 125)
FASTH3_VIDEO_SIGMA_SHIFT = 10.0


def distilled_h3_sigmas(
    model_type: str,
    *,
    dmd_steps: tuple[int, ...] = FASTH3_DMD_DENOISING_STEPS,
    video_shift: float = FASTH3_VIDEO_SIGMA_SHIFT,
) -> list[float]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    if model_type == "fast_h3":
        # num_steps counts DiT forwards; the grid has one more point.
        return minimax_h3_time_shift_sigmas(
            num_steps=len(dmd_steps), shift_scale=video_shift, dmd_steps=dmd_steps
        )
    if model_type == "vdn_h3":
        return minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=12.0)
    raise ValueError("Distilled sigma grid requires fast_h3 or vdn_h3")


def validate_distilled_h3_step(packed, model_type, expected: list[float]):
    sigmas = packed.extra_req.get("h3_sample_sigmas")
    expected_t = torch.tensor(expected)
    actual = (
        torch.as_tensor(sigmas).detach().float().cpu() if sigmas is not None else None
    )
    if (
        actual is None
        or actual.numel() != expected_t.numel()
        or not torch.allclose(actual.reshape(-1), expected_t, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            f"{model_type} requires its trained {expected_t.numel() - 1}-NFE sigma "
            "grid; use SGLDH3DistilledSigmas"
        )
    payload = packed.extra_req.get("h3_payload") or {}
    if payload.get("refs") or (model_type == "fast_h3" and payload.get("keyframes")):
        raise ValueError(f"{model_type} does not support this reference/keyframe task")


def load_fasth3_release(model_path: str):
    import json
    import os

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
        MiniMaxH3ReleaseMetadata,
    )

    with open(os.path.join(model_path, "model_index.json")) as f:
        release = MiniMaxH3ReleaseMetadata.from_model_index(json.load(f))
    if release.dmd_denoising_steps is None:
        raise ValueError(
            "fast_h3 runtime_model_path has no trained DMD rungs "
            "(model_index.json._minimax_h3.dmd_denoising_steps); use "
            "FastVideo/FastVideo-FastH3-8-Step-V2"
        )
    index_path = os.path.join(
        model_path, "transformer", "diffusion_pytorch_model.safetensors.index.json"
    )
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        # The Diffusers export drops this derivable buffer; the model overlay adds it.
        if "rope.inv_freq" not in weight_map:
            raise ValueError(
                "fast_h3 runtime_model_path is a raw FastH3 download; point it at the "
                "materialized copy (run `sglang generate --model-path "
                "FastVideo/FastVideo-FastH3-8-Step-V2` once, then use "
                "$SGLANG_DIFFUSION_CACHE_ROOT/materialized_models/...)"
            )
    return release


class FastH3Executor(MiniMaxH3Executor):
    adapter_cls = FastH3Adapter

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        self.release = load_fasth3_release(model_path)
        self.expected_sigmas = distilled_h3_sigmas(
            "fast_h3",
            dmd_steps=self.release.dmd_denoising_steps,
            video_shift=self.release.video_sigma_shift,
        )

    def forward(self, x, timestep, context, **kwargs):
        kwargs["transformer_options"] = self._with_release_shifts(
            kwargs.get("transformer_options")
        )
        return super().forward(x, timestep, context, **kwargs)

    def _with_release_shifts(self, transformer_options):
        # The audio sigma is derived from the video sigma through these shifts, on
        # both sides of the IPC; a missing MiniMaxH3SigmaShift must not mean 12/3.
        opts = dict(transformer_options or {})
        for key, trained in (
            ("minimax_h3_sigma_shift_video", self.release.video_sigma_shift),
            ("minimax_h3_sigma_shift_audio", self.release.audio_sigma_shift),
        ):
            if opts.get(key) is None:
                opts[key] = trained
            elif abs(float(opts[key]) - trained) > 1e-6:
                raise ValueError(
                    f"FastH3 was distilled with sigma shift video/audio "
                    f"{self.release.video_sigma_shift:g}/{self.release.audio_sigma_shift:g}; "
                    f"got {key}={float(opts[key]):g}"
                )
        return opts

    def _execute_packed(self, packed, x, timestep):
        validate_distilled_h3_step(packed, "fast_h3", self.expected_sigmas)
        return super()._execute_packed(packed, x, timestep)

    def _sampling_params_kwargs(self, packed, timestep):
        kwargs = super()._sampling_params_kwargs(packed, timestep)
        kwargs["num_inference_steps"] = len(self.expected_sigmas) - 1
        return kwargs


class VDNH3Executor(MiniMaxH3Executor):
    adapter_cls = VDNH3Adapter

    def _execute_packed(self, packed, x, timestep):
        validate_distilled_h3_step(packed, "vdn_h3", distilled_h3_sigmas("vdn_h3"))
        return super()._execute_packed(packed, x, timestep)

    def _sampling_params_kwargs(self, packed, timestep):
        kwargs = super()._sampling_params_kwargs(packed, timestep)
        kwargs["num_inference_steps"] = 8
        return kwargs
