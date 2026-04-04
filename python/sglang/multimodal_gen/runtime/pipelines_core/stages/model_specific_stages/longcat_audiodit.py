# SPDX-License-Identifier: Apache-2.0
"""LongCat-AudioDiT stages.

Three-stage pipeline: BeforeDenoising -> Denoising -> Decoding.

Reference: https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/inference.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sglang.multimodal_gen.configs.sample.longcat_audiodit import (
    _require_positive_duration_seconds,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-request CFG / APG (not the shared PipelineConfig.cfg_policy)
# ---------------------------------------------------------------------------


class _MomentumBuffer:
    """Running-average buffer for APG momentum."""

    def __init__(self, momentum: float = -0.3) -> None:
        self.momentum = momentum
        self.running_average: torch.Tensor | None = None

    def update(self, update_value: torch.Tensor) -> None:
        if self.running_average is None:
            self.running_average = update_value
            return
        self.running_average = update_value + self.momentum * self.running_average


def _project(
    v0: torch.Tensor, v1: torch.Tensor, dims=(-1, -2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose *v0* into parallel and orthogonal components w.r.t. *v1*."""
    dtype = v0.dtype
    device = v0.device
    if device.type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype=dtype, device=device), v0_orthogonal.to(
        dtype=dtype, device=device
    )


def _apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: _MomentumBuffer | None = None,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
    dims=(-1, -2),
) -> torch.Tensor:
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + guidance_scale * normalized_update


@dataclass
class AudioDiTCFGPolicy(CFGPolicy):
    """CFG policy supporting both standard CFG and APG for LongCat-AudioDiT.

    For ``guidance_method="cfg"``: standard CFG linear extrapolation.
    For ``guidance_method="apg"``: Adaptive Projected Guidance with momentum
    buffer, orthogonal projection, and sample-space computation.

    APG operates in sample space (not velocity space). The current latent
    ``x`` and timestep ``t`` are stashed on ``batch`` by
    ``rewrite_prompt_region`` as ``batch._current_latent`` /
    ``batch._current_t`` so ``combine()`` can compute
    ``x + (1 - t) * velocity``.
    """

    guidance_method: str = "cfg"
    momentum: float = -0.3
    eta: float = 0.5
    norm_threshold: float = 0.0
    apg_buffer: _MomentumBuffer | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.guidance_method not in ("cfg", "apg"):
            raise ValueError(
                f"Unknown guidance_method '{self.guidance_method}', must be 'cfg' or 'apg'"
            )
        if self.guidance_method == "apg":
            self.apg_buffer = _MomentumBuffer(momentum=self.momentum)

    def combine(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        batch: Req,
        cfg_scale: float,
        pipeline_config: Any,
        *,
        cfg_parallel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(predictions) == 1:
            return predictions[0]

        pred = predictions[0]
        null_pred = predictions[1]
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(null_pred, tuple):
            null_pred = null_pred[0]

        if self.guidance_method == "cfg":
            return super().combine(
                predictions,
                batch,
                cfg_scale,
                pipeline_config,
                cfg_parallel=cfg_parallel,
            )

        x = batch._current_latent
        t = batch._current_t
        latent_len = batch._audio_prompt_latent_len

        x_s = x[:, latent_len:]
        pred_s = pred[:, latent_len:]
        null_s = null_pred[:, latent_len:]

        pred_sample = x_s + (1 - t) * pred_s
        null_sample = x_s + (1 - t) * null_s

        out = _apg_forward(
            pred_sample,
            null_sample,
            cfg_scale,
            self.apg_buffer,
            eta=self.eta,
            norm_threshold=self.norm_threshold,
            dims=[-1, -2],
        )

        # AudioDiTFlowMatchScheduler.set_timesteps is linspace(0, 1, N+1)[:-1],
        # so max t = (N-1)/N and 1-t >= 1/N > 0. Including t=1.0 would divide by 0.
        out = (out - x_s) / (1 - t)
        return F.pad(out, (0, 0, latent_len, 0), value=0.0)


# ---------------------------------------------------------------------------
# Text / audio utilities (adapted from LongCat-AudioDiT/utils.py)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["“”‘’]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)


def _coerce_single_prompt(prompt) -> str:
    """LongCat-AudioDiT runs one utterance per request.

    ``supports_dynamic_batching`` is False; reject a merged prompt list here
    as a backstop.
    """
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise ValueError(
                "LongCat-AudioDiT supports one prompt per request, "
                f"got {len(prompt)}"
            )
        prompt = prompt[0]
    if prompt is None:
        return ""
    return str(prompt)


def _coerce_seed(seed) -> int:
    """One generator per request; SamplingParams.seed may be ``int | list[int]``."""
    if isinstance(seed, (list, tuple)):
        if not seed:
            raise ValueError("seed list must not be empty")
        seed = seed[0]
    return int(seed)


def _resolve_duration_frames(
    *,
    gen_text: str,
    prompt_dur: int,
    prompt_time: float,
    prompt_text: str | None,
    duration_seconds: float | None,
    sr: int,
    full_hop: int,
    max_duration: float,
) -> int:
    """Total latent frames (prompt region + generated region).

    ``duration_seconds``, when set, is the generated-audio length (the wav
    written after the prompt region is stripped). Clone still adds
    ``prompt_dur`` internally so conditioning has a canvas.
    """
    max_frames = int(max_duration * sr // full_hop)
    if duration_seconds is not None:
        _require_positive_duration_seconds(duration_seconds)
        gen_frames = int(duration_seconds * sr // full_hop)
    elif prompt_dur > 0:
        dur_sec = _approx_duration_from_text(
            gen_text, max_duration=max(0.0, max_duration - prompt_time)
        )
        if prompt_text:
            approx_pd = _approx_duration_from_text(
                prompt_text, max_duration=max_duration
            )
            if approx_pd > 0:
                ratio = float(np.clip(prompt_time / approx_pd, 1.0, 1.5))
                dur_sec = dur_sec * ratio
        gen_frames = int(dur_sec * sr // full_hop)
    else:
        dur_sec = _approx_duration_from_text(gen_text, max_duration=max_duration)
        gen_frames = int(dur_sec * sr // full_hop)
    gen_frames = max(1, gen_frames)
    return max(1, min(gen_frames + prompt_dur, max_frames))


def _load_audio_tensor(wav_path: str, sr: int) -> torch.Tensor:
    """Load a WAV file and return a (1, num_samples) float32 tensor."""
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required to load clone reference audio. "
            "Install with: pip install 'sglang[diffusion]'"
        ) from e

    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)  # (1, T)


def _lens_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.BoolTensor:
    if length is None:
        length = lengths.amax()
    seq = torch.arange(length, device=lengths.device)
    return seq[None, :] < lengths[:, None]


def _padding_mask_if_needed(
    lengths: torch.Tensor,
    length: int,
    *,
    has_padding: bool,
) -> torch.BoolTensor | None:
    """Build a key-padding mask only when some positions are pad.

    Callers must decide ``has_padding`` from host-side lengths (tokenizer
    attention_mask, known-equal durations). Do not probe a CUDA mask with
    ``.all()`` on the attention hot path.
    """
    if not has_padding:
        return None
    return _lens_to_mask(lengths, length=length)


# ---------------------------------------------------------------------------
# BeforeDenoisingStage
# ---------------------------------------------------------------------------


class LongCatAudioDiTBeforeDenoisingStage(PipelineStage):
    """Pre-processing stage for LongCat-AudioDiT.

    Produces a ``Req`` batch with all fields required by
    ``LongCatAudioDiTDenoisingStage``: ``latents``, ``timesteps``,
    ``scheduler``, ``prompt_embeds``, ``negative_prompt_embeds``,
    ``do_classifier_free_guidance``, and per-request CFG/APG on
    ``batch.extra["cfg_policy"]`` (do not write the shared pipeline config).
    Audio-specific conditioning is stashed as ``batch._audio_*``.
    """

    def __init__(self, model, tokenizer, scheduler):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        params = batch.sampling_params
        config = self.model.config

        # ── resolve parameters ─────────────────────────────────────────
        gen_text = _coerce_single_prompt(params.prompt)

        prompt_audio_path = params.prompt_audio_path
        prompt_text = params.prompt_text
        duration_seconds = params.duration_seconds
        guidance_method = params.guidance_method
        if guidance_method not in ("cfg", "apg"):
            raise ValueError(
                f"Unknown guidance_method '{guidance_method}', must be 'cfg' or 'apg'"
            )
        cfg_strength: float = params.guidance_scale
        steps: int = params.num_inference_steps

        sr = config.sampling_rate
        full_hop = config.latent_hop
        max_duration = config.max_wav_duration

        # Per-request CPU generator. Voice-cloning VAE sampling uses a second
        # generator with the same seed so it does not consume the noise stream.
        seed = _coerce_seed(params.seed)
        batch.generator = torch.Generator(device="cpu").manual_seed(seed)
        vae_generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── prompt audio (voice cloning) ────────────────────────────────
        prompt_latent = None
        prompt_dur = 0
        prompt_time = 0.0
        if prompt_audio_path is not None:
            if not os.path.isfile(prompt_audio_path):
                raise ValueError(
                    "prompt_audio_path must be a local file readable by the "
                    f"server process, got: {prompt_audio_path}"
                )
            prompt_wav_1d = _load_audio_tensor(prompt_audio_path, sr)  # (1, T)
            prompt_wav = prompt_wav_1d.unsqueeze(0)  # (1, 1, T)
            # Encode ONCE — the latent is reused across all denoising steps.
            prompt_latent, prompt_dur = self.model.encode_prompt_audio(
                prompt_wav.to(device), generator=vae_generator
            )
            prompt_time = prompt_dur * full_hop / sr  # seconds
            if prompt_text:
                full_text = (
                    f"{_normalize_text(prompt_text)} {_normalize_text(gen_text)}"
                )
            else:
                full_text = _normalize_text(gen_text)
        else:
            full_text = _normalize_text(gen_text)

        duration = _resolve_duration_frames(
            gen_text=gen_text,
            prompt_dur=prompt_dur,
            prompt_time=prompt_time,
            prompt_text=prompt_text,
            duration_seconds=duration_seconds,
            sr=sr,
            full_hop=full_hop,
            max_duration=max_duration,
        )

        logger.info(
            f"LongCatAudioDiT: text='{full_text[:80]}...', "
            f"duration={duration} frames ({duration * full_hop / sr:.2f}s), "
            f"steps={steps}, cfg={cfg_strength}, method={guidance_method}"
        )

        # ── text encoding (UMT5) ────────────────────────────────────────
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask_cpu = inputs.attention_mask
        has_text_padding = bool((attention_mask_cpu == 0).any())
        attention_mask = attention_mask_cpu.to(device)

        text_condition = self.model.encode_text(input_ids, attention_mask)
        text_condition_len = attention_mask.sum(dim=1)

        bsz = text_condition.shape[0]
        latent_dim = config.latent_dim
        max_duration_frames = int(max_duration * sr // full_hop)
        total_duration = max(1, min(duration, max_duration_frames))

        # ── negative text (for CFG) ─────────────────────────────────────
        neg_text = torch.zeros_like(text_condition)

        # ── latent conditioning ─────────────────────────────────────────
        latent_len = prompt_dur
        if prompt_latent is not None:
            gen_len = max(total_duration - latent_len, 0)
            if total_duration < latent_len:
                raise ValueError(
                    f"Prompt audio ({latent_len} frames) exceeds total duration "
                    f"({total_duration} frames). Use a shorter prompt or increase duration."
                )
            latent_cond = F.pad(prompt_latent, (0, 0, 0, gen_len))
            empty_latent_cond = torch.zeros_like(latent_cond)
        else:
            latent_cond = torch.zeros(bsz, total_duration, latent_dim, device=device)
            empty_latent_cond = latent_cond

        # ── masks ───────────────────────────────────────────────────────
        # Uniform duration (torch.full) has no pad. Text pad is known from the
        # host tokenizer mask — do not .all() these on CUDA in every attn.
        duration_tensor = torch.full(
            (bsz,), total_duration, device=device, dtype=torch.long
        )
        mask = _padding_mask_if_needed(
            duration_tensor, total_duration, has_padding=False
        )
        text_mask = _padding_mask_if_needed(
            text_condition_len,
            text_condition.shape[1],
            has_padding=has_text_padding,
        )

        # ── initial noise ───────────────────────────────────────────────
        noise_generator = batch.generator
        if isinstance(noise_generator, list):
            noise_generator = noise_generator[0] if noise_generator else None
        y0 = []
        for dur in duration_tensor:
            noise = torch.randn(
                dur.item(),
                latent_dim,
                generator=noise_generator,
            )
            y0.append(noise.to(device))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        prompt_noise = y0[:, :latent_len].clone() if latent_len > 0 else None

        # ── scheduler / timesteps ───────────────────────────────────────
        batch.num_inference_steps = steps
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        scheduler.set_timesteps(steps, device=device)
        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps

        # ── populate batch for DenoisingStage ───────────────────────────
        batch.latents = y0
        batch.raw_latent_shape = y0.shape
        batch.prompt_embeds = [text_condition]
        batch.negative_prompt_embeds = [neg_text]
        batch.do_classifier_free_guidance = cfg_strength >= 1e-5
        batch.guidance_scale = cfg_strength

        # Audio-specific conditioning (read by LongCatAudioDiTDenoisingStage).
        batch._audio_prompt_latent_len = latent_len
        batch._audio_prompt_noise = prompt_noise
        batch._audio_latent_cond = latent_cond
        batch._audio_empty_latent_cond = empty_latent_cond
        batch._audio_mask = mask
        batch._audio_cond_mask = text_mask
        batch._audio_text_condition_len = text_condition_len
        batch._audio_repa_dit_layer = int(config.repa_dit_layer)

        # Per-request policy: never write back to shared pipeline_config.
        batch.extra["cfg_policy"] = AudioDiTCFGPolicy(
            guidance_method=guidance_method,
            momentum=-0.3,
            eta=0.5,
            norm_threshold=0.0,
        )

        return batch


# ---------------------------------------------------------------------------
# DenoisingStage
# ---------------------------------------------------------------------------


def resolve_cfg_policy(batch, pipeline_config):
    """Prefer the per-request policy stashed on ``batch.extra``."""
    policy = batch.extra.get("cfg_policy")
    if policy is not None:
        return policy
    return pipeline_config.cfg_policy


def rewrite_prompt_region(latent_model_input, timestep, batch):
    """Rewrite the prompt region before each forward; stash ``x`` / ``t`` for APG.

    The returned tensor is a clone when rewritten so the scheduler's internal
    state is not mutated in-place.
    """
    latent_len = batch._audio_prompt_latent_len
    prompt_noise = batch._audio_prompt_noise
    latent_cond = batch._audio_latent_cond

    batch._current_latent = latent_model_input
    batch._current_t = timestep

    if latent_len == 0 or prompt_noise is None or latent_cond is None:
        return latent_model_input

    latent_model_input = latent_model_input.clone()
    latent_model_input[:, :latent_len] = (
        prompt_noise * (1 - timestep) + latent_cond[:, :latent_len] * timestep
    )
    batch._current_latent = latent_model_input
    return latent_model_input


def prepare_branch_latent(latent_model_input, batch):
    """Zero the prompt region on the uncond CFG branch."""
    latent_len = batch._audio_prompt_latent_len
    if latent_len == 0 or not batch.is_cfg_negative:
        return latent_model_input
    latent_model_input = latent_model_input.clone()
    latent_model_input[:, :latent_len] = 0
    return latent_model_input


class LongCatAudioDiTDenoisingStage(DenoisingStage):
    """Standard Euler CFG loop with LongCat prompt-region conditioning.

    Prompt-region rewrite and uncond zeroing run in ``_predict_noise`` so
    the shared ``_run_denoising_step`` is used as-is.
    """

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        ctx = super()._prepare_denoising_loop(batch, server_args)
        policy = resolve_cfg_policy(batch, server_args.pipeline_config)
        ctx.cfg_policy = policy.build(
            batch, ctx.image_kwargs, ctx.pos_cond_kwargs, ctx.neg_cond_kwargs
        )
        return ctx

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance: torch.Tensor,
        **kwargs,
    ):
        # Parent predict_fn calls configure_batch (sets batch.is_cfg_negative)
        # then set_forward_context, then this method.
        try:
            batch = get_forward_context().forward_batch
        except AssertionError:
            batch = None
        if batch is not None:
            latent_model_input = rewrite_prompt_region(
                latent_model_input, timestep, batch
            )
            latent_model_input = prepare_branch_latent(latent_model_input, batch)
        return super()._predict_noise(
            current_model,
            latent_model_input,
            timestep,
            target_dtype,
            guidance,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# DecodingStage (audio-specific, bypasses standard 5D DecodingStage)
# ---------------------------------------------------------------------------


class LongCatAudioDiTDecodingStage(PipelineStage):
    """Decode the generated latent to a waveform via WAV-VAE.

    The standard ``DecodingStage`` assumes a 5D spatial VAE and does
    ``(image / 2 + 0.5).clamp(0, 1)`` — wrong for 1D audio.  This stage
    calls ``vae.decode()`` directly on the 3D latent ``[B, C, T]``.
    """

    def __init__(self, vae, model):
        super().__init__()
        self.vae = vae
        self.model = model

    @property
    def role_affinity(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        return RoleType.DECODER

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        sr = self.model.config.sampling_rate

        # The prompt region was already stripped by post_denoising_loop.
        pred_latent = batch.latents
        pred_latent = pred_latent.permute(0, 2, 1).float()  # [B, C, T]
        waveform = self.vae.decode(pred_latent).squeeze(1)  # [B, T]

        return OutputBatch(
            output=[waveform],
            audio_sample_rate=sr,
            metrics=batch.metrics,
        )
