"""Monolithic inference stage for LongCat-AudioDiT.

LongCat-AudioDiT is a Conditional Flow Matching TTS model whose entire generation
pipeline (text encoding → ODE solve → VAE decode) is encapsulated in
``AudioDiTModel.forward``.  Rather than splitting this into the standard
TextEncoding / Denoising / Decoding stages, we run everything here in a single
stage and return the ``OutputBatch`` directly.

This approach is correct because:
- The ODE solver is a custom inline Euler integrator tightly coupled to the model.
- The VAE encode/decode for prompt audio must happen inside the same forward pass.
- The CFG / APG guidance is woven into the ODE function closure.

Reference: https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/inference.py
"""

import re
from typing import Optional

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Text / audio utilities (adapted from LongCat-AudioDiT/utils.py)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["""' "\u2018\u2019\u201c\u201d]", " ", text)
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


def _load_audio_tensor(wav_path: str, sr: int) -> torch.Tensor:
    """Load a WAV file and return a (1, num_samples) float32 tensor."""
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required for prompt audio loading. "
            "Install it with: pip install librosa"
        ) from e
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)  # (1, T)


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------


class LongCatAudioDiTInferenceStage(PipelineStage):
    """Monolithic stage that drives the full LongCat-AudioDiT inference.

    Accepts a ``Req`` whose ``sampling_params`` is a
    ``LongCatAudioDiTSamplingParams`` instance and returns an ``OutputBatch``
    with ``audio`` set to the generated waveform tensor
    ``(batch=1, num_samples)`` and ``audio_sample_rate`` set to 24000.
    """

    def __init__(self, model, tokenizer):
        """
        Args:
            model: ``AudioDiTModel`` instance (already on the target device).
            tokenizer: HuggingFace tokenizer for the UMT5 text encoder.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_prompt_audio(
        self,
        prompt_audio_path: str,
        prompt_text: Optional[str],
        gen_text: str,
        device: torch.device,
    ):
        """Encode the prompt audio and compute total duration.

        Returns:
            prompt_wav: (1, 1, num_samples) tensor on *cpu* (model.forward moves it)
            full_text:  combined "[prompt_text] [gen_text]" string for tokenization
            duration:   total latent-frame count (prompt + generation)
        """
        sr = self.model.config.sampling_rate
        full_hop = self.model.config.latent_hop
        max_duration = self.model.config.max_wav_duration

        prompt_wav_1d = _load_audio_tensor(prompt_audio_path, sr)  # (1, T)
        prompt_wav = prompt_wav_1d.unsqueeze(0)  # (1, 1, T)

        # Run the VAE encode to get the exact prompt frame count.
        # model.forward will call encode_prompt_audio again to get the latent.
        _, prompt_dur = self.model.encode_prompt_audio(prompt_wav.to(device))

        prompt_time = prompt_dur * full_hop / sr  # seconds

        # Duration estimation for generated part
        dur_sec = _approx_duration_from_text(
            gen_text, max_duration=max_duration - prompt_time
        )
        if prompt_text:
            approx_pd = _approx_duration_from_text(
                prompt_text, max_duration=max_duration
            )
            if approx_pd > 0:
                ratio = float(np.clip(prompt_time / approx_pd, 1.0, 1.5))
                dur_sec = dur_sec * ratio

        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

        # Build combined text
        if prompt_text:
            full_text = f"{_normalize_text(prompt_text)} {_normalize_text(gen_text)}"
        else:
            full_text = _normalize_text(gen_text)

        return prompt_wav, full_text, duration

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        device = get_local_torch_device()
        params = batch.sampling_params

        # ── resolve text ──────────────────────────────────────────────
        gen_text = params.prompt
        if isinstance(gen_text, list):
            gen_text = gen_text[0]

        prompt_audio_path: Optional[str] = getattr(params, "prompt_audio_path", None)
        prompt_text: Optional[str] = getattr(params, "prompt_text", None)
        duration_seconds: Optional[float] = getattr(params, "duration_seconds", None)
        guidance_method: str = getattr(params, "guidance_method", "cfg")
        cfg_strength: float = params.guidance_scale
        steps: int = params.num_inference_steps

        sr = self.model.config.sampling_rate
        full_hop = self.model.config.latent_hop
        max_duration = self.model.config.max_wav_duration

        # ── seed ──────────────────────────────────────────────────────
        seed = params.seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # ── prompt audio (voice cloning) ──────────────────────────────
        prompt_wav = None
        if prompt_audio_path is not None:
            prompt_wav, full_text, duration = self._encode_prompt_audio(
                prompt_audio_path, prompt_text, gen_text, device
            )
        else:
            full_text = _normalize_text(gen_text)
            # Duration from explicit seconds or auto-estimate
            if duration_seconds is not None:
                duration = int(duration_seconds * sr // full_hop)
                duration = min(duration, int(max_duration * sr // full_hop))
            else:
                dur_sec = _approx_duration_from_text(
                    full_text, max_duration=max_duration
                )
                duration = int(dur_sec * sr // full_hop)

        logger.info(
            f"LongCatAudioDiT: text='{full_text[:80]}...', "
            f"duration={duration} frames ({duration * full_hop / sr:.2f}s), "
            f"steps={steps}, cfg={cfg_strength}, method={guidance_method}"
        )

        # ── tokenize ──────────────────────────────────────────────────
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # ── run model forward (ODE solve + VAE decode) ────────────────
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_audio=prompt_wav,
            duration=duration,
            steps=steps,
            cfg_strength=cfg_strength,
            guidance_method=guidance_method,
        )

        waveform = output.waveform  # (1, num_samples) float32

        # output holds per-sample tensors iterated by save_outputs; for
        # DataType.AUDIO, post_process_sample reads each element directly as the
        # waveform.  audio= mirrors the waveform so GenerationResult.audio is
        # populated for Python API callers (save_outputs appends output_batch.audio
        # into audios_out for non-VIDEO data types).
        return OutputBatch(
            output=[waveform],
            audio=waveform,
            audio_sample_rate=sr,
            metrics=batch.metrics,
        )
