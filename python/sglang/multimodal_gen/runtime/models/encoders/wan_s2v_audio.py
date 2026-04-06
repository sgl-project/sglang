# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from sglang.multimodal_gen.configs.models.encoders.wan_s2v_audio import (
    WanS2VAudioEncoderConfig,
)
from sglang.multimodal_gen.runtime.models.encoders.base import AudioEncoder

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def get_sample_indices(
    original_fps: int,
    total_frames: int,
    target_fps: int,
    num_sample: int,
    fixed_start: int | None = None,
) -> np.ndarray:
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)

    start_time = start_frame / original_fps
    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)
    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    return np.clip(frame_indices, 0, total_frames - 1)


def linear_interpolation(
    features: torch.Tensor,
    input_fps: int,
    output_fps: int,
    output_len: int | None = None,
) -> torch.Tensor:
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)


class WanS2VAudioEncoder(AudioEncoder):
    def __init__(
        self,
        config: WanS2VAudioEncoderConfig,
        component_model_path: str,
        *,
        dtype: torch.dtype = torch.float32,
        target_device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(config)
        model_path = component_model_path
        nested_model_path = (
            pathlib.Path(component_model_path) / config.arch_config.model_id
        )
        if nested_model_path.exists():
            model_path = str(nested_model_path)
        self.component_model_path = model_path
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_path,
            dtype=dtype,
        )
        self.model = self.model.to(target_device)
        self.video_rate = config.arch_config.video_rate
        self.sample_rate = config.arch_config.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        if sf is None:
            raise ImportError("soundfile is required to load Wan S2V audio")
        audio_input, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=False
        )
        if isinstance(audio_input, np.ndarray) and audio_input.ndim > 1:
            audio_input = audio_input.mean(axis=-1)
        if sample_rate != self.sample_rate:
            waveform = torch.from_numpy(np.asarray(audio_input, dtype=np.float32))
            waveform = waveform.view(1, 1, -1)
            target_len = max(
                1, int(round(waveform.shape[-1] * self.sample_rate / sample_rate))
            )
            waveform = F.interpolate(
                waveform,
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            audio_input = waveform.view(-1).numpy()
            sample_rate = self.sample_rate
        return np.asarray(audio_input, dtype=np.float32), sample_rate

    def extract_audio_feat(
        self,
        audio_path: str,
        *,
        return_all_layers: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        audio_input, sample_rate = self._load_audio(audio_path)
        input_values = self.processor(
            audio_input,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_values

        result = self.model(
            input_values.to(self.device),
            output_hidden_states=True,
        )
        feat = (
            torch.cat(result.hidden_states)
            if return_all_layers
            else result.hidden_states[-1]
        )
        feat = linear_interpolation(feat, input_fps=50, output_fps=self.video_rate)
        return feat.to(dtype)

    def get_audio_embed_bucket_fps(
        self,
        audio_embed: torch.Tensor,
        *,
        fps: int = 16,
        batch_frames: int = 81,
        m: int = 0,
    ) -> tuple[torch.Tensor, int]:
        num_layers, audio_frame_num, audio_dim = audio_embed.shape
        return_all_layers = num_layers > 1

        scale = self.video_rate / fps
        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
        bucket_num = min_batch_num * batch_frames
        padd_audio_num = (
            math.ceil(min_batch_num * batch_frames / fps * self.video_rate)
            - audio_frame_num
        )
        batch_idx = get_sample_indices(
            original_fps=self.video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0,
        )
        batch_audio_embeds = []
        audio_sample_stride = int(self.video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:
                chosen_idx = list(
                    range(
                        bi - m * audio_sample_stride,
                        bi + (m + 1) * audio_sample_stride,
                        audio_sample_stride,
                    )
                )
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [
                    audio_frame_num - 1 if c >= audio_frame_num else c
                    for c in chosen_idx
                ]
                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                        start_dim=-2, end_dim=-1
                    )
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                shape = (
                    [num_layers, audio_dim * (2 * m + 1)]
                    if return_all_layers
                    else [audio_dim * (2 * m + 1)]
                )
                frame_audio_embed = torch.zeros(
                    shape,
                    device=audio_embed.device,
                    dtype=audio_embed.dtype,
                )
            batch_audio_embeds.append(frame_audio_embed.unsqueeze(0))
        return torch.cat(batch_audio_embeds, dim=0), min_batch_num

    def encode_audio(
        self,
        audio_path: str,
        *,
        infer_frames: int,
        fps: int,
        dtype: torch.dtype,
        m: int = 0,
    ) -> tuple[torch.Tensor, int]:
        embeddings = self.extract_audio_feat(
            audio_path, return_all_layers=True, dtype=torch.float32
        )
        bucket, num_repeat = self.get_audio_embed_bucket_fps(
            embeddings,
            fps=fps,
            batch_frames=infer_frames,
            m=m,
        )
        bucket = bucket.to(self.device, dtype).unsqueeze(0)
        if bucket.ndim == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.ndim == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        return bucket, num_repeat

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "WanS2VAudioEncoder is used via extract_audio_feat/encode_audio, not forward()."
        )


EntryClass = WanS2VAudioEncoder
