# SPDX-License-Identifier: Apache-2.0
"""JoyEcho memory bank utilities and memory-related pipeline stages."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TVF

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# --- Audio peak-window helpers ---


def latent_window_size_to_pixel_window_size(
    latent_window_size: int,
    *,
    downsample_factor: int,
    is_causal: bool = True,
) -> int:
    if latent_window_size <= 0:
        raise ValueError(
            f"latent_window_size must be positive, got {latent_window_size}"
        )
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be positive, got {downsample_factor}")

    pixel_window_size = int(latent_window_size) * int(downsample_factor)
    if is_causal:
        pixel_window_size = max(pixel_window_size - (int(downsample_factor) - 1), 1)
    return pixel_window_size


def select_max_response_audio_window_with_bounds(
    segment: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if segment.dim() != 4:
        raise ValueError(
            f"Expected segment shape [B, C, T, F], got {tuple(segment.shape)}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    num_time_steps = segment.shape[2]
    if num_time_steps <= 0:
        raise ValueError("Cannot select from an empty audio segment")

    scan_stride = max(1, window_size // 4)
    offsets = torch.arange(window_size, device=segment.device)
    max_start_idx = (
        num_time_steps - window_size
        if num_time_steps >= window_size
        else num_time_steps - 1
    )
    candidate_start_indices = list(range(0, max_start_idx + 1, scan_stride))
    if candidate_start_indices[-1] != max_start_idx:
        candidate_start_indices.append(max_start_idx)

    candidate_windows = []
    candidate_scores = []
    candidate_start_indices_tensor = torch.tensor(
        candidate_start_indices, device=segment.device, dtype=torch.long
    )
    for start_idx in candidate_start_indices:
        gather_indices = (start_idx + offsets).clamp(0, num_time_steps - 1).long()
        window = segment.index_select(dim=2, index=gather_indices)
        candidate_windows.append(window)
        candidate_scores.append(window.float().exp().sum(dim=(1, 2, 3)))

    scores = torch.stack(candidate_scores, dim=1)
    best_window_indices = scores.argmax(dim=1)
    best_start_indices = candidate_start_indices_tensor[best_window_indices]
    best_end_indices = torch.clamp(
        best_start_indices + window_size - 1, max=num_time_steps - 1
    )
    selected_windows = torch.cat(
        [
            candidate_windows[int(best_window_indices[batch_index])][
                batch_index : batch_index + 1
            ]
            for batch_index in range(segment.shape[0])
        ],
        dim=0,
    )
    return selected_windows, best_start_indices, best_end_indices


def select_audio_window_with_bounds(
    segment: Tensor,
    window_size: int,
    *,
    mode: Literal["max_response", "random"] = "random",
    rng: random.Random | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if mode == "max_response":
        return select_max_response_audio_window_with_bounds(segment, window_size)
    if mode == "random":
        offsets = torch.arange(window_size, device=segment.device, dtype=torch.long)
        num_time_steps = segment.shape[2]
        max_start_idx = max(0, num_time_steps - window_size)
        start_indices = torch.randint(
            0, max_start_idx + 1, (segment.shape[0],), device=segment.device
        )
        selected_windows = torch.cat(
            [
                segment[
                    batch_index : batch_index + 1,
                    :,
                    (start_indices[batch_index] + offsets).clamp(0, num_time_steps - 1),
                    :,
                ]
                for batch_index in range(segment.shape[0])
            ],
            dim=0,
        )
        end_indices = torch.clamp(
            start_indices + window_size - 1, max=num_time_steps - 1
        )
        return selected_windows, start_indices, end_indices
    raise ValueError(f"Unsupported audio window selection mode: {mode}")


def mel_window_bounds_to_seconds(
    start_index: int,
    end_index: int,
    *,
    hop_length: int,
    sample_rate: int,
) -> tuple[float, float]:
    start_time_sec = float(start_index * hop_length) / float(sample_rate)
    end_time_sec = float((end_index + 1) * hop_length) / float(sample_rate)
    return start_time_sec, end_time_sec


def select_video_frame_indices_from_time_range(
    *,
    num_frames: int,
    fps: float,
    start_time_sec: float,
    end_time_sec: float,
    count: int = 1,
    mode: Literal["first", "random", "center"] = "center",
    rng: random.Random | None = None,
) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    start_frame = int(math.ceil(start_time_sec * fps))
    end_frame = int(math.ceil(end_time_sec * fps)) - 1
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(0, min(end_frame, num_frames - 1))

    if end_frame < start_frame:
        center_time_sec = max(0.0, 0.5 * (start_time_sec + end_time_sec))
        center_frame = int(round(center_time_sec * fps))
        candidate_frames = [max(0, min(center_frame, num_frames - 1))]
    else:
        candidate_frames = list(range(start_frame, end_frame + 1))

    if mode == "center":
        if len(candidate_frames) <= count:
            selected = candidate_frames[:]
        else:
            center_offset = max(0, (len(candidate_frames) - count) // 2)
            selected = candidate_frames[center_offset : center_offset + count]
    elif mode == "first":
        selected = candidate_frames[:count]
    else:
        rng = rng or random
        selected = (
            candidate_frames[:]
            if len(candidate_frames) <= count
            else sorted(rng.sample(candidate_frames, count))
        )

    if len(selected) < count:
        selected.extend([selected[-1]] * (count - len(selected)))
    return selected


# --- Slot-aware attention masks ---


def memory_slot_ranges(total_seq_len: int, num_slots: int) -> list[tuple[int, int]]:
    if total_seq_len <= 0 or num_slots <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    for slot_idx in range(num_slots):
        end = round((slot_idx + 1) * total_seq_len / num_slots)
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def memory_slot_ranges_from_lengths(
    lengths: tuple[int, ...] | None,
    *,
    total_seq_len: int,
    num_slots: int,
) -> list[tuple[int, int]]:
    if not lengths or len(lengths) != num_slots:
        return memory_slot_ranges(total_seq_len, num_slots)

    ranges: list[tuple[int, int]] = []
    start = 0
    for raw_length in lengths:
        length = max(0, int(raw_length))
        end = min(start + length, total_seq_len)
        if end > start:
            ranges.append((start, end))
        start = end
    if start != total_seq_len:
        return memory_slot_ranges(total_seq_len, num_slots)
    return ranges


def build_paired_memory_cross_mask(
    *,
    batch_size: int,
    query_memory_seq_len: int,
    query_target_seq_len: int,
    kv_memory_seq_len: int,
    kv_target_seq_len: int,
    num_memory_slots: int,
    device: torch.device,
    query_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
    kv_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    query_total_seq_len = query_memory_seq_len + query_target_seq_len
    kv_total_seq_len = kv_memory_seq_len + kv_target_seq_len
    mask = torch.zeros(
        batch_size,
        query_total_seq_len,
        kv_total_seq_len,
        dtype=torch.bool,
        device=device,
    )

    for batch_idx in range(batch_size):
        query_lengths = (
            query_segment_lengths[batch_idx]
            if query_segment_lengths is not None
            and batch_idx < len(query_segment_lengths)
            else None
        )
        kv_lengths = (
            kv_segment_lengths[batch_idx]
            if kv_segment_lengths is not None and batch_idx < len(kv_segment_lengths)
            else None
        )
        query_ranges = memory_slot_ranges_from_lengths(
            query_lengths,
            total_seq_len=query_memory_seq_len,
            num_slots=num_memory_slots,
        )
        kv_ranges = memory_slot_ranges_from_lengths(
            kv_lengths,
            total_seq_len=kv_memory_seq_len,
            num_slots=num_memory_slots,
        )
        for (q_start, q_end), (k_start, k_end) in zip(
            query_ranges, kv_ranges, strict=False
        ):
            mask[batch_idx, q_start:q_end, k_start:k_end] = True

    if query_target_seq_len > 0 and kv_target_seq_len > 0:
        mask[:, query_memory_seq_len:, kv_memory_seq_len:] = True
    return mask


def build_memory_self_attention_block_mask(
    *,
    batch_size: int,
    memory_seq_len: int,
    target_seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if memory_seq_len <= 0:
        return None

    total_seq_len = memory_seq_len + target_seq_len
    attention_mask = torch.ones(
        batch_size,
        total_seq_len,
        total_seq_len,
        dtype=torch.bool,
        device=device,
    )
    attention_mask[:, :, :memory_seq_len] = False
    attention_mask[:, :memory_seq_len, :] = False
    attention_mask[:, :memory_seq_len, :memory_seq_len] = True
    return attention_mask


# --- Memory VAE encode ---


def frames_to_video_tensor(
    frames: list[Image.Image], target_h: int, target_w: int
) -> torch.Tensor:
    tensors = []
    for idx, image in enumerate(frames):
        if image.size != (target_w, target_h):
            raise ValueError(
                f"Frame size mismatch at index {idx}: got={image.size}, "
                f"expected={(target_w, target_h)}"
            )
        tensor = TVF.to_tensor(image)
        tensors.append(tensor * 2.0 - 1.0)
    return torch.stack(tensors, dim=1).contiguous()


@torch.no_grad()
def encode_memory_frames_batch(
    *,
    video_vae: Any,
    batch_memory_frames: list[list[Image.Image | list[Image.Image]]],
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
    pipeline_config: Any,
) -> torch.Tensor:
    """Encode memory PIL frames to packed video latent tokens [B, S_mem, D]."""
    if video_vae.encoder is None:
        raise RuntimeError("video VAE encoder is not initialized for memory encoding")

    packed_latents = []
    for memory_frames in batch_memory_frames:
        if not memory_frames:
            raise ValueError("memory_frames cannot be empty when encoding memory video")
        per_slot_latents = []
        for memory_item in memory_frames:
            is_clip_memory = isinstance(memory_item, list)
            frame_video = (
                frames_to_video_tensor(
                    memory_item if is_clip_memory else [memory_item],
                    target_h,
                    target_w,
                )
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )
            encode_output = video_vae.encode(frame_video, return_dict=True)
            latent = encode_output.sample
            if latent.ndim != 5:
                raise ValueError(
                    f"Expected encoded memory latent [B,C,F,H,W], got {tuple(latent.shape)}"
                )
            # Align with official encode_memory_frames_batch layout [B, F, C, H, W].
            latent = latent.permute(0, 2, 1, 3, 4).contiguous()
            if is_clip_memory:
                latent = latent[:, -1:, :, :, :].contiguous()
            latent = latent.permute(0, 2, 1, 3, 4).contiguous()
            packed = pipeline_config.maybe_pack_latents(latent, 1, None)
            per_slot_latents.append(packed)
        packed_latents.append(torch.cat(per_slot_latents, dim=1))

    return torch.cat(packed_latents, dim=0)


# --- Memory RoPE coordinates ---


# Official ltx_wrapper hardcodes VIDEO_FPS=24.0 for RoPE position conversion.
JOYAI_VIDEO_ROPE_FPS = 24.0


def normalize_memory_position_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized == "reference":
        return "legacy"
    if normalized not in {"legacy", "prefix_continuous"}:
        raise ValueError(
            "memory_position_mode must be one of "
            "{'reference', 'legacy', 'prefix_continuous'}, "
            f"got {mode}"
        )
    return normalized


def apply_memory_video_downscale(
    video_coords: torch.Tensor,
    downscale_factor: int,
) -> torch.Tensor:
    if int(downscale_factor) == 1:
        return video_coords
    scaled = video_coords.clone()
    scaled[:, 1, ...] *= int(downscale_factor)
    scaled[:, 2, ...] *= int(downscale_factor)
    return scaled


def build_memory_video_rope_coords(
    *,
    rope,
    batch_size: int,
    memory_video_len: int,
    target_num_frames: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    fps: float,
    memory_position_mode: str,
    memory_downscale_factor: int = 1,
    sp_target_start_offset: int = 0,
) -> torch.Tensor:
    """Build [memory | target] video RoPE coordinates.

    Under sequence parallelism the target video latents are time-sharded, so
    ``target_num_frames`` is the *local* shard frame count and
    ``sp_target_start_offset`` is the global frame index of this rank's first
    target frame. The memory prefix is replicated (full) on every rank.
    """
    tokens_per_latent_frame = int(latent_height) * int(latent_width)
    if tokens_per_latent_frame <= 0:
        raise ValueError(
            f"Invalid latent grid for memory RoPE: {latent_height=} {latent_width=}"
        )
    if memory_video_len % tokens_per_latent_frame != 0:
        raise ValueError(
            "memory_video_len must be a multiple of latent_height * latent_width, "
            f"got {memory_video_len=} {latent_height=} {latent_width=}"
        )

    memory_latent_frames = memory_video_len // tokens_per_latent_frame
    position_mode = normalize_memory_position_mode(memory_position_mode)

    memory_coords = rope.prepare_video_coords(
        batch_size=batch_size,
        num_frames=memory_latent_frames,
        height=latent_height,
        width=latent_width,
        device=device,
        fps=JOYAI_VIDEO_ROPE_FPS,
        start_frame=0,
    )
    memory_coords = apply_memory_video_downscale(memory_coords, memory_downscale_factor)

    target_start_frame = (
        memory_latent_frames if position_mode == "prefix_continuous" else 0
    ) + int(sp_target_start_offset)
    target_coords = rope.prepare_video_coords(
        batch_size=batch_size,
        num_frames=target_num_frames,
        height=latent_height,
        width=latent_width,
        device=device,
        fps=JOYAI_VIDEO_ROPE_FPS,
        start_frame=target_start_frame,
    )
    return torch.cat([memory_coords, target_coords], dim=2)


def build_memory_audio_rope_coords(
    *,
    audio_rope,
    batch_size: int,
    memory_audio_len: int,
    target_audio_len: int,
    device: torch.device,
    memory_position_mode: str,
) -> torch.Tensor:
    position_mode = normalize_memory_position_mode(memory_position_mode)

    memory_coords = audio_rope.prepare_audio_coords(
        batch_size=batch_size,
        num_frames=memory_audio_len,
        device=device,
        start_frame=0,
    )
    target_start_frame = memory_audio_len if position_mode == "prefix_continuous" else 0
    target_coords = audio_rope.prepare_audio_coords(
        batch_size=batch_size,
        num_frames=target_audio_len,
        device=device,
        start_frame=target_start_frame,
    )
    return torch.cat([memory_coords, target_coords], dim=2)


# --- Paired audio-video memory bank ---


@dataclass
class MemoryEntry:
    frame: Image.Image | list[Image.Image]
    audio_latent: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def video_uint8_to_pil_frames(video_uint8: torch.Tensor) -> list[Image.Image]:
    if video_uint8.ndim != 4:
        raise ValueError(
            f"Expected [F, H, W, C] uint8 video, got shape={tuple(video_uint8.shape)}"
        )
    if video_uint8.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB video with trailing channel dim 3, got shape={tuple(video_uint8.shape)}"
        )
    video_uint8 = video_uint8.detach().cpu().contiguous()
    return [Image.fromarray(frame.numpy()) for frame in video_uint8]


def normalize_audio_waveform_for_media(
    audio_waveform: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if audio_waveform is None:
        return None

    waveform = torch.as_tensor(audio_waveform).detach().cpu().float()

    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for decoded audio, got shape={tuple(waveform.shape)}"
            )
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif (
        waveform.ndim == 2
        and waveform.shape[0] not in {1, 2}
        and waveform.shape[1]
        in {
            1,
            2,
        }
    ):
        waveform = waveform.transpose(0, 1)
    elif waveform.ndim != 2:
        raise ValueError(
            f"Expected decoded audio with 1, 2, or 3 dims, got shape={tuple(waveform.shape)}"
        )

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.contiguous()


class PairedAudioVideoMemoryBank:
    def __init__(self, max_size: int, num_fix_frames: int = 0) -> None:
        self.max_size = int(max_size)
        self.num_fix_frames = max(0, int(num_fix_frames))
        self.memory: list[MemoryEntry] = []

    @staticmethod
    def _prepare_audio_latent(
        audio_latent: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if audio_latent is None:
            return None
        if audio_latent.dim() != 3:
            raise ValueError(
                f"Expected audio_latent shape [B, T, C], got shape={tuple(audio_latent.shape)}"
            )
        return audio_latent.detach().cpu().contiguous()

    @staticmethod
    def _select_audio_window(
        audio_latent: torch.Tensor, window_size: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        total_frames = int(audio_latent.shape[1])
        window_size = max(1, int(window_size))
        window_len = min(total_frames, window_size)
        window_start = max((total_frames - window_len) // 2, 0)
        window_end = window_start + window_len
        metadata = {
            "audio_window_start": int(window_start),
            "audio_window_end": int(window_end),
            "audio_window_length": int(window_len),
            "audio_total_frames": int(total_frames),
        }
        return audio_latent[:, window_start:window_end].contiguous(), metadata

    @staticmethod
    def _waveform_to_mel(
        waveform: torch.Tensor,
        *,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> torch.Tensor:
        mono = waveform.mean(dim=0, keepdim=True)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=mel_hop_length,
            n_mels=mel_bins,
            center=True,
            power=1.0,
        )
        mel = mel_transform(mono)
        return mel.unsqueeze(0)

    @staticmethod
    def _select_video_clip_for_audio_window(
        frames: list[Image.Image],
        *,
        audio_window_start: int,
        audio_window_end: int,
        audio_total_frames: int,
        video_clip_num_frames: int,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        audio_total_frames = max(1, int(audio_total_frames))
        window_center = (float(audio_window_start) + float(audio_window_end - 1)) * 0.5
        center_ratio = window_center / float(max(audio_total_frames - 1, 1))
        center_frame = int(round(center_ratio * float(max(len(frames) - 1, 0))))
        return PairedAudioVideoMemoryBank._select_video_clip_around_frame(
            frames,
            center_frame=center_frame,
            video_clip_num_frames=video_clip_num_frames,
        )

    @staticmethod
    def _select_video_clip_around_frame(
        frames: list[Image.Image],
        *,
        center_frame: int,
        video_clip_num_frames: int,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        video_clip_num_frames = max(1, int(video_clip_num_frames))
        center_frame = max(0, min(int(center_frame), len(frames) - 1))
        left_context = (video_clip_num_frames - 1) // 2
        clip_start = max(
            0,
            min(
                center_frame - left_context, max(len(frames) - video_clip_num_frames, 0)
            ),
        )
        clip_end = min(clip_start + video_clip_num_frames, len(frames))
        clip = list(frames[clip_start:clip_end])
        if clip and len(clip) < video_clip_num_frames:
            clip.extend([clip[-1]] * (video_clip_num_frames - len(clip)))
        metadata = {
            "video_clip_start": int(clip_start),
            "video_clip_end": int(clip_end),
            "video_clip_length": int(len(clip)),
            "video_clip_center_frame": int(center_frame),
            "video_total_frames": int(len(frames)),
        }
        return clip, metadata

    def _trim(self) -> None:
        if self.max_size <= 0 or len(self.memory) <= self.max_size:
            return
        fixed = self.memory[: self.num_fix_frames]
        tail = self.memory[self.num_fix_frames :]
        keep_tail = max(0, self.max_size - len(fixed))
        self.memory = fixed + tail[-keep_tail:]

    def save_memory_slot(
        self,
        frames: list[Image.Image],
        audio_latent: torch.Tensor,
        *,
        audio_window_size: int,
        video_clip_num_frames: int,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        video_fps: float = 25.0,
        audio_window_selection_mode: str = "max_response",
        video_frame_selection_mode: str = "center",
        audio_memory_mel_bins: int = 128,
        audio_memory_mel_hop_length: int = 160,
        audio_memory_n_fft: int = 1024,
        audio_memory_downsample_factor: int = 4,
        audio_memory_is_causal: bool = True,
    ) -> dict[str, Any]:
        audio_latent = self._prepare_audio_latent(audio_latent)
        if audio_latent is None:
            raise ValueError("paired audio memory slot requires audio_latent")

        selection_mode = str(audio_window_selection_mode).lower()
        if audio_waveform is not None and selection_mode != "center":
            try:
                waveform = normalize_audio_waveform_for_media(audio_waveform)
                mel = self._waveform_to_mel(
                    waveform,
                    sample_rate=audio_sample_rate,
                    mel_bins=audio_memory_mel_bins,
                    mel_hop_length=audio_memory_mel_hop_length,
                    n_fft=audio_memory_n_fft,
                )
                pixel_window_size = latent_window_size_to_pixel_window_size(
                    int(audio_window_size),
                    downsample_factor=int(audio_memory_downsample_factor),
                    is_causal=bool(audio_memory_is_causal),
                )
                _, window_start_indices, window_end_indices = (
                    select_audio_window_with_bounds(
                        mel.float(),
                        pixel_window_size,
                        mode="max_response",
                    )
                )
                mel_start = int(window_start_indices[0].item())
                mel_end = int(window_end_indices[0].item())
                start_time_sec, end_time_sec = mel_window_bounds_to_seconds(
                    mel_start,
                    mel_end,
                    hop_length=int(audio_memory_mel_hop_length),
                    sample_rate=int(audio_sample_rate),
                )
                total_frames = int(audio_latent.shape[1])
                window_len = min(total_frames, max(1, int(audio_window_size)))
                duration_sec = max(
                    float(waveform.shape[-1]) / float(audio_sample_rate), 1e-6
                )
                center_time_sec = max(
                    0.0, min(0.5 * (start_time_sec + end_time_sec), duration_sec)
                )
                center_latent = int(
                    round(
                        center_time_sec / duration_sec * float(max(total_frames - 1, 0))
                    )
                )
                window_start = max(
                    0,
                    min(
                        center_latent - window_len // 2,
                        max(total_frames - window_len, 0),
                    ),
                )
                window_end = window_start + window_len
                window_latent = audio_latent[:, window_start:window_end].contiguous()
                audio_metadata = {
                    "audio_window_selection_mode": selection_mode,
                    "audio_window_start": int(window_start),
                    "audio_window_end": int(window_end),
                    "audio_window_length": int(window_len),
                    "audio_total_frames": int(total_frames),
                    "mel_window_start": int(mel_start),
                    "mel_window_end": int(mel_end),
                    "audio_window_start_time_sec": float(start_time_sec),
                    "audio_window_end_time_sec": float(end_time_sec),
                }
                selected_frame = select_video_frame_indices_from_time_range(
                    num_frames=len(frames),
                    fps=float(video_fps),
                    start_time_sec=float(start_time_sec),
                    end_time_sec=float(end_time_sec),
                    count=1,
                    mode=str(video_frame_selection_mode).lower(),
                )[0]
                video_clip, video_metadata = self._select_video_clip_around_frame(
                    frames,
                    center_frame=int(selected_frame),
                    video_clip_num_frames=video_clip_num_frames,
                )
            except Exception as exc:
                window_latent, audio_metadata = self._select_audio_window(
                    audio_latent, audio_window_size
                )
                audio_metadata["audio_window_selection_mode"] = "center"
                audio_metadata["selection_fallback"] = f"{selection_mode}: {exc}"
                video_clip, video_metadata = self._select_video_clip_for_audio_window(
                    frames,
                    audio_window_start=int(audio_metadata["audio_window_start"]),
                    audio_window_end=int(audio_metadata["audio_window_end"]),
                    audio_total_frames=int(audio_metadata["audio_total_frames"]),
                    video_clip_num_frames=video_clip_num_frames,
                )
        else:
            window_latent, audio_metadata = self._select_audio_window(
                audio_latent, audio_window_size
            )
            audio_metadata["audio_window_selection_mode"] = "center"
            video_clip, video_metadata = self._select_video_clip_for_audio_window(
                frames,
                audio_window_start=int(audio_metadata["audio_window_start"]),
                audio_window_end=int(audio_metadata["audio_window_end"]),
                audio_total_frames=int(audio_metadata["audio_total_frames"]),
                video_clip_num_frames=video_clip_num_frames,
            )

        metadata = {
            "selection_mode": "paired_audio_window",
            **audio_metadata,
            **video_metadata,
        }
        entry = MemoryEntry(
            frame=video_clip, audio_latent=window_latent, metadata=metadata
        )
        fixed = self.memory[: self.num_fix_frames]
        free = self.memory[self.num_fix_frames :]
        free.append(entry)
        self.memory = fixed + free
        self._trim()
        return metadata

    def get_memory_frames(self) -> list[Image.Image | list[Image.Image]]:
        return [entry.frame for entry in self.memory]

    def get_memory_audio(self) -> Optional[torch.Tensor]:
        audio_latents = [entry.audio_latent for entry in self.memory]
        if not audio_latents or any(item is None for item in audio_latents):
            return None
        first = audio_latents[0]
        assert first is not None
        for audio_latent in audio_latents:
            assert audio_latent is not None
            if (
                audio_latent.shape[0] != first.shape[0]
                or audio_latent.shape[2] != first.shape[2]
            ):
                raise ValueError(
                    "All memory audio latents must share batch and channel dimensions"
                )
        return torch.cat(audio_latents, dim=1).contiguous()

    def get_memory_audio_segment_lengths(self) -> tuple[tuple[int, ...], ...]:
        audio_latents = [entry.audio_latent for entry in self.memory]
        if not audio_latents or any(item is None for item in audio_latents):
            return ()
        return (
            tuple(
                int(audio_latent.shape[1])
                for audio_latent in audio_latents
                if audio_latent is not None
            ),
        )

    def __len__(self) -> int:
        return len(self.memory)


# --- Pipeline stages ---


class JoyEchoMemoryBankFetchStage(PipelineStage):
    """Prepare memory video/audio prefixes before DMD denoising."""

    def __init__(self, memory_bank: PairedAudioVideoMemoryBank, vae) -> None:
        super().__init__()
        self.memory_bank = memory_bank
        self.vae = vae

    @staticmethod
    def _resolve_vae_encode_dtype(vae) -> torch.dtype:
        encoder = vae.encoder
        if encoder is not None:
            try:
                return next(encoder.parameters()).dtype
            except StopIteration:
                pass
        return torch.bfloat16

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra.pop("joy_echo_memory", None)
        config = server_args.pipeline_config
        if not isinstance(config, JoyEchoPipelineConfig):
            return batch

        if not batch.enable_memory_bank or len(self.memory_bank) == 0:
            return batch

        device = get_local_torch_device()
        vae_dtype = self._resolve_vae_encode_dtype(self.vae)
        latent_dtype = batch.latents.dtype if batch.latents is not None else vae_dtype

        memory_frames = self.memory_bank.get_memory_frames()
        memory_video = encode_memory_frames_batch(
            video_vae=self.vae,
            batch_memory_frames=[memory_frames],
            target_h=int(batch.height),
            target_w=int(batch.width),
            device=device,
            dtype=vae_dtype,
            pipeline_config=config,
        )

        memory_audio = self.memory_bank.get_memory_audio()
        if memory_audio is None:
            raise RuntimeError(
                "JoyEcho memory bank has video frames but no audio latents"
            )

        memory_audio = memory_audio.to(device=device, dtype=latent_dtype)
        batch.extra["joy_echo_memory"] = {
            "memory_video_packed": memory_video,
            "memory_audio": memory_audio,
            "memory_audio_segment_lengths": self.memory_bank.get_memory_audio_segment_lengths(),
            "num_memory_slots": len(self.memory_bank),
            "paired_audio_memory": True,
            "memory_position_mode": str(config.memory_position_mode),
            "memory_downscale_factor": 1,
        }
        logger.info(
            "JoyEcho memory fetch: bank_size=%d video_tokens=%d audio_tokens=%d",
            len(self.memory_bank),
            int(memory_video.shape[1]),
            int(memory_audio.shape[1]),
        )
        return batch


class JoyEchoAVDecodingStage(LTX2AVDecodingStage):
    """Decode AV outputs and commit paired memory slots when enabled."""

    def __init__(
        self,
        vae,
        audio_vae,
        vocoder,
        memory_bank: PairedAudioVideoMemoryBank,
        pipeline=None,
    ):
        super().__init__(vae, audio_vae, vocoder, pipeline=pipeline)
        self.memory_bank = memory_bank

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        audio_latent_for_memory = None
        if batch.audio_latents is not None:
            audio_latent_for_memory = batch.audio_latents.detach().cpu()
            # Denoising unpacks audio to [B, C, L, M] before decode; memory bank
            # expects the packed [B, L, C] layout used during generation.
            if audio_latent_for_memory.dim() == 4:
                audio_latent_for_memory = (
                    server_args.pipeline_config.maybe_pack_audio_latents(
                        audio_latent_for_memory, batch.batch_size, batch
                    ).cpu()
                )

        output_batch = super().forward(batch, server_args)

        config = server_args.pipeline_config
        if not isinstance(config, JoyEchoPipelineConfig):
            return output_batch

        if not batch.enable_memory_bank or audio_latent_for_memory is None:
            return output_batch

        video_np = output_batch.output
        if video_np is None:
            return output_batch

        if isinstance(video_np, np.ndarray):
            if video_np.ndim == 5:
                video_uint8 = torch.from_numpy(video_np[0])
            elif video_np.ndim == 4:
                video_uint8 = torch.from_numpy(video_np)
            else:
                logger.warning(
                    "Unexpected decoded video shape for memory commit: %s",
                    video_np.shape,
                )
                return output_batch
        else:
            logger.warning(
                "Unsupported decoded video type for memory commit: %s", type(video_np)
            )
            return output_batch

        if video_uint8.dtype != torch.uint8:
            video_uint8 = (
                (video_uint8.clamp(0, 1) * 255).to(torch.uint8)
                if video_uint8.is_floating_point()
                else video_uint8.to(torch.uint8)
            )

        pil_frames = video_uint8_to_pil_frames(video_uint8)
        metadata = self.memory_bank.save_memory_slot(
            pil_frames,
            audio_latent_for_memory,
            audio_window_size=int(config.audio_window_size),
            video_clip_num_frames=int(config.memory_video_clip_num_frames),
            audio_waveform=output_batch.audio,
            audio_sample_rate=int(
                output_batch.audio_sample_rate
                or config.audio_vae_config.arch_config.sample_rate
            ),
            video_fps=float(batch.fps),
            audio_window_selection_mode=str(config.audio_window_selection_mode),
            video_frame_selection_mode=str(config.video_memory_frame_selection_mode),
            audio_memory_mel_bins=int(config.audio_mel_bins),
            audio_memory_mel_hop_length=int(config.audio_mel_hop_length),
            audio_memory_n_fft=int(config.audio_n_fft),
            audio_memory_downsample_factor=int(config.audio_downsample_factor),
            audio_memory_is_causal=True,
        )
        logger.info(
            "JoyEcho memory commit: bank_size=%d metadata=%s",
            len(self.memory_bank),
            metadata,
        )
        return output_batch
