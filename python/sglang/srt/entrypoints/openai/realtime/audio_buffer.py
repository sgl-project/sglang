"""Rolling PCM buffer and audio helpers for realtime ASR.

Owns the item-local PCM byte timeline. The resident buffer is compacted as old
audio is consumed, but every offset stays absolute within the item, so the
scheduling/commit cursors in session.py remain valid after bytes are dropped.
Input is PCM16 throughout, so byte offsets are kept sample-aligned.
"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from sglang.srt.entrypoints.openai.streaming_asr import StreamingASRState

# Realtime input is validated as PCM16; keep all byte offsets sample-aligned.
PCM_SAMPLE_WIDTH = 2

# Only skip windows that are effectively digital silence. This avoids empty
# audio-feature requests without treating low-volume speech as silence.
SLICED_SILENCE_RMS_THRESHOLD = 0.005


def slice_pcm_range(buffer: Union[bytes, bytearray], start: int, end: int) -> bytes:
    """Snapshot mutable rolling PCM before append/compaction can move it."""
    if not (0 <= start <= end <= len(buffer)):
        raise ValueError(
            f"slice_pcm_range: range=[{start}, {end}) not in [0, {len(buffer)}]"
        )
    return bytes(memoryview(buffer)[start:end])


def resample_to_target_rate(pcm: bytes, src_rate: int, target_rate: int) -> bytes:
    if src_rate == target_rate or not pcm:
        return pcm
    import torch
    import torchaudio

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples).unsqueeze(0)
    audio = torchaudio.functional.resample(
        audio, orig_freq=src_rate, new_freq=target_rate
    )
    samples = audio.squeeze(0).numpy()
    # Clip to int16 range via 2^15 - 1 so a clipped 1.0 stays representable.
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def pcm_to_float_samples(pcm: bytes) -> np.ndarray:
    # /32768.0 matches soundfile.read's default int16 normalization so the
    # samples are bit-equal to the prior PCM->WAV->sf.read path.
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def is_near_silent_pcm(pcm: bytes) -> bool:
    if not pcm:
        return True
    samples = np.frombuffer(pcm, dtype=np.int16)
    float_samples = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(float_samples * float_samples)))
    return rms < SLICED_SILENCE_RMS_THRESHOLD


@dataclass
class AudioState:
    """Rolling PCM buffer with item-absolute cursors."""

    max_buffer_bytes: int
    chunk_size_bytes: int
    left_overlap_bytes: int
    slicing_min_chunk_index: int
    state: StreamingASRState
    slicing_enabled: bool
    pcm_buffer: bytearray = field(default_factory=bytearray)
    # Absolute byte timeline for the current item. The resident buffer may be
    # compacted, but scheduling/commit decisions still use absolute offsets.
    pcm_buffer_base_offset_bytes: int = 0
    total_pcm_bytes_received: int = 0
    # Scheduling can advance on a deferred/empty slice; inferred advances only
    # after accepted model output, so commit still covers skipped audio and the
    # next sliced window re-anchors on it (frozen across deferred slices).
    last_scheduled_offset_bytes: int = 0
    last_inferred_offset_bytes: int = 0

    def append_pcm(self, pcm: bytes) -> None:
        self.pcm_buffer.extend(pcm)
        self.total_pcm_bytes_received += len(pcm)

    def compact_after_sliced_inference(self) -> None:
        # Keep one chunk beyond the overlap: a commit-time too-short/near-silent
        # window widens to the whole resident buffer and needs voiced context.
        keep_bytes = self.left_overlap_bytes + self.chunk_size_bytes
        keep_start = max(0, self.last_inferred_offset_bytes - keep_bytes)
        drop_bytes = keep_start - self.pcm_buffer_base_offset_bytes
        drop_bytes -= drop_bytes % PCM_SAMPLE_WIDTH
        if drop_bytes <= 0:
            return

        del self.pcm_buffer[:drop_bytes]
        self.pcm_buffer_base_offset_bytes += drop_bytes

    def reset_pcm_offsets(self) -> None:
        self.pcm_buffer.clear()
        self.pcm_buffer_base_offset_bytes = 0
        self.total_pcm_bytes_received = 0
        self.last_scheduled_offset_bytes = 0
        self.last_inferred_offset_bytes = 0

    def global_to_local(self, global_offset: int) -> int:
        return global_offset - self.pcm_buffer_base_offset_bytes
