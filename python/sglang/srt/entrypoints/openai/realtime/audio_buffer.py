"""Rolling PCM buffer and audio helpers for realtime ASR.

Owns the current input buffer's PCM byte timeline. Resident bytes may be
compacted after inference, but offsets remain absolute until commit or clear, so
inference policy can keep valid cursors after bytes are dropped.
"""

import msgspec
import numpy as np

# Realtime input is validated as PCM16; keep all byte offsets sample-aligned.
PCM_SAMPLE_WIDTH_BYTES = 2


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


class AudioBuffer(msgspec.Struct):
    """Rolling PCM16 store addressed by absolute byte offsets.

    Offsets count every byte the item ever received (0 = its first byte), so
    cursors stay valid after old audio is dropped. ``data`` holds the resident
    range ``[base_offset_bytes, received_bytes)``.
    """

    data: bytearray = bytearray()
    # Resident bytes may start after offset zero once compaction drops them.
    base_offset_bytes: int = 0
    # A deferred decode advances only `last_attempted`, so pacing waits for new
    # input while `last_processed` keeps that audio in the next request.
    last_attempted_offset_bytes: int = 0
    last_processed_offset_bytes: int = 0

    @property
    def received_bytes(self) -> int:
        return self.base_offset_bytes + len(self.data)

    def append_pcm(self, pcm: bytes) -> None:
        self.data.extend(pcm)

    def snapshot(self, start_offset_bytes: int, end_offset_bytes: int) -> bytes:
        """Copy the audio in absolute range [start, end); the copy stays valid
        after discard_before() drops the underlying bytes."""
        start = start_offset_bytes - self.base_offset_bytes
        end = end_offset_bytes - self.base_offset_bytes
        if not (0 <= start <= end <= len(self.data)):
            raise ValueError(
                "audio range "
                f"[{start_offset_bytes}, {end_offset_bytes}) is outside resident "
                f"[{self.base_offset_bytes}, {self.received_bytes})"
            )
        return bytes(memoryview(self.data)[start:end])

    def discard_before(self, offset_bytes: int) -> None:
        """Free memory by dropping all audio before an absolute offset."""
        if offset_bytes < self.base_offset_bytes or offset_bytes > self.received_bytes:
            raise ValueError(
                f"discard offset {offset_bytes} is outside resident range "
                f"[{self.base_offset_bytes}, {self.received_bytes}]"
            )
        drop_bytes = offset_bytes - self.base_offset_bytes
        drop_bytes -= drop_bytes % PCM_SAMPLE_WIDTH_BYTES
        if drop_bytes <= 0:
            return

        del self.data[:drop_bytes]
        self.base_offset_bytes += drop_bytes
