"""Frame sampling and audio interleaving helpers for dots video requests."""

import base64
import io
import math
import random
import re
import wave

import numpy as np

_VIDEO_KEY_RE = re.compile(r"video_(\d+)")
_VIDEO_MARKER_RE = re.compile(r"<video_(\d+)>")


def _format_timestamp(sec: float, fmt: str = "hms", seconds_decimals: int = 1) -> str:
    """Format a non-negative timestamp in the model's training format."""
    sec = max(0.0, sec)
    if fmt == "seconds":
        return f"{sec:.{seconds_decimals}f} seconds"
    total_cs = round(sec * 100)
    hours = total_cs // (3600 * 100)
    minutes = (total_cs // (60 * 100)) % 60
    seconds = (total_cs // 100) % 60
    centiseconds = total_cs % 100
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _sorted_image_keys(video_dict: dict) -> list[str]:
    """Return image keys ordered by their numeric suffix."""
    pairs = []
    for key in video_dict:
        match = re.fullmatch(r"image_(\d+)", key)
        if match:
            pairs.append((int(match.group(1)), key))
    return [key for _, key in sorted(pairs)]


def _uniform_pick_indices(n_total: int, n_keep: int) -> list[int]:
    """Pick evenly spaced frame indices."""
    if n_keep >= n_total:
        return list(range(n_total))
    if n_keep <= 1:
        return [n_total // 2]
    step = (n_total - 1) / (n_keep - 1)
    return sorted({round(i * step) for i in range(n_keep)})[:n_keep]


class VideoQAFlattener:
    """Apply the frame and audio sampling policy used during training."""

    def __init__(
        self,
        fps_scale_range: tuple[float, float] = (1.0, 1.0),
        min_frames: int = 1,
        max_frames: int | None = None,
        time_format: str = "random",
        seconds_decimals: int = 1,
        audio_interleave: bool = False,
        ai_seg_min_sec: float = 1.0,
        ai_k_mode: str = "eval30",
        ai_sample_rate: int = 16000,
    ):
        lo, hi = map(float, fps_scale_range)
        if not 0 < lo <= hi <= 1.0:
            raise ValueError(
                "fps_scale_range must satisfy 0 < lo <= hi <= 1.0, "
                f"got {fps_scale_range}"
            )
        if time_format not in ("hms", "seconds", "random"):
            raise ValueError(f"unsupported time format: {time_format!r}")
        if ai_k_mode not in ("logk", "eval30", "eval_ek", "whole"):
            raise ValueError(f"unsupported audio interleave mode: {ai_k_mode!r}")

        self.fps_scale_lo = lo
        self.fps_scale_hi = hi
        self.min_frames = max(1, min_frames)
        self.max_frames = max_frames
        self.time_format = time_format
        self.seconds_decimals = max(0, int(seconds_decimals))
        self.audio_interleave = bool(audio_interleave)
        self.ai_seg_min_sec = max(1e-6, float(ai_seg_min_sec))
        self.ai_k_mode = ai_k_mode
        self.ai_sample_rate = int(ai_sample_rate)

    def _sample_n_keep(self, n_total: int) -> int:
        if n_total <= 0:
            return 0
        scale = (
            self.fps_scale_lo
            if self.fps_scale_lo == self.fps_scale_hi
            else random.uniform(self.fps_scale_lo, self.fps_scale_hi)
        )
        n_keep = max(self.min_frames, round(n_total * scale))
        if self.max_frames is not None and self.max_frames > 0:
            n_keep = min(n_keep, self.max_frames)
        return min(n_keep, n_total)

    def _subsample_one_video(self, video_dict: dict):
        """Return sampled frames, timestamps, and optional WAV data."""
        original_fps = float(video_dict.get("fps", 1.0)) or 1.0
        image_keys = _sorted_image_keys(video_dict)
        indices = _uniform_pick_indices(
            len(image_keys), self._sample_n_keep(len(image_keys))
        )
        frames = [video_dict[image_keys[i]] for i in indices]
        timestamps = [round(i / original_fps, 3) for i in indices]
        return frames, timestamps, video_dict.get("audio_0") or None

    @staticmethod
    def _decode_wav_b64(audio_b64: str):
        """Decode a base64 WAV into mono int16 PCM."""
        with wave.open(io.BytesIO(base64.b64decode(audio_b64)), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            channels = wav.getnchannels()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return pcm, sample_rate, sample_width, channels

    @staticmethod
    def _encode_wav_b64(pcm, sample_rate: int):
        """Encode mono int16 PCM as a base64 WAV."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _decide_group_bounds(self, n_frames: int, duration: float):
        """Split frames into the configured number of audio windows."""
        if n_frames <= 1 or duration <= 0:
            return [0, n_frames] if n_frames else [0, 0]
        k_max = min(n_frames, max(1, int(duration // self.ai_seg_min_sec)))
        if self.ai_k_mode == "whole" or k_max <= 1:
            groups = 1
        elif self.ai_k_mode == "eval30":
            groups = round(math.sqrt(k_max))
        elif self.ai_k_mode == "eval_ek":
            groups = round((k_max - 1) / math.log(k_max))
        else:
            groups = round(math.exp(random.uniform(0.0, math.log(k_max))))
        groups = max(1, min(k_max, groups))
        if groups == 1:
            return [0, n_frames]
        if self.ai_k_mode == "logk":
            cuts = sorted(random.sample(range(1, n_frames), groups - 1))
        else:
            cuts = sorted(
                {
                    cut
                    for i in range(1, groups)
                    if 0 < (cut := round(i * n_frames / groups)) < n_frames
                }
            )
        return [0, *cuts, n_frames]
