"""Train-compatible v2 preprocessing for dots video serving."""

import base64
import io
import math
import re
import wave

import numpy as np
from PIL import Image
from torchcodec.decoders import VideoDecoder

from .v2core import (
    ALIGN,
    INTERLEAVE_SEG_MIN_SEC,
    V2_FPS_CAP,
    V2_FPS_MIN,
    V2_OVH,
    V2_PF_CEIL,
    V2_PF_FLOOR,
    compute_target_size,
    v2_solve_degrade,
    v2_split_visual_budget,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_SAMPLES_PER_TOKEN = 1280
DEFAULT_AUDIO_CHUNK_SEC = 30

TOK_SYS_START = "<|system|>"
TOK_SYS_END = "<|endofsystem|>"
AUDIO_WRAP_TOKENS = 2
ROLE_WRAP_TOKENS = 2
_OVERHEAD_MARGIN = 64


class SkipSample(Exception):
    """Raised when a video cannot fit or cannot be decoded."""


def tokenize_len(text, tokenizer):
    """Count tokens without adding tokenizer special tokens."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def system_block_tokens(prompt=DEFAULT_SYSTEM_PROMPT, tokenizer=None):
    if tokenizer is None:
        raise ValueError("dots video preprocessing requires a tokenizer")
    return tokenize_len(f"{TOK_SYS_START}{prompt}{TOK_SYS_END}\n", tokenizer=tokenizer)


def audio_block_tokens(
    duration_sec,
    samples_per_token=DEFAULT_AUDIO_SAMPLES_PER_TOKEN,
    chunk_sec=DEFAULT_AUDIO_CHUNK_SEC,
    sr=DEFAULT_AUDIO_SAMPLE_RATE,
):
    """Estimate tokens for one independently encoded audio segment."""
    if duration_sec is None or duration_sec <= 0:
        return 0
    total_samples = int(duration_sec * sr)
    chunk_samples = chunk_sec * sr
    padding_tokens = 0
    for position in range(0, total_samples, chunk_samples):
        length = min(chunk_samples, total_samples - position)
        padding_tokens += math.ceil(length / samples_per_token)
    return AUDIO_WRAP_TOKENS + padding_tokens


def conversation_tokens(conversations, tokenizer=None):
    """Count conversation text while preserving video-marker boundaries."""
    if tokenizer is None:
        raise ValueError("dots video preprocessing requires a tokenizer")
    total = 0
    for conversation in conversations:
        total += ROLE_WRAP_TOKENS
        parts = re.split(r"<video_(\d+)>", conversation.get("value", "") or "")
        for index, part in enumerate(parts):
            if index % 2 == 0:
                total += tokenize_len(part, tokenizer=tokenizer)
            else:
                video_index = int(part)
                total += tokenize_len(f"<video_{video_index}>", tokenizer=tokenizer)
    return total


def _make_video_decoder(video_bytes):
    try:
        return VideoDecoder(
            video_bytes,
            dimension_order="NHWC",
            num_ffmpeg_threads=1,
            seek_mode="approximate",
        )
    except TypeError:
        return VideoDecoder(video_bytes, dimension_order="NHWC", num_ffmpeg_threads=1)


def extract_frames_v2(
    decoder,
    seq_length,
    visual_budget,
    *,
    pf_floor=V2_PF_FLOOR,
    pf_ceil=V2_PF_CEIL,
    fps_cap=V2_FPS_CAP,
    fps_min=V2_FPS_MIN,
    overhead=V2_OVH,
    jpeg_quality=85,
):
    """Decode, resize, and JPEG-encode frames selected by the v2 policy."""
    metadata = decoder.metadata
    duration = float(metadata.duration_seconds or 0)
    original_height = int(metadata.height)
    original_width = int(metadata.width)
    total_frames = int(metadata.num_frames or 0)
    if duration <= 0 or original_height <= 0 or original_width <= 0:
        raise ValueError(
            f"bad metadata: dur={duration} h={original_height} w={original_width}"
        )
    original_fps = float(metadata.average_fps or 0) or 25.0
    if total_frames <= 0:
        total_frames = max(1, int(duration * original_fps))

    aligned_height = max(ALIGN, round(original_height / ALIGN) * ALIGN)
    aligned_width = max(ALIGN, round(original_width / ALIGN) * ALIGN)
    original_patches = (aligned_height // ALIGN) * (aligned_width // ALIGN)
    num_frames, _, target_patches = v2_solve_degrade(
        visual_budget,
        duration,
        original_patches,
        original_fps,
        seq_length,
        fps_cap=fps_cap,
        fps_min=fps_min,
        pf_floor=pf_floor,
        pf_ceil=pf_ceil,
        ovh=overhead,
        orig_h=original_height,
        orig_w=original_width,
    )
    max_pixels = min(target_patches, original_patches) * ALIGN * ALIGN
    target_height, target_width = compute_target_size(
        original_height,
        original_width,
        pf_floor * ALIGN * ALIGN,
        max_pixels,
    )

    num_frames = max(4, min(num_frames, total_frames))
    if num_frames == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (num_frames - 1)
        indices = sorted(
            {
                max(0, min(round(index * step), total_frames - 1))
                for index in range(num_frames)
            }
        )
    try:
        frames = decoder.get_frames_at(indices=indices).data
    except (IndexError, RuntimeError):
        safe_indices = list(indices)
        while safe_indices and safe_indices[-1] > 0:
            safe_indices.pop()
            try:
                frames = decoder.get_frames_at(indices=safe_indices).data
                indices = safe_indices
                break
            except (IndexError, RuntimeError):
                continue
        else:
            raise

    timestamps = [round(index / original_fps, 3) for index in indices[: len(frames)]]
    encoded_frames = []
    for timestamp, frame in zip(timestamps, frames):
        array = frame.numpy()
        image = Image.fromarray(array)
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        encoded_frames.append(
            (timestamp, base64.b64encode(buffer.getvalue()).decode("ascii"))
        )

    actual_fps = round(len(encoded_frames) / duration, 4)
    return encoded_frames, duration, target_height, target_width, actual_fps


def extract_audio_wav(
    video_bytes, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE, max_duration_sec=None
):
    """Extract mono 16-bit WAV audio; return ``(None, 0.0)`` when unavailable."""
    from torchcodec.decoders import AudioDecoder

    try:
        samples = AudioDecoder(
            video_bytes, sample_rate=sample_rate, num_channels=1
        ).get_all_samples()
    except Exception:  # noqa: BLE001 - videos without decodable audio are valid
        return None, 0.0
    waveform = samples.data
    sample_rate = int(samples.sample_rate)
    if waveform is None or waveform.numel() == 0:
        return None, 0.0
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if max_duration_sec and max_duration_sec > 0:
        waveform = waveform[: int(max_duration_sec * sample_rate)]
    pcm = (np.clip(waveform.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    duration = len(pcm) / sample_rate if sample_rate else 0.0
    return base64.b64encode(buffer.getvalue()).decode("ascii"), duration


def _find_video_keys(conversations):
    text = " ".join(conversation.get("value", "") for conversation in conversations)
    return sorted({int(match) for match in re.findall(r"<video_(\d+)>", text)})


def _normalize_sample(sample):
    meta = dict(sample.get("meta") or {})
    conversations = sample.get("conversations") or []
    return meta, [dict(conversation) for conversation in conversations]


def process_sample_video(sample, cfg, *, tokenizer):
    """Convert an in-memory video sample into nested frame/audio metadata."""
    meta, conversations = _normalize_sample(sample)
    if not conversations:
        raise SkipSample("empty_conversations")
    video_keys = _find_video_keys(conversations)
    if not video_keys:
        raise SkipSample("no_video_marker_in_conversations")
    seq_length = cfg["seq_length"]
    if seq_length <= 0:
        raise SkipSample("invalid_seq_length")

    video_data = {}
    audio_data = {}
    audio_tokens = {}
    durations = {}
    for video_index in video_keys:
        encoded = meta.get(f"video_{video_index}")
        if not encoded:
            raise SkipSample(f"no_video_data:video_{video_index}")
        try:
            video_bytes = base64.b64decode(encoded)
            decoder = _make_video_decoder(video_bytes)
            duration = float(decoder.metadata.duration_seconds or 0)
        except Exception as exc:
            raise SkipSample(
                f"bad_video_meta:video_{video_index}:{type(exc).__name__}:{exc}"
            ) from exc
        if duration <= 0:
            raise SkipSample(f"bad_video_duration:video_{video_index}:{duration}")
        video_data[video_index] = (video_bytes, decoder)
        durations[video_index] = duration
        audio_tokens[video_index] = 0

        if cfg["process_audio"]:
            audio_b64, audio_duration = extract_audio_wav(
                video_bytes, sample_rate=cfg["audio_sample_rate"]
            )
            if audio_b64 and audio_duration > 0:
                audio_data[video_index] = (audio_b64, audio_duration)
                tokens = audio_block_tokens(audio_duration, sr=cfg["audio_sample_rate"])
                if cfg["reserve_interleave"]:
                    frame_upper_bound = max(1, int(duration * V2_FPS_CAP))
                    max_groups = min(
                        frame_upper_bound,
                        max(1, int(audio_duration // INTERLEAVE_SEG_MIN_SEC)),
                    )
                    tokens += 3 * max_groups
                audio_tokens[video_index] = tokens

    fixed_tokens = (
        system_block_tokens(tokenizer=tokenizer)
        + conversation_tokens(conversations, tokenizer=tokenizer)
        + _OVERHEAD_MARGIN
    )
    total_audio_tokens = sum(audio_tokens.values())
    pure_audio_tokens = sum(
        audio_block_tokens(duration, sr=cfg["audio_sample_rate"])
        for _, duration in audio_data.values()
    )
    audio_cap = cfg["audio_token_ratio_cap"]
    if audio_cap > 0 and pure_audio_tokens > audio_cap * seq_length:
        raise SkipSample("audio_token_ratio_exceed")
    minimum_visual_tokens = len(video_keys) * 4 * (V2_PF_FLOOR + V2_OVH)
    if fixed_tokens + total_audio_tokens + minimum_visual_tokens > seq_length:
        reason = "audio_token_ratio_exceed" if audio_data else "input_budget_exhausted"
        raise SkipSample(reason)

    visual_total = seq_length - fixed_tokens - total_audio_tokens
    budgets = v2_split_visual_budget(
        [durations[index] for index in video_keys], visual_total
    )

    new_meta = {}
    for video_index, visual_budget in zip(video_keys, budgets):
        _, decoder = video_data[video_index]
        try:
            frames, _, _, _, actual_fps = extract_frames_v2(
                decoder,
                seq_length,
                visual_budget,
                jpeg_quality=cfg["video_jpeg_quality"],
            )
        except Exception as exc:
            raise SkipSample(
                f"video_decode_fail:video_{video_index}:{type(exc).__name__}:{exc}"
            ) from exc
        nested = {"fps": actual_fps}
        nested.update(
            {f"image_{index}": encoded for index, (_, encoded) in enumerate(frames)}
        )
        if video_index in audio_data:
            audio_b64, audio_duration = audio_data[video_index]
            nested.update(
                audio_0=audio_b64,
                audio_duration=audio_duration,
                audio_sample_rate=cfg["audio_sample_rate"],
            )
        new_meta[f"video_{video_index}"] = nested
    return new_meta, conversations
