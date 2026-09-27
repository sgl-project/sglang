"""Build and render the train-compatible dots video interleave plan."""

import hashlib
import random

import numpy as np

from .video_qa_flattener import (
    _VIDEO_KEY_RE,
    _VIDEO_MARKER_RE,
    VideoQAFlattener,
    _format_timestamp,
)


def _derive_seed(record_key: str) -> int:
    digest = hashlib.sha1(f"42|flatten|{record_key}".encode()).hexdigest()
    return int(digest[:8], 16)


def _normalize_conversations(conversations):
    if conversations is None:
        return []
    if isinstance(conversations, np.ndarray):
        conversations = conversations.tolist()
    normalized = []
    for conversation in conversations:
        if isinstance(conversation, np.ndarray):
            conversation = conversation.tolist()
        normalized.append(dict(conversation))
    return normalized


def _plan_interleaved_emissions(flattener, frames, timestamps, audio_b64, video_dict):
    """Plan frame/audio emissions without coupling the policy to an output schema."""
    duration = float(video_dict.get("audio_duration", 0) or 0)
    if duration <= 0:
        duration = float(timestamps[-1]) if timestamps else float(len(frames))

    bounds = flattener._decide_group_bounds(len(frames), duration)
    try:
        pcm, sample_rate = flattener._decode_wav_b64(audio_b64)
    except Exception:  # noqa: BLE001 - malformed audio falls back to one block
        emissions = [
            {"kind": "frame", "ts": timestamp, "b64": frame}
            for frame, timestamp in zip(frames, timestamps)
        ]
        emissions.append({"kind": "audio", "b64": audio_b64, "dur": duration})
        return emissions

    emissions = []
    for group in range(len(bounds) - 1):
        frame_start, frame_end = bounds[group : group + 2]
        if frame_end <= frame_start:
            continue
        time_start = 0.0 if group == 0 else float(timestamps[frame_start])
        time_end = (
            duration
            if group == len(bounds) - 2
            else float(timestamps[bounds[group + 1]])
        )
        if time_end <= time_start:
            time_end = time_start + duration / max(len(bounds) - 1, 1)
        emissions.extend(
            {
                "kind": "frame",
                "ts": timestamps[index],
                "b64": frames[index],
            }
            for index in range(frame_start, frame_end)
        )
        sample_start = max(0, round(time_start * sample_rate))
        sample_end = min(len(pcm), round(time_end * sample_rate))
        if sample_end > sample_start:
            segment = pcm[sample_start:sample_end]
            emissions.append(
                {
                    "kind": "audio",
                    "b64": flattener._encode_wav_b64(segment, sample_rate),
                    "dur": len(segment) / sample_rate,
                }
            )
    return emissions


def build_plan(
    meta,
    conversations,
    record_key: str,
    *,
    k_mode: str,
    process_audio: bool,
):
    """Create a deterministic intermediate plan for one request."""
    rng = random.Random(_derive_seed(record_key))
    flattener = VideoQAFlattener(
        time_format="hms",
        audio_interleave=process_audio,
        ai_k_mode=k_mode,
        rng=rng,
    )
    old_meta = dict(meta) if meta else {}
    video_pairs = []
    for key, video in old_meta.items():
        match = _VIDEO_KEY_RE.fullmatch(key) if isinstance(key, str) else None
        if match and isinstance(video, dict):
            video_pairs.append((int(match.group(1)), video))
    video_pairs.sort()
    passthrough = {
        key: value
        for key, value in old_meta.items()
        if not (isinstance(key, str) and _VIDEO_KEY_RE.fullmatch(key))
    }

    videos = []
    for video_index, video in video_pairs:
        frames, timestamps, audio_b64 = flattener._subsample_one_video(video)
        if flattener.audio_interleave and audio_b64 is not None and frames:
            emissions = _plan_interleaved_emissions(
                flattener, frames, timestamps, audio_b64, video
            )
        else:
            emissions = [
                {"kind": "frame", "ts": timestamp, "b64": frame}
                for frame, timestamp in zip(frames, timestamps)
            ]
            if audio_b64 is not None:
                emissions.append({"kind": "audio", "b64": audio_b64})
        videos.append(
            {
                "index": video_index,
                "time_format": flattener.time_format,
                "emissions": emissions,
            }
        )

    return {
        "seconds_decimals": flattener.seconds_decimals,
        "videos": videos,
        "passthrough_meta": passthrough,
        "conversations": _normalize_conversations(conversations),
    }


def render_flat(plan):
    """Render a plan as globally numbered image/audio markers."""
    new_meta = dict(plan["passthrough_meta"])
    next_image = 0
    next_audio = 0
    replacements = {}
    decimals = plan["seconds_decimals"]
    for video in plan["videos"]:
        parts = []
        for emission in video["emissions"]:
            if emission["kind"] == "frame":
                key = f"image_{next_image}"
                new_meta[key] = emission["b64"]
                timestamp = _format_timestamp(
                    emission["ts"],
                    fmt=video["time_format"],
                    seconds_decimals=decimals,
                )
                parts.append(f"<{timestamp}><{key}>")
                next_image += 1
            else:
                key = f"audio_{next_audio}"
                new_meta[key] = emission["b64"]
                parts.append(f"<{key}>")
                next_audio += 1
        replacements[video["index"]] = "".join(parts)

    conversations = []
    for conversation in plan["conversations"]:
        updated = dict(conversation)
        value = conversation.get("value", "") or ""
        updated["value"] = _VIDEO_MARKER_RE.sub(
            lambda match: replacements.get(int(match.group(1)), ""), value
        )
        conversations.append(updated)
    return {"meta": new_meta, "conversations": conversations, "data_type": "mm"}
