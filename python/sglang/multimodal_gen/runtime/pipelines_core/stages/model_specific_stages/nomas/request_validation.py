# SPDX-License-Identifier: Apache-2.0
"""Request-side Nomas contracts; no dependency on the upstream hymm package."""

import json
import math

from msgspec import Struct

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class NomasSegment(Struct, frozen=True):
    start: int
    end: int
    caption: str


def plan_segments(num_frames, prompt, segments, continuations, segment_size):
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    latent_frames = (num_frames - 1) // 4 + 1
    if segments is None:
        starts = list(range(0, latent_frames, segment_size))
        captions = [prompt] * len(starts)
        if continuations is not None:
            if len(continuations) != len(starts) - 1:
                raise ValueError(
                    "continuation_prompts must contain one caption per segment after the first"
                )
            captions[1:] = continuations
        boundaries = starts + [latent_frames]
    else:
        if continuations is not None or not segments:
            raise ValueError(
                "segments must be nonempty and cannot be combined with continuation_prompts"
            )
        pixel_end = 0
        boundaries, captions = [0], []
        for segment in segments:
            if not isinstance(segment, dict) or set(segment) != {"frames", "caption"}:
                raise ValueError(
                    "each segment must contain exactly frames and caption; camera uses the global norm_extrinsics timeline"
                )
            frames = segment["frames"]
            if type(frames) is not int or frames < 1:
                raise ValueError("segment frames must be a positive integer")
            pixel_end += frames
            # Match causal_ar's pixel-boundary conversion, including min-one-frame.
            boundaries.append(
                max(boundaries[-1] + 1, min(pixel_end // 4, latent_frames))
            )
            captions.append(segment["caption"])
        if pixel_end != num_frames:
            raise ValueError(
                f"sum of segment frames ({pixel_end}) must equal num_frames ({num_frames})"
            )
        boundaries[-1] = latent_frames
    if not all(isinstance(caption, str) and caption.strip() for caption in captions):
        raise ValueError("each Nomas caption must be a nonempty string")
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError(
            "event boundaries produce empty latent segments; combine short events"
        )
    return [
        NomasSegment(a, b, c) for a, b, c in zip(boundaries, boundaries[1:], captions)
    ]


def canonical_caption(caption, prompt_format):
    if prompt_format == "plain":
        return caption
    if prompt_format != "json":
        raise ValueError("prompt_format must be json or plain")
    try:
        obj = json.loads(caption)
    except json.JSONDecodeError as exc:
        raise ValueError("json mode requires an upstream segment JSON caption") from exc
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("segments"), list)
        or len(obj["segments"]) != 1
        or not isinstance(obj["segments"][0], dict)
    ):
        raise ValueError("each caption must contain exactly one object in segments")
    event = obj["segments"][0]
    actions = event.pop("contained_actions", None)
    if actions is not None:
        if not isinstance(actions, (list, dict)):
            raise ValueError("contained_actions must be a list or ordered object")
        event["contained_actions"] = (
            actions  # Must be last, as in upstream prompt builder.
        )
    return json.dumps(obj, ensure_ascii=False)


def _seconds(value):
    if value is None:
        return 0.0
    parts = str(value).split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("action timestamp must be seconds, MM:SS or HH:MM:SS")
    seconds = sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))
    if not math.isfinite(seconds):
        raise ValueError("action timestamps must be finite")
    return seconds


def action_metadata(caption, config, tokenizer, text_mask, frames, fps):
    """Common-prefix offsets match upstream; tail end uses the *cropped* mask."""
    obj = json.loads(caption)
    event = obj["segments"][0]
    actions = event.get("contained_actions", [])
    values = list(actions.values()) if isinstance(actions, dict) else actions
    if not values:
        return -1, None, None
    length = text_mask.shape[1]
    full = config.tokenize_prompt([caption], tokenizer, {"max_length": length})
    crop = int(full["nomas_crop_start"])
    ids = full["input_ids"][0].tolist()
    valid = int(text_mask.sum())
    offsets = []
    for k in range(len(values)):
        variant = dict(event)
        if k == 0:
            variant.pop("contained_actions")
        else:
            variant["contained_actions"] = (
                dict(list(actions.items())[:k])
                if isinstance(actions, dict)
                else actions[:k]
            )
        text = json.dumps(obj | {"segments": [variant]}, ensure_ascii=False)
        other = config.tokenize_prompt([text], tokenizer, {"max_length": length})[
            "input_ids"
        ][0].tolist()
        common = next(
            (i for i, (a, b) in enumerate(zip(ids, other)) if a != b),
            min(len(ids), len(other)),
        )
        offset = max(0, min(common - crop, length - 1, valid))
        offsets.append(max(offset, offsets[-1] if offsets else 0))
    offsets.append(valid)
    windows = action_video_windows(values, frames, fps)
    if windows is None:
        return offsets[0], None, None
    return offsets[0], tuple(offsets), windows


def action_video_windows(actions, frames, fps):
    """Return None, meaning full intra-segment attention, where upstream does.

    Upstream degrades instead of failing: missing or malformed timestamps and
    overlapping actions disable the intra-segment mask for that segment only.
    """
    spans = []
    for index, action in enumerate(actions):
        if (
            not isinstance(action, dict)
            or not {"start_time", "end_time"} <= action.keys()
        ):
            logger.warning(
                "Nomas action %d has no start_time/end_time; intra-segment action mask disabled",
                index,
            )
            return None
        start, end = (_seconds(action[key]) for key in ("start_time", "end_time"))
        a, b = (min(frames, int(max(t, 0.0) * fps) // 4) for t in (start, end))
        if b > a:
            spans.append((a, b, index))
    if not spans:
        return None
    windows, cursor = [], 0
    for a, b, index in sorted(spans):
        if a < cursor:
            logger.warning(
                "Nomas action video windows overlap; intra-segment action mask disabled"
            )
            return None
        if a > cursor:
            windows.append((cursor, a, -1))
        windows.append((a, b, index))
        cursor = b
    if cursor < frames:
        windows.append((cursor, frames, -1))
    if len(windows) == 1 and windows[0][2] == 0 and len(actions) == 1:
        return None
    return tuple(windows)


def cache_bytes_per_token(config, head_shards=1):
    """Per-rank K/V bytes for one cached token (bf16); shards divide the heads."""
    return (
        2
        * config.dit_config.num_layers
        * config.dit_config.hidden_size
        * 2
        // head_shards
    )


def cache_budget_bytes(config, free_bytes=None):
    """KV budget: max_cache_gib, capped by free memory minus the reserve."""
    limit = config.max_cache_gib * 2**30
    if free_bytes is not None:
        limit = min(
            limit, max(0, free_bytes - config.cache_free_memory_reserve_gib * 2**30)
        )
    return limit


def check_cache_budget(config, capacity, *, free_bytes=None, head_shards=1):
    """Bound persistent KV before allocation; this is not a peak-memory predictor."""
    required = capacity * cache_bytes_per_token(config, head_shards)
    limit = cache_budget_bytes(config, free_bytes)
    if required > limit:
        raise ValueError(
            f"Nomas KV requires {required / 2**30:.2f} GiB, budget {limit / 2**30:.2f} GiB; "
            "reduce resolution, event count or text length."
        )
    return required
