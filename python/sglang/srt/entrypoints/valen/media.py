# SPDX-License-Identifier: Apache-2.0
"""Bridge Valen's HF processor to SGLang's processor_output media input.

Expand the entire state before appending the independently tokenized schema.
Passing the resulting pixel tensors and grids avoids a second decode, resize,
frame sampling or prompt tokenization in SGLang's multimodal processor.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlparse

import torch

from sglang.srt.entrypoints.valen.encoding import (
    ValenEncodedPrompt,
    encode_valen_schema,
    encode_valen_text,
    valen_messages,
)


@dataclass
class ValenProcessedPrompt:
    encoded: ValenEncodedPrompt
    image_data: list[dict[str, Any]] | None = None
    video_data: list[dict[str, Any]] | None = None


@contextmanager
def _materialize_media(messages):
    """Use SGLang's bounded URL loader before HF decodes downloaded assets."""
    with TemporaryDirectory(prefix="sglang-valen-") as directory:
        index = 0
        for message in messages:
            for item in message["content"]:
                if item["type"] == "text":
                    continue
                url = item["url"]
                if url.startswith("file://"):
                    item["url"] = unquote(urlparse(url).path)
                elif url.startswith(("http://", "https://", "data:")):
                    # The common loaders enforce configured domains, redirects,
                    # byte limits and timeouts. HF must not download again.
                    from sglang.srt.utils import get_image_bytes, get_video_bytes

                    load = (
                        get_image_bytes if item["type"] == "image" else get_video_bytes
                    )
                    path = Path(directory) / f"{index}.media"
                    path.write_bytes(load(url))
                    item["url"] = str(path)
                    index += 1
        yield messages


def encode_valen_prompt(
    tokenizer,
    processor,
    state,
    questions,
    *,
    media_kwargs,
    max_length,
    media_limits=None,
):
    messages = valen_messages(tokenizer, state, allow_media=True)
    # processor_output bundles contain all assets of one modality. Check the
    # actual asset count before bundling, so the server's media limits still apply.
    for kind, limit in (media_limits or {}).items():
        count = sum(item["type"] == kind for m in messages for item in m["content"])
        if count > limit:
            raise ValueError(
                f"Valen request contains {count} {kind} inputs; server limit is {limit}"
            )
    if all(item["type"] == "text" for m in messages for item in m["content"]):
        return ValenProcessedPrompt(
            encode_valen_schema(
                tokenizer,
                encode_valen_text(tokenizer, state),
                questions,
                max_length=max_length,
            )
        )
    if processor is None:
        raise ValueError(
            "Valen image/video requests require the checkpoint's HF processor"
        )
    return encode_valen_media(
        processor, state, questions, media_kwargs=media_kwargs, max_length=max_length
    )


def encode_valen_media(processor, state, questions, *, media_kwargs, max_length):
    """Use the checkpoint's sampling/resizing settings, as Compiler.compile does."""
    messages = valen_messages(processor.tokenizer, state, allow_media=True)
    kwargs = deepcopy(media_kwargs)
    kwargs["return_metadata"] = True
    template_kwargs = {}
    if callable(getattr(processor, "replace_video_token", None)):
        # The trained HF processor replaced the entire wrapped video marker
        # with timestamped, individually wrapped frames. Newer HF processors
        # replace only video_pad, leaving an extra outer pair of vision tokens.
        # Adapt a request-local template before tokenization: removing tokens
        # afterwards would also miss BPE merges adjacent to the timestamps.
        template = processor.chat_template
        marker = (
            processor.vision_start_token
            + processor.video_token
            + processor.vision_end_token
        )
        template_kwargs["chat_template"] = template.replace(
            marker, processor.video_token
        )
    try:
        with _materialize_media(messages):
            processed = dict(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs=kwargs,
                    **template_kwargs,
                )
            )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Unable to process Valen media: {error}") from error
    context_ids = processed["input_ids"]
    if context_ids.ndim != 2 or context_ids.shape[0] != 1:
        raise ValueError("Valen processor must return exactly one state context")
    encoded = encode_valen_schema(
        processor.tokenizer, context_ids[0].tolist(), questions, max_length=max_length
    )

    def payload(kind, feature_key, grid_key):
        features, grid = processed.get(feature_key), processed.get(grid_key)
        if features is None and grid is None:
            return None
        if not isinstance(features, torch.Tensor) or not isinstance(grid, torch.Tensor):
            raise ValueError(f"Valen processor returned incomplete {kind} features")
        return [
            {
                "format": "processor_output",
                "modality": kind,
                "input_ids": encoded.input_ids,
                feature_key: features,
                grid_key: grid,
            }
        ]

    images = payload("image", "pixel_values", "image_grid_thw")
    videos = payload("video", "pixel_values_videos", "video_grid_thw")
    # Some Qwen processor versions expose temporal spacing. Preserve it when
    # present so SGLang can compute positions for the already sampled frames.
    if videos:
        for name in ("second_per_grid_ts", "video_second_per_grid"):
            if name in processed:
                videos[0][name] = processed[name]
    expected = {
        item["type"]
        for m in messages
        for item in m["content"]
        if item["type"] != "text"
    }
    if ("image" in expected and images is None) or (
        "video" in expected and videos is None
    ):
        raise ValueError("Valen processor did not return the requested media features")
    return ValenProcessedPrompt(encoded, images, videos)


def validate_valen_media_payload(obj):
    """Raw media would resample/re-tokenize an already finalized Valen layout."""
    if obj.audio_data:
        raise ValueError("Valen supports image/video inputs, not audio")
    for group in (obj.image_data, obj.video_data):
        if group is None:
            continue
        items = group if isinstance(group, list) else [group]
        for item in items:
            if not isinstance(item, dict) or item.get("format") != "processor_output":
                raise ValueError("Valen media needs HF processor_output from /v1/valen")
            if item.get("input_ids") != obj.input_ids:
                raise ValueError(
                    "Valen media token IDs must match the finalized prompt"
                )
