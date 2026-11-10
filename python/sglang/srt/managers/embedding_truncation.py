# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request-scoped embedding truncation after multimodal preprocessing.

This module deliberately does not change tokenizer defaults or server settings.
Media features are never silently dropped or partially sliced.
"""

from typing import Literal, Optional, Union

import msgspec
import torch

from sglang.srt.managers.req_types import MultimodalProcessorOutput


def embedding_input_token_limit(
    *,
    context_len: int,
    num_reserved_tokens: int,
    max_req_input_len: Optional[int],
    max_prefill_tokens: Optional[int] = None,
) -> int:
    """Return the inclusive input budget accepted by the existing serving path.

    Both the tokenizer's context check and the scheduler's input check use >=,
    so their upper bounds are exclusive. The scheduler may advertise a smaller
    limit than context_len because of KV capacity and reserved slots. Without
    chunked prefill, the entire input must also fit in one prefill.
    """
    limits = [context_len - num_reserved_tokens - 1]
    if max_req_input_len is not None:
        limits.append(max_req_input_len - 1)
    if max_prefill_tokens is not None:
        limits.append(max_prefill_tokens)
    limit = min(limits)
    if limit < 1:
        raise ValueError("The server has no available embedding input token budget.")
    return limit


def _slice_token_metadata(
    value: Optional[Union[list[int], torch.Tensor]],
    start: int,
    end: int,
    length: int,
    name: str,
) -> Optional[Union[list[int], torch.Tensor]]:
    """Slice a flat list or a tensor whose final axis indexes prompt tokens."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if not value.shape or value.shape[-1] != length:
            raise ValueError(f"Cannot safely truncate misaligned {name}.")
        return value[..., start:end]
    if len(value) != length:
        raise ValueError(f"Cannot safely truncate misaligned {name}.")
    return value[start:end]


def _truncate_multimodal_tail(
    mm_inputs: MultimodalProcessorOutput, input_ids: list[int], end: int
) -> MultimodalProcessorOutput:
    """Keep all media and trim only a provably text-only tail.

    Offsets are inclusive and may contain several rows for one image. Retain
    the token following each item's last feature span too: processors such as
    Pixtral place an image-end delimiter there. Model-specific positional
    metadata needs a dedicated adapter and is rejected instead of guessed.
    """
    for name, value in (
        ("mrope_positions", mm_inputs.mrope_positions),
        ("mrope_position_delta", mm_inputs.mrope_position_delta),
        ("vision_position_ids", mm_inputs.vision_position_ids),
    ):
        if value is not None:
            raise ValueError(
                f"Cannot safely truncate multimodal input with {name}; "
                "increase max_tokens or shorten the input."
            )

    spans = []
    for item in mm_inputs.mm_items:
        if not item.offsets:
            raise ValueError(
                "Cannot safely truncate multimodal input without media token offsets."
            )
        for start, stop in item.offsets:
            if start < 0 or stop < start or stop >= len(input_ids):
                raise ValueError("Cannot safely truncate invalid media token offsets.")
            spans.append((start, stop))
        # Do not cut a feature span, remove a media item, or remove its delimiter.
        if max(stop for _, stop in item.offsets) + 1 >= end:
            raise ValueError(
                "Truncation would remove media tokens or their closing delimiter; "
                "increase max_tokens or shorten the text input."
            )
    spans.sort()
    if any(left[1] >= right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Cannot safely truncate overlapping media token offsets.")

    # Also retain any explicit media boundaries not included in feature offsets.
    special_ids = {
        value
        for value in (
            mm_inputs.im_token_id,
            mm_inputs.im_start_id,
            mm_inputs.im_end_id,
            mm_inputs.slice_start_id,
            mm_inputs.slice_end_id,
            mm_inputs.video_token_id,
            mm_inputs.audio_token_id,
            mm_inputs.audio_start_id,
            mm_inputs.audio_end_id,
        )
        if value is not None
    }
    if any(token in special_ids for token in input_ids[end:]):
        raise ValueError("Truncation would remove a multimodal boundary token.")

    return msgspec.structs.replace(
        mm_inputs,
        input_ids=input_ids[:end],
        padded_input_ids=_slice_token_metadata(
            mm_inputs.padded_input_ids,
            0,
            end,
            len(input_ids),
            "multimodal padded_input_ids",
        ),
        token_type_ids=_slice_token_metadata(
            mm_inputs.token_type_ids,
            0,
            end,
            len(input_ids),
            "multimodal token_type_ids",
        ),
    )


def truncate_embedding_input(
    input_ids,
    *,
    truncate: Literal["END", "START", "NONE"],
    max_tokens: Optional[int],
    input_token_limit: int,
    token_type_ids=None,
    mm_inputs: Optional[MultimodalProcessorOutput] = None,
    special_tokens_mask=None,
):
    """Return aligned IDs and metadata without mutating the supplied request.

    END retains the beginning and START retains the end. NONE enforces the
    budget even when the server's legacy auto-truncation option is enabled.
    -1 means the server limit only for END/START; NONE checks the literal value.
    A zero END/START budget produces no tokens, which the caller must reject.
    For single-sequence text, a tokenizer-provided mask preserves tokens added
    by its template, while truncating only content tokens. Explicit token IDs
    and processor-expanded multimodal inputs retain their existing semantics.
    """
    if truncate not in ("END", "START", "NONE"):
        raise ValueError("embedding_truncate must be END, START, or NONE.")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < -1
    ):
        raise ValueError("max_tokens must be an integer greater than or equal to -1.")
    if input_ids is None:
        raise ValueError(
            "Embedding input token IDs are unavailable after preprocessing."
        )
    if truncate != "NONE" and max_tokens is not None and max_tokens > input_token_limit:
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the server's embedding input budget "
            f"of {input_token_limit} tokens."
        )
    if max_tokens is None or (max_tokens == -1 and truncate != "NONE"):
        limit = input_token_limit
    else:
        limit = min(max_tokens, input_token_limit)
    length = len(input_ids)
    if mm_inputs is not None and mm_inputs.mm_items and mm_inputs.input_ids is None:
        raise ValueError(
            "Cannot apply an embedding token budget to a processor that defers "
            "multimodal token expansion to the scheduler."
        )
    padded_ids = mm_inputs.padded_input_ids if mm_inputs is not None else None
    if padded_ids is not None and len(padded_ids) != length:
        raise ValueError(
            "Cannot apply an embedding token budget before multimodal expansion "
            "is complete: padded_input_ids and input_ids have different lengths."
        )
    if length <= limit:
        return input_ids, token_type_ids, mm_inputs
    if truncate == "NONE":
        raise ValueError(
            f"Input of {length} tokens exceeds the embedding input budget of "
            f"{limit} tokens with truncate=NONE."
        )

    if special_tokens_mask is not None:
        if mm_inputs is not None:
            raise ValueError(
                "Text special-token masks cannot truncate multimodal input."
            )
        if len(special_tokens_mask) != length or any(
            value not in (0, 1) for value in special_tokens_mask
        ):
            raise ValueError("Cannot safely truncate misaligned special_tokens_mask.")
        content_budget = limit - sum(special_tokens_mask)
        if content_budget < 0:
            raise ValueError(
                f"Embedding input budget of {limit} tokens cannot fit the "
                "tokenizer's required special tokens."
            )
        content_positions = [
            index for index, special in enumerate(special_tokens_mask) if not special
        ]
        start = 0 if truncate == "END" else len(content_positions) - content_budget
        retained_content = set(content_positions[start : start + content_budget])
        positions = [
            index
            for index, special in enumerate(special_tokens_mask)
            if special or index in retained_content
        ]
        if token_type_ids is not None:
            # Validate the original alignment before selecting noncontiguous
            # positions (e.g. BOS + a retained text suffix + EOS).
            _slice_token_metadata(token_type_ids, 0, length, length, "token_type_ids")
            token_type_ids = (
                token_type_ids[..., positions]
                if isinstance(token_type_ids, torch.Tensor)
                else [token_type_ids[index] for index in positions]
            )
        return [input_ids[index] for index in positions], token_type_ids, mm_inputs

    start, end = (0, limit) if truncate == "END" else (length - limit, length)
    if mm_inputs is not None:
        if truncate == "START":
            raise ValueError(
                "Cannot safely apply truncate=START to multimodal input: "
                "processor-specific positions cannot be rebased generically. "
                "Increase max_tokens or shorten the text input."
            )
        mm_inputs = _truncate_multimodal_tail(mm_inputs, input_ids, end)
    token_type_ids = _slice_token_metadata(
        token_type_ids, start, end, length, "token_type_ids"
    )
    return input_ids[start:end], token_type_ids, mm_inputs
