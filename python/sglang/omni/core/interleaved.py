# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for AR-driven interleaved media generation.

The model-specific policy owns token grammar; this module only names the handoff
metadata used when AR output asks the generation backend to produce media.
"""

from __future__ import annotations

from typing import Literal

BoundaryModality = Literal["image", "audio", "video"]
GenerationTokenCountSource = Literal["request", "ar"]

INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY = "generation_boundary"
INTERLEAVED_BOUNDARY_MODALITY_KEY = "modality"
INTERLEAVED_BOUNDARY_TOKEN_ID_KEY = "boundary_token_id"
INTERLEAVED_BOUNDARY_POSITION_ID_KEY = "boundary_position_id"
INTERLEAVED_GENERATION_TOKEN_COUNT_KEY = "generation_token_count"
INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY = "generation_token_count_source"
STREAMED_TEXT_METADATA_KEY = "_omni_streamed_text"


def get_ar_decode_input_position(next_output_position: int) -> int:
    """Return the replayed input position used to sample the next AR token."""

    return max(0, int(next_output_position) - 1)


def ar_output_position(decode_input_position: int) -> int:
    return int(decode_input_position) + 1


def get_ar_next_output_position(output_position: int) -> int:
    return int(output_position) + 1


def ar_appended_token_positions(
    *,
    previous_decode_input_position: int,
    token_count: int,
) -> list[int]:
    token_count = int(token_count)
    if token_count <= 0:
        return []
    first_position = ar_output_position(previous_decode_input_position)
    return list(range(first_position, first_position + token_count))


def build_generation_boundary_metadata(
    *,
    modality: BoundaryModality,
    token_id: int | None = None,
    position_id: int | None = None,
    generation_token_count: int | None = None,
    generation_token_count_source: GenerationTokenCountSource | None = None,
) -> dict[str, int | str]:
    """Metadata for the AR token that hands control to media generation.

    Future MoT models may decide media token budgets in AR, while current
    diffusion-style models often read shape/steps from the request.
    """

    metadata: dict[str, int | str] = {
        INTERLEAVED_BOUNDARY_MODALITY_KEY: modality,
    }
    if token_id is not None:
        metadata[INTERLEAVED_BOUNDARY_TOKEN_ID_KEY] = int(token_id)
    if position_id is not None:
        metadata[INTERLEAVED_BOUNDARY_POSITION_ID_KEY] = int(position_id)
    if generation_token_count is not None:
        metadata[INTERLEAVED_GENERATION_TOKEN_COUNT_KEY] = int(generation_token_count)
        if generation_token_count_source is not None:
            metadata[INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY] = (
                generation_token_count_source
            )
    return metadata
