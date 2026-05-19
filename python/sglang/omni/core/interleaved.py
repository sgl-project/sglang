# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for AR-driven interleaved media generation.

The model-specific hooks own token grammar; this module only names the handoff
metadata used when AR output asks the generation backend to produce media.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

MediaModality = Literal["image", "audio", "video"]
GenerationTokenCountSource = Literal["request", "ar"]

INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY = "generation_boundary"
INTERLEAVED_BOUNDARY_MODALITY_KEY = "modality"
INTERLEAVED_BOUNDARY_TOKEN_ID_KEY = "boundary_token_id"
INTERLEAVED_BOUNDARY_POSITION_ID_KEY = "boundary_position_id"
INTERLEAVED_GENERATION_TOKEN_COUNT_KEY = "generation_token_count"
INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY = "generation_token_count_source"
STREAMED_TEXT_METADATA_KEY = "_omni_streamed_text"
TEXT_ROLE_METADATA_KEY = "role"
TEXT_ROLE_THINK = "think"
_BOUNDARY_MODALITIES = {"image", "audio", "video"}
_GENERATION_TOKEN_COUNT_SOURCES = {"request", "ar"}


@dataclass(frozen=True, slots=True)
class GenerationBoundaryMetadata:
    """Structured metadata for an AR token that hands off to media generation.

    The coordinator only sees the target modality; model adapters can also
    attach the marker token position and media token budget when AR decides it.
    """

    modality: MediaModality
    token_id: int | None = None
    position_id: int | None = None
    # the number of tokens of the media to generate
    generation_token_count: int | None = None
    # who decides the number of tokens
    generation_token_count_source: GenerationTokenCountSource | None = None

    def __post_init__(self) -> None:
        if self.modality not in _BOUNDARY_MODALITIES:
            raise ValueError(f"Unsupported boundary modality: {self.modality!r}")
        if (
            self.generation_token_count_source is not None
            and self.generation_token_count_source
            not in _GENERATION_TOKEN_COUNT_SOURCES
        ):
            raise ValueError(
                "Unsupported generation token count source: "
                f"{self.generation_token_count_source!r}"
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        default_modality: MediaModality | None = None,
    ) -> "GenerationBoundaryMetadata":
        """Parse boundary metadata at the AR/backend boundary."""

        raw_modality = metadata.get(INTERLEAVED_BOUNDARY_MODALITY_KEY)
        if raw_modality is None:
            raw_modality = default_modality
        if raw_modality not in _BOUNDARY_MODALITIES:
            raise ValueError(f"Unsupported boundary modality: {raw_modality!r}")

        token_count_source = metadata.get(INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY)
        if (
            token_count_source is not None
            and token_count_source not in _GENERATION_TOKEN_COUNT_SOURCES
        ):
            raise ValueError(
                "Unsupported generation token count source: " f"{token_count_source!r}"
            )
        return cls(
            modality=raw_modality,
            token_id=_optional_int(metadata.get(INTERLEAVED_BOUNDARY_TOKEN_ID_KEY)),
            position_id=_optional_int(
                metadata.get(INTERLEAVED_BOUNDARY_POSITION_ID_KEY)
            ),
            generation_token_count=_optional_int(
                metadata.get(INTERLEAVED_GENERATION_TOKEN_COUNT_KEY)
            ),
            generation_token_count_source=token_count_source,
        )

    def to_metadata(self) -> dict[str, int | str]:
        metadata: dict[str, int | str] = {
            INTERLEAVED_BOUNDARY_MODALITY_KEY: self.modality,
        }
        if self.token_id is not None:
            metadata[INTERLEAVED_BOUNDARY_TOKEN_ID_KEY] = int(self.token_id)
        if self.position_id is not None:
            metadata[INTERLEAVED_BOUNDARY_POSITION_ID_KEY] = int(self.position_id)
        if self.generation_token_count is not None:
            metadata[INTERLEAVED_GENERATION_TOKEN_COUNT_KEY] = int(
                self.generation_token_count
            )
            if self.generation_token_count_source is not None:
                metadata[INTERLEAVED_GENERATION_TOKEN_COUNT_SOURCE_KEY] = (
                    self.generation_token_count_source
                )
        return metadata


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
    modality: MediaModality,
    token_id: int | None = None,
    position_id: int | None = None,
    generation_token_count: int | None = None,
    generation_token_count_source: GenerationTokenCountSource | None = None,
) -> dict[str, int | str]:
    """Metadata for the AR token that hands control to media generation.

    Future MoT models may decide media token budgets in AR, while current
    diffusion-style models often read shape/steps from the request.
    """

    return GenerationBoundaryMetadata(
        modality=modality,
        token_id=token_id,
        position_id=position_id,
        generation_token_count=generation_token_count,
        generation_token_count_source=generation_token_count_source,
    ).to_metadata()
