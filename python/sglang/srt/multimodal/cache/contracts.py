"""Model-facing contracts for early multimodal preprocess-cache lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.multimodal.cache.identity import MediaSnapshot


@dataclass(frozen=True)
class PreprocessCacheLookup:
    """Prompt-independent metadata found before the full processor runs.

    ``processor_state`` is opaque to the serving layer. A model processor owns
    its type and receives it back on the miss path. Only final processor-output
    hashes cross the generic scheduler boundary.
    """

    processor_state: Any
    feature_hashes: tuple[Optional[int], ...]
    feature_identities: tuple[Optional[str], ...]
    identity_sources: tuple[str, ...]


@dataclass(frozen=True)
class EncoderPreprocessArtifact:
    """Small CPU-only sidecar needed to reuse one encoder embedding."""

    content_digest: str
    artifact_key: str
    feature_hash: int
    grid_thw: tuple[int, ...]
    original_size: tuple[int, int]


@dataclass(frozen=True)
class EncoderMediaLookup:
    """Verified media snapshot plus its optional cached encoder metadata."""

    source: Any
    expected_digest: Optional[str]
    content_digest: str
    artifact_key: str
    snapshot: Optional[MediaSnapshot]
    artifact: Optional[EncoderPreprocessArtifact]
