"""Model-facing contracts for early multimodal preprocess-cache lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
