"""Draft components that need not mirror the target; config specs only, no torch."""

from sglang.srt.layers.draft.ffn_spec import (
    FFN_KIND_DENSE,
    FFN_KIND_MOE,
    DraftDenseFFNSpec,
    DraftMixFFNSpec,
    DraftMoeFFNSpec,
)

__all__ = [
    "FFN_KIND_DENSE",
    "FFN_KIND_MOE",
    "DraftDenseFFNSpec",
    "DraftMixFFNSpec",
    "DraftMoeFFNSpec",
]
