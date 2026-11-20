"""Engine entry points into the configured watermark algorithm.

The only place that decides which algorithm runs: each entry point dispatches on
``--watermark-algorithm`` to that algorithm's adapter module.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.sampling.watermarking.interface import (
        WatermarkEngine,
        WatermarkRows,
    )

_ADAPTERS: dict[str, ModuleType] = {}

# Provider ABI v2 (`VerifySamplingView.watermark_*`) fields of a batch without
# watermark rows.
_NO_WATERMARK_VIEW_FIELDS: dict[str, Any] = {
    "watermark_keys": None,
    "watermark_context_windows": None,
    "watermark_enabled": None,
    "has_watermark_candidates": False,
}


def make_watermark_engine(
    *,
    max_num_reqs: int,
    max_contexts_per_req: int,
    vocab_size: int,
    device: str,
) -> Optional[WatermarkEngine]:
    features = get_exec().features
    if not features.enable_watermark:
        return None
    return _ADAPTERS[features.watermark_algorithm].make_engine(
        features,
        max_num_reqs=max_num_reqs,
        max_contexts_per_req=max_contexts_per_req,
        vocab_size=vocab_size,
        device=device,
    )


def build_watermark_rows(
    reqs: list[Req], device: torch.device | str
) -> Optional[WatermarkRows]:
    features = get_exec().features
    if not features.enable_watermark:
        return None
    return _ADAPTERS[features.watermark_algorithm].build_rows(
        reqs, features=features, device=device
    )


def watermark_request_enabled(sampling_params: SamplingParams) -> bool:
    """Whether the request is watermarked; raises ValueError for an invalid request."""
    features = get_exec().features
    return _ADAPTERS[features.watermark_algorithm].request_enabled(
        sampling_params, features=features
    )


def check_watermark_algorithm_args(cfg: Any) -> None:
    """The configured algorithm's own server-argument checks."""
    if cfg.watermark_algorithm not in _ADAPTERS:
        raise ValueError(
            f"unknown --watermark-algorithm {cfg.watermark_algorithm!r}; "
            f"choose from {sorted(_ADAPTERS)}"
        )
    _ADAPTERS[cfg.watermark_algorithm].check_server_args(cfg)


def watermark_view_fields(rows: Optional[WatermarkRows]) -> dict[str, Any]:
    """The provider ABI v2 watermark fields of ``VerifySamplingView`` for these rows."""
    if rows is None:
        return _NO_WATERMARK_VIEW_FIELDS
    return _ADAPTERS[get_exec().features.watermark_algorithm].watermark_view_fields(
        rows
    )
