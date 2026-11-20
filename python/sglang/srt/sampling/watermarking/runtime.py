"""Engine entry points into the configured watermark algorithm.

The only place that decides which algorithm runs: each entry point dispatches on
``--watermark-algorithm`` to that algorithm's adapter module, which implements
``interface.WatermarkAdapter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.runtime_context import get_exec
from sglang.srt.sampling.watermarking.gumbel import adapter as gumbel_adapter
from sglang.srt.sampling.watermarking.interface import NO_WATERMARK_VIEW_FIELDS
from sglang.srt.sampling.watermarking.synthid import adapter as synthid_adapter

if TYPE_CHECKING:
    from sglang.srt.arg_groups.model_override_base import ResolvingConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.sampling.watermarking.interface import (
        WatermarkAdapter,
        WatermarkEngine,
        WatermarkRows,
    )

_ADAPTERS: dict[str, WatermarkAdapter] = {
    "gumbel": gumbel_adapter,
    "synthid": synthid_adapter,
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
    features = get_exec().features
    return _ADAPTERS[features.watermark_algorithm].request_enabled(
        sampling_params, features=features
    )


def check_watermark_algorithm_args(cfg: ResolvingConfig) -> None:
    _ADAPTERS[cfg.watermark_algorithm].check_server_args(cfg)


def watermark_view_fields(rows: Optional[WatermarkRows]) -> dict[str, Any]:
    if rows is None:
        return NO_WATERMARK_VIEW_FIELDS
    return _ADAPTERS[get_exec().features.watermark_algorithm].watermark_view_fields(
        rows
    )
