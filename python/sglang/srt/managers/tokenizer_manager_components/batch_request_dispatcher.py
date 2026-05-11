from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager_components.request_preparer import (
    RequestPreparer,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.managers.tokenizer_manager_components.response_emitter import (
    ResponseEmitter,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRequestDispatcherConfig:
    enable_trace: bool
    disaggregation_mode: DisaggregationMode


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRequestDispatcher:
    request_preparer: RequestPreparer
    response_emitter: ResponseEmitter
    rid_to_state: Dict[str, ReqState]
    send_to_scheduler: Any
    send_one_request: Callable[..., None]
    send_batch_request: Callable[..., None]
    config: BatchRequestDispatcherConfig
