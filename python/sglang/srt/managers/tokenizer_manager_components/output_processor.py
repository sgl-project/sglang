from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from sglang.srt.managers.tokenizer_manager_components.lora_controller import (
    LoraController,
)
from sglang.srt.managers.tokenizer_manager_components.request_log_manager import (
    RequestLogManager,
)
from sglang.srt.managers.tokenizer_manager_components.request_metrics_recorder import (
    RequestMetricsRecorder,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputProcessorConfig:
    weight_version: Optional[str]
    batch_notify_size: int
    incremental_streaming_output: bool
    enable_metrics: bool
    skip_tokenizer_init: bool
    speculative_algorithm: str
    speculative_num_draft_tokens: int
    dp_size: int
    enable_lora: bool
    served_model_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputProcessor:
    """Consumes BatchStrOutput / BatchTokenIDOutput / BatchEmbeddingOutput from scheduler."""

    rid_to_state: Dict[str, ReqState]
    tokenizer: Optional[Any]
    request_metrics_recorder: RequestMetricsRecorder
    request_log_manager: RequestLogManager
    lora_controller: LoraController
    send_to_scheduler: Any
    config: OutputProcessorConfig
