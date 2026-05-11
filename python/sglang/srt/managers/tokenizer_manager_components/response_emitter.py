from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sglang.srt.managers.tokenizer_manager_components.lora_controller import (
    LoraController,
)
from sglang.srt.managers.tokenizer_manager_components.request_log_manager import (
    RequestLogManager,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseEmitter:
    """Drains rid_to_state[rid].out_list and yields per-request dicts to HTTP clients."""

    rid_to_state: Dict[str, ReqState]
    lora_controller: LoraController
    request_log_manager: RequestLogManager
    abort_request: Callable[..., None]
    server_args: ServerArgs
