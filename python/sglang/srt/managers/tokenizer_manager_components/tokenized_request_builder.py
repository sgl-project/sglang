from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedRequestBuilderConfig:
    vocab_size: int
    preferred_sampling_params: Optional[dict]
    sampling_params_class: Type[SamplingParams]
    disaggregation_transfer_backend: str


@dataclass(slots=True, kw_only=True)
class TokenizedRequestBuilder:
    tokenizer: Optional[Any]
    config: TokenizedRequestBuilderConfig
    rid_to_state: Dict[str, ReqState]
    fake_bootstrap_room_counter: int = 0
