from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedRequestBuilderConfig:
    vocab_size: int
    preferred_sampling_params: Optional[dict]
    sampling_params_class: Type[SamplingParams]
    disaggregation_transfer_backend: str


@dataclass(slots=True, kw_only=True)
class TokenizedRequestBuilder:
    """Build TokenizedGenerateReqInput / TokenizedEmbeddingReqInput from
    (obj, input_ids, mm_inputs, ...). fake_bootstrap_room_counter mutates per build.
    """

    tokenizer: Optional[Any]
    config: TokenizedRequestBuilderConfig
    fake_bootstrap_room_counter: int = 0
