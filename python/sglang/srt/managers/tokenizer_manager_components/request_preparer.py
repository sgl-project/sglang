from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from sglang.srt.managers.tokenizer_manager_components.multimodal_processor import (
    MultimodalProcessor,
)
from sglang.srt.managers.tokenizer_manager_components.raw_tokenizer_wrapper import (
    RawTokenizerWrapper,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.managers.tokenizer_manager_components.request_validator import (
    RequestValidator,
)
from sglang.srt.managers.tokenizer_manager_components.tokenized_request_builder import (
    TokenizedRequestBuilder,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestPreparerConfig:
    skip_tokenizer_init: bool
    enable_dp_attention: bool
    enable_tokenizer_batch_encode: bool
    is_generation: bool
    disable_radix_cache: bool
    is_multimodal: bool
    architectures: List[str]
    language_only: bool
    encoder_transfer_backend: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestPreparer:
    raw_tokenizer_wrapper: RawTokenizerWrapper
    multimodal_processor: MultimodalProcessor
    request_validator: RequestValidator
    tokenized_request_builder: TokenizedRequestBuilder
    rid_to_state: Dict[str, ReqState]
    get_max_req_input_len: Callable[[], Optional[int]]
    config: RequestPreparerConfig
