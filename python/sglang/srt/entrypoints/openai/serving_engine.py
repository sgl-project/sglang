# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    OpenAIServingRequest,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.utils import create_error_response
from sglang.srt.entrypoints.openai.validation import get_validation_rules
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class RequestContext:
    """Context object for tracking request state throughout the pipeline"""

    def __init__(
        self,
        raw_request: Request,
        openai_request: OpenAIServingRequest,
        request_id: str,
    ):
        self.raw_request = raw_request
        self.openai_request = openai_request
        self.request_id = request_id
        self.start_time = time.time()
        self.metadata: Dict[str, Any] = {}

    def elapsed_time(self) -> float:
        """Get elapsed time since request started"""
        return time.time() - self.start_time

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the request context"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the request context"""
        return self.metadata.get(key, default)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    @abstractmethod
    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Any:
        """Handle the specific request type"""
        pass

    def _validate_request(
        self, request: OpenAIServingRequest
    ) -> Optional[ErrorResponse]:
        """Validate request"""
        validation_rules = get_validation_rules(request)
        for rule in validation_rules:
            param_value = rule.param_getter(request)
            error_msg = rule.validator_func(param_value)
            if error_msg:
                return create_error_response(error_msg, param=rule.param_name)
        return None

    def _calculate_streaming_usage_base(
        self,
        prompt_tokens: Dict[int, int],
        completion_tokens: Dict[int, int],
        cached_tokens: Dict[int, int],
        n_choices: int,
    ) -> UsageInfo:
        """Calculate usage information for streaming responses (common logic)"""
        total_prompt_tokens = sum(
            tokens for i, tokens in prompt_tokens.items() if i % n_choices == 0
        )
        total_completion_tokens = sum(tokens for tokens in completion_tokens.values())

        cache_report = self.tokenizer_manager.server_args.enable_cache_report
        prompt_tokens_details = None
        if cache_report:
            cached_tokens_sum = sum(tokens for tokens in cached_tokens.values())
            if cached_tokens_sum > 0:
                prompt_tokens_details = {"cached_tokens": cached_tokens_sum}

        return UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            prompt_tokens_details=prompt_tokens_details,
        )
