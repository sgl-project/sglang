import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    OpenAIServingRequest,
    UsageInfo,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern"""
        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                [request], [self._generate_request_id_base(request)]
            )

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )

        except Exception as e:
            logger.error(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> str:
        """Generate request ID based on request type"""
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    @abstractmethod
    def _convert_to_internal_request(
        self,
        all_requests: List[OpenAIServingRequest],
        request_ids: List[str],
    ) -> tuple[
        GenerateReqInput, Union[OpenAIServingRequest, List[OpenAIServingRequest]]
    ]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[Any, ErrorResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, request: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

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

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})
