import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
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
                request
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
            logger.exception(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Generate request ID based on request type"""
        return None

        # TODO(chang): the rid is used in io_strcut check and often violates `The rid should be a list` AssertionError
        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse]:
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
    ) -> Union[Any, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        # TODO: remove fastapi dependency in openai and move response handling to the entrypoint
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
