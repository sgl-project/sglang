from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.allowed_custom_labels = (
            set(
                self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            )
            if isinstance(self.tokenizer_manager.server_args, ServerArgs)
            and self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            else None
        )

    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Parse 'base-model:adapter-name' syntax to extract LoRA adapter.

        Returns (base_model, adapter_name) or (model, None) if no colon present.
        """
        if ":" not in model:
            return model, None

        # Split on first colon only to handle model paths with multiple colons
        parts = model.split(":", 1)
        base_model = parts[0].strip()
        adapter_name = parts[1].strip() or None

        return base_model, adapter_name

    def _resolve_lora_path(
        self,
        request_model: str,
        explicit_lora_path: Optional[Union[str, List[Optional[str]]]],
    ) -> Optional[Union[str, List[Optional[str]]]]:
        """Resolve LoRA adapter with priority: model parameter > explicit lora_path.

        Returns adapter name or None. Supports both single values and lists (batches).
        """
        _, adapter_from_model = self._parse_model_parameter(request_model)

        # Model parameter adapter takes precedence
        if adapter_from_model is not None:
            return adapter_from_model

        # Fall back to explicit lora_path
        return explicit_lora_path

    def _validate_lora_enabled(self, adapter_name: str) -> None:
        """Check that LoRA is enabled before attempting to use an adapter.

        Raises ValueError with actionable guidance if --enable-lora flag is missing.
        Adapter existence is validated later by TokenizerManager.lora_registry.
        """
        if not self.tokenizer_manager.server_args.enable_lora:
            raise ValueError(
                f"LoRA adapter '{adapter_name}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        try:
            # Validate request
            validation_start = time.perf_counter()
            error_msg = self._validate_request(request)
            validation_time = time.perf_counter() - validation_start
            if error_msg:
                return self.create_error_response(error_msg)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )
            if hasattr(adapted_request, "validation_time"):
                adapted_request.validation_time = validation_time

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
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

    def _compute_extra_key(self, request: OpenAIServingRequest) -> Optional[str]:
        """Compute the final extra_key by concatenating cache_salt and extra_key if both are provided."""
        parts = []
        for key in ["cache_salt", "extra_key"]:
            value = getattr(request, key, None)
            if value:
                if not isinstance(value, str):
                    raise TypeError(
                        f"Value of {key} must be a string, but got {type(value).__name__}"
                    )
                parts.append(value)
        return "".join(parts) if parts else None

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
        raw_request: Request = None,
        validation_time: float = None,
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

    def extract_custom_labels(self, raw_request):
        if (
            not self.allowed_custom_labels
            or not self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        ):
            return None

        custom_labels = None
        header = (
            self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        )
        try:
            raw_labels = (
                orjson.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception(f"Error in request: {e}")
            raw_labels = None

        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels
