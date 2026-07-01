import logging
from http import HTTPStatus
from typing import Any, Dict, List, Union

from fastapi import HTTPException, Request
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.encoding_dsv32 import DS32EncodingError
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.observability.req_time_stats import monotonic_time

logger = logging.getLogger(__name__)


class OpenAIServingTokenize(OpenAIServingBase):
    """Handler for /v1/tokenize requests"""

    def __init__(
        self,
        tokenizer_manager,
        chat_serving: OpenAIServingChat,
        completion_serving: OpenAIServingCompletion,
    ):
        super().__init__(tokenizer_manager)
        self.chat_serving = chat_serving
        self.completion_serving = completion_serving

    def _request_id_prefix(self) -> str:
        return "tok-"

    def _convert_to_internal_request(
        self, request: TokenizeRequest, raw_request: Request
    ) -> tuple[TokenizeRequest, TokenizeRequest]:
        return request, request

    def _validate_request(self, _: TokenizeRequest) -> Union[str, None]:
        return None

    async def handle_request(
        self, request: Union[TokenizeRequest, Dict[str, Any]], raw_request: Request
    ) -> Union[TokenizeResponse, ErrorResponse]:
        try:
            body = self._request_to_dict(request)
            has_messages = body.get("messages") is not None
            has_prompt = body.get("prompt") is not None
            if has_messages == has_prompt:
                return self.create_error_response(
                    "Exactly one of 'prompt' or 'messages' must be provided."
                )

            if has_messages:
                body.pop("prompt", None)
                body.pop("add_special_tokens", None)
                chat_request = ChatCompletionRequest.model_validate(body)
                return await self.handle_chat_request(chat_request, raw_request)

            body.pop("messages", None)
            for field_name in (
                "tools",
                "tool_choice",
                "reasoning_effort",
                "continue_final_message",
                "chat_template_kwargs",
                "add_special_tokens",
            ):
                body.pop(field_name, None)
            completion_request = CompletionRequest.model_validate(body)
            return await self.handle_completions_request(
                completion_request, raw_request
            )
        except Exception as e:
            return self._create_exception_response(e)

    async def handle_chat_request(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[TokenizeResponse, ErrorResponse]:
        return await self._handle_tokenize_request(
            self.chat_serving, request, raw_request
        )

    async def handle_completions_request(
        self, request: CompletionRequest, raw_request: Request
    ) -> Union[TokenizeResponse, ErrorResponse]:
        return await self._handle_tokenize_request(
            self.completion_serving, request, raw_request
        )

    async def _handle_tokenize_request(
        self,
        serving: Union[OpenAIServingChat, OpenAIServingCompletion],
        request: Union[ChatCompletionRequest, CompletionRequest],
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse]:
        received_time = monotonic_time()
        try:
            error_msg = serving._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            self._log_openai_request(request, raw_request)
            adapted_request, _ = serving._convert_to_internal_request(
                request, raw_request
            )
            adapted_request.received_time = received_time
            token_ids = await self.tokenizer_manager.tokenize_request(
                adapted_request, raw_request
            )
            return self._create_tokenize_response(token_ids)
        except Exception as e:
            return self._create_exception_response(e)

    def _request_to_dict(
        self, request: Union[TokenizeRequest, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(request, dict):
            return dict(request)

        data = request.model_dump(exclude_none=True)
        extra = getattr(request, "__pydantic_extra__", None)
        if extra:
            data.update(extra)
        return data

    def _log_openai_request(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        raw_request: Request,
    ):
        request_logger = self.tokenizer_manager.request_logger
        if request_logger.log_requests and request_logger.log_requests_level >= 2:
            request_logger.log_openai_received_request(request, request=raw_request)

    def _create_tokenize_response(
        self, tokens: Union[list[int], list[list[int]]]
    ) -> TokenizeResponse:
        count: Union[int, list[int]]
        if tokens and isinstance(tokens[0], list):
            count = [len(token_ids) for token_ids in tokens]
        else:
            count = len(tokens)

        return TokenizeResponse(
            tokens=tokens,
            count=count,
            max_model_len=self._get_max_model_len(),
        )

    def _get_max_model_len(self) -> int:
        tokenizer = self.tokenizer_manager.tokenizer
        max_model_len = getattr(tokenizer, "model_max_length", -1)
        if not isinstance(max_model_len, int):
            model_config = getattr(self.tokenizer_manager, "model_config", None)
            max_model_len = getattr(model_config, "context_len", -1)
        return max_model_len if isinstance(max_model_len, int) else -1

    def _create_exception_response(self, e: Exception):
        if isinstance(e, HTTPException):
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        if isinstance(e, (ValueError, ValidationError, DS32EncodingError)):
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        logger.exception(f"Error during tokenization: {e}")
        return self.create_error_response(
            f"Internal server error during tokenization: {e}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class OpenAIServingDetokenize(OpenAIServingBase):
    """Handler for /v1/detokenize requests"""

    def _request_id_prefix(self) -> str:
        return "detok-"

    def _convert_to_internal_request(
        self, request: DetokenizeRequest, raw_request: Request
    ) -> tuple[DetokenizeRequest, DetokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: DetokenizeRequest,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> Union[DetokenizeResponse, ErrorResponse]:
        try:
            tokenizer = self.tokenizer_manager.tokenizer

            if (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], int)
            ):
                if not all(isinstance(t, int) for t in request.tokens):
                    return self.create_error_response(
                        "Invalid input: 'tokens' must be a list of integers."
                    )
                tokens_to_decode = [int(t) for t in request.tokens]
                text = tokenizer.decode(
                    tokens_to_decode, skip_special_tokens=request.skip_special_tokens
                )
                text_out: Union[str, List[str]] = text
            elif (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], list)
            ):
                texts: List[str] = []
                for token_list in request.tokens:
                    if not all(isinstance(t, int) for t in token_list):
                        return self.create_error_response(
                            f"Invalid input: Sublist in 'tokens' must contain only integers. Found: {token_list}"
                        )
                    decoded_text = tokenizer.decode(
                        [int(t) for t in token_list],
                        skip_special_tokens=request.skip_special_tokens,
                    )
                    texts.append(decoded_text)
                text_out = texts
            elif isinstance(request.tokens, list) and not request.tokens:
                text_out = ""
            else:
                return self.create_error_response(
                    f"Invalid tokens type: {type(request.tokens)}. Expected List[int] or List[List[int]]."
                )

            return DetokenizeResponse(text=text_out)
        except Exception as e:
            logger.error("Error during detokenization", exc_info=True)
            if "decode" in str(e).lower():
                return self.create_error_response(
                    f"Error decoding tokens: {e}. Input tokens might be invalid for the model.",
                    err_type="DecodeError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            return self.create_error_response(
                f"Internal server error during detokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
