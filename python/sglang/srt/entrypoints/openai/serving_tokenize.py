import logging
from http import HTTPStatus
from typing import List, Optional, Union

from fastapi import Request

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
from sglang.srt.managers.io_struct import GenerateReqInput

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
        self,
        request: Union[TokenizeRequest, ChatCompletionRequest, CompletionRequest],
        raw_request: Request,
    ) -> tuple[GenerateReqInput, Union[ChatCompletionRequest, CompletionRequest]]:
        add_special_tokens = True
        if isinstance(request, TokenizeRequest):
            if request.messages is not None:
                request = request.to_chat_completion_request()
            else:
                add_special_tokens = request.add_special_tokens
                request = CompletionRequest.model_validate(
                    request.model_dump(exclude_unset=True, exclude_none=True)
                )

        if isinstance(request, ChatCompletionRequest):
            if error := self.chat_serving._validate_request(request):
                raise ValueError(error)
            return self.chat_serving._convert_to_internal_request(request, raw_request)

        if error := self.completion_serving._validate_request(request):
            raise ValueError(error)
        adapted, request = self.completion_serving._convert_to_internal_request(
            request, raw_request
        )
        if not add_special_tokens and adapted.text is not None:
            encode = self.tokenizer_manager.tokenizer.encode
            if isinstance(adapted.text, str):
                adapted.input_ids = encode(adapted.text, add_special_tokens=False)
            else:
                adapted.input_ids = [
                    encode(text, add_special_tokens=False) for text in adapted.text
                ]
            adapted.text = None
        return adapted, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: Union[ChatCompletionRequest, CompletionRequest],
        raw_request: Request,
    ) -> TokenizeResponse:
        tokens = await self.tokenizer_manager.tokenize_request(
            adapted_request, raw_request
        )
        count = (
            [len(ids) for ids in tokens]
            if tokens and isinstance(tokens[0], list)
            else len(tokens)
        )
        max_model_len = getattr(
            self.tokenizer_manager.tokenizer, "model_max_length", -1
        )
        # ORJSON accepts signed negative and unsigned positive 64-bit integers.
        if not isinstance(max_model_len, int) or not -(2**63) <= max_model_len < 2**64:
            max_model_len = self.tokenizer_manager.model_config.context_len
        return TokenizeResponse(tokens=tokens, count=count, max_model_len=max_model_len)

    # A generation payload with stream=true still has one tokenize response.
    _handle_streaming_request = _handle_non_streaming_request


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
                if (error := self._out_of_vocab_error(request.tokens)) is not None:
                    return error
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
                    if (error := self._out_of_vocab_error(token_list)) is not None:
                        return error
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

    def _out_of_vocab_error(self, token_ids: List[int]) -> Optional[ErrorResponse]:
        vocab_size = self.tokenizer_manager.model_config.vocab_size
        for token_id in token_ids:
            if token_id < 0 or token_id >= vocab_size:
                return self.create_error_response(
                    f"Invalid input: token id {token_id} is out of vocabulary; "
                    f"valid range is [0, {vocab_size})."
                )
        return None
