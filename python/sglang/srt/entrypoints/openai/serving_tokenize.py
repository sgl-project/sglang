import logging
from http import HTTPStatus
from typing import List, Optional, Union

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.conversation import chat_templates

logger = logging.getLogger(__name__)


class OpenAIServingTokenize(OpenAIServingBase):
    """Handler for /v1/tokenize requests"""

    def __init__(self, tokenizer_manager, template_manager=None):
        super().__init__(tokenizer_manager)
        self.chat_serving: Optional[OpenAIServingChat] = (
            OpenAIServingChat(tokenizer_manager, template_manager)
            if template_manager is not None
            else None
        )

    def _request_id_prefix(self) -> str:
        return "tok-"

    def _convert_to_internal_request(
        self, request: TokenizeRequest, raw_request: Request
    ) -> tuple[TokenizeRequest, TokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: TokenizeRequest,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse]:
        try:
            tokenizer = self.tokenizer_manager.tokenizer
            max_model_len = getattr(tokenizer, "model_max_length", -1)
            # ORJSON accepts signed negative and unsigned positive 64-bit integers.
            if isinstance(max_model_len, int) and not (
                -(2**63) <= max_model_len < 2**64
            ):
                max_model_len = self.tokenizer_manager.model_config.context_len

            stop = None
            stop_token_ids = None
            if request.messages is not None:
                token_ids, stop, stop_token_ids = self._tokenize_chat_request(request)
                tokens = token_ids
                count = len(token_ids)
            elif isinstance(request.prompt, str):
                token_ids = tokenizer.encode(
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
                tokens = token_ids
                count = len(token_ids)
            elif isinstance(request.prompt, list):
                token_ids_list = [
                    tokenizer.encode(
                        text, add_special_tokens=request.add_special_tokens
                    )
                    for text in request.prompt
                ]
                tokens = token_ids_list
                count = [len(ids) for ids in token_ids_list]
            else:
                return self.create_error_response(
                    f"Invalid prompt type: {type(request.prompt)}. Expected str or List[str]."
                )

            return TokenizeResponse(
                tokens=tokens,
                count=count,
                max_model_len=max_model_len,
                stop=stop,
                stop_token_ids=stop_token_ids,
            )
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error("Error during tokenization", exc_info=True)
            return self.create_error_response(
                f"Internal server error during tokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _collect_template_stop_token_ids(
        self, chat_request: ChatCompletionRequest
    ) -> Optional[List[int]]:
        stop_ids: set[int] = set()
        if chat_request.stop_token_ids:
            stop_ids.update(chat_request.stop_token_ids)

        if not chat_request.ignore_eos:
            tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
            if tokenizer is not None:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if isinstance(eos_id, int):
                    stop_ids.add(eos_id)
                elif isinstance(eos_id, (list, tuple, set)):
                    stop_ids.update(int(t) for t in eos_id if t is not None)

                add_stops = getattr(tokenizer, "additional_stop_token_ids", None)
                if isinstance(add_stops, int):
                    stop_ids.add(add_stops)
                elif isinstance(add_stops, (list, tuple, set)):
                    stop_ids.update(int(t) for t in add_stops if t is not None)

            model_config = getattr(self.tokenizer_manager, "model_config", None)
            if model_config is not None:
                hf_eos = getattr(model_config, "hf_eos_token_id", None)
                if isinstance(hf_eos, int):
                    stop_ids.add(hf_eos)
                elif isinstance(hf_eos, (list, tuple, set)):
                    stop_ids.update(int(t) for t in hf_eos if t is not None)

            if (
                self.chat_serving is not None
                and self.chat_serving.template_manager is not None
                and self.chat_serving.template_manager.chat_template_name is not None
            ):
                template = chat_templates.get(
                    self.chat_serving.template_manager.chat_template_name
                )
                if template is not None and template.stop_token_ids is not None:
                    if isinstance(template.stop_token_ids, int):
                        stop_ids.add(template.stop_token_ids)
                    elif isinstance(template.stop_token_ids, (list, tuple, set)):
                        stop_ids.update(
                            int(t) for t in template.stop_token_ids if t is not None
                        )

        return sorted(stop_ids) if stop_ids else None

    def _tokenize_chat_request(
        self, request: TokenizeRequest
    ) -> tuple[List[int], Optional[List[str]], Optional[List[int]]]:
        if self.chat_serving is None:
            raise ValueError("Chat template tokenization requires a template manager.")

        chat_request = request.to_chat_completion_request()
        validation_error = self.chat_serving._validate_request(chat_request)
        if validation_error:
            raise ValueError(validation_error)
        if chat_request._format_add_generation_prompt is not None:
            if (
                self.chat_serving.template_manager.chat_template_name is not None
                or self.chat_serving.chat_encoding_spec is not None
            ):
                raise ValueError(
                    "add_generation_prompt is supported only with a Jinja chat template."
                )
            if (
                chat_request.continue_final_message
                and not chat_request._format_add_generation_prompt
            ):
                raise ValueError(
                    "continue_final_message needs add_generation_prompt to open the "
                    "assistant turn."
                )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self.chat_serving._process_messages(
            chat_request, is_multimodal
        )

        stop = processed_messages.stop if processed_messages.stop else None
        stop_token_ids = self._collect_template_stop_token_ids(chat_request)

        prompt_ids = processed_messages.prompt_ids
        if isinstance(prompt_ids, list) and (
            prompt_ids or not processed_messages.prompt
        ):
            return prompt_ids, stop, stop_token_ids
        if isinstance(prompt_ids, str):
            return (
                self.tokenizer_manager.tokenizer.encode(
                    prompt_ids, add_special_tokens=False
                ),
                stop,
                stop_token_ids,
            )
        if processed_messages.prompt:
            return (
                self.tokenizer_manager.tokenizer.encode(
                    processed_messages.prompt, add_special_tokens=False
                ),
                stop,
                stop_token_ids,
            )

        raise ValueError("Failed to render chat messages into token ids.")


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
