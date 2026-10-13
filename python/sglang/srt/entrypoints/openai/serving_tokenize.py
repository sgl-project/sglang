import json
import logging
from http import HTTPStatus
from typing import List, Optional, Union

from fastapi import Request
from jsonschema import Draft202012Validator, ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatMessage,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    FunctionResponse,
    ParseRequest,
    TokenizeRequest,
    TokenizeResponse,
    ToolCall,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    _incomplete_tool_call_indices,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

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

            if request.messages is not None:
                token_ids = self._tokenize_chat_request(request)
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
                tokens=tokens, count=count, max_model_len=max_model_len
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

    def _tokenize_chat_request(self, request: TokenizeRequest) -> List[int]:
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

        prompt_ids = processed_messages.prompt_ids
        if isinstance(prompt_ids, list) and (
            prompt_ids or not processed_messages.prompt
        ):
            return prompt_ids
        if isinstance(prompt_ids, str):
            return self.tokenizer_manager.tokenizer.encode(
                prompt_ids, add_special_tokens=False
            )
        if processed_messages.prompt:
            return self.tokenizer_manager.tokenizer.encode(
                processed_messages.prompt, add_special_tokens=False
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


class OpenAIServingParse(OpenAIServingBase):
    """Handler for /v1/parse requests: output token IDs to an assistant message"""

    def __init__(self, chat_serving: OpenAIServingChat):
        super().__init__(chat_serving.tokenizer_manager)
        self.chat_serving = chat_serving

    def _request_id_prefix(self) -> str:
        return "parse-"

    def _convert_to_internal_request(
        self, request: ParseRequest, raw_request: Request
    ) -> tuple[ParseRequest, ParseRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: ParseRequest,
        request: ParseRequest,
        raw_request: Request,
    ) -> ChatMessage:
        chat = self.chat_serving
        tokenizer = self.tokenizer_manager.tokenizer
        tools = request.tools or []
        parse_tools = bool(tools and chat.tool_call_parser)
        # Parsers read the delimiters that skipping special tokens would drop.
        text = tokenizer.decode(
            request.tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        reasoning = None
        if chat.reasoning_parser:
            reasoning, text = ReasoningParser(
                model_type=chat.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=chat.template_manager.force_reasoning,
                tokenizer=tokenizer,
                tool_call_parser_active=parse_tools,
            ).parse_non_stream(text)

        tool_calls = None
        if parse_tools:
            parser = FunctionCallParser(
                tools, chat.tool_call_parser, tokenizer=tokenizer
            )
            if parser.has_tool_call(text):
                parsed = parser.parse_non_stream_result(text, preserve_incomplete=True)
                if not parsed.calls or _incomplete_tool_call_indices(parser):
                    raise ValueError(
                        "The output has a malformed or incomplete tool call."
                    )
                schemas = {
                    tool.function.name: tool.function.parameters for tool in tools
                }
                for call in parsed.calls:
                    if schemas.get(call.name) is None:
                        continue
                    try:
                        Draft202012Validator(schemas[call.name]).validate(
                            json.loads(call.parameters)
                        )
                    except ValidationError as e:
                        raise ValueError(
                            f"Tool call {call.name!r} does not match its schema: "
                            f"{e.message}"
                        ) from e
                text = parsed.normal_text
                tool_calls = [
                    ToolCall(
                        index=index,
                        function=FunctionResponse(
                            name=call.name, arguments=call.parameters
                        ),
                    )
                    for index, call in enumerate(parsed.calls)
                ]

        return ChatMessage(
            role="assistant",
            content=text,
            reasoning_content=reasoning or None,
            tool_calls=tool_calls,
        )
