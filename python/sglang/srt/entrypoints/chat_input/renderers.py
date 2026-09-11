from __future__ import annotations

import copy
import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import jinja2
import orjson

from sglang.srt.entrypoints.chat_input.schema import (
    ChatCompletionMessageGenericParam,
)
from sglang.srt.entrypoints.chat_input.types import (
    ChatInput,
    ChatModelConfig,
    RenderedPrompt,
    TextPrompt,
    TokenPrompt,
)
from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32
from sglang.srt.environ import envs
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import (
    MEDIA_URL_PART_TYPES,
    process_content_for_template_format,
)

logger = logging.getLogger(__name__)

try:
    from mistral_common.exceptions import MistralCommonException

    _MISTRAL_COMMON_ERRORS: tuple[type[BaseException], ...] = (MistralCommonException,)
except ImportError:
    _MISTRAL_COMMON_ERRORS = ()

_CHAT_TEMPLATE_CLIENT_ERRORS: tuple[type[BaseException], ...] = (
    jinja2.TemplateError,
    TypeError,
) + _MISTRAL_COMMON_ERRORS


class ThinkingMode(str, Enum):
    """Mode for message encoding - chat vs thinking/reasoning."""

    CHAT = "chat"
    THINKING = "thinking"


def normalize_tool_content(role: str, content):
    """Normalize tool message content from OpenAI array format to plain string.

    OpenAI clients may send tool content as a list of content parts
    (e.g. [{"type":"text","text":"..."}]) but most chat templates expect
    a plain string for tool messages. Only flatten when ALL items are
    pure OpenAI text parts; preserve lists containing non-text-type items
    that some templates intentionally iterate over.
    """
    if role != "tool" or not isinstance(content, list):
        return content
    parts = content
    is_openai_text_parts = all(
        (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        for p in parts
    )
    if is_openai_text_parts:
        text_parts = [p.get("text", "") if isinstance(p, dict) else p for p in parts]
        return " ".join(text_parts)
    return content


def parse_tool_call_arguments(arguments: str) -> Dict[str, Any]:
    """Parse OpenAI tool call arguments for chat templates."""
    try:
        parsed_arguments = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    return parsed_arguments


def normalize_assistant_tool_call_arguments(
    message: Dict[str, Any], *, strict: bool = True
) -> None:
    """Normalize assistant history tool call arguments in-place."""
    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return

    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    raise


KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED = "<| kimi_image_placeholder |>"


def neutralize_kimi_k3_image_placeholder(text: str) -> str:
    return text.replace(KIMI_K3_IMAGE_PLACEHOLDER, KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED)


def neutralize_kimi_k3_image_placeholder_value(value: Any) -> Any:
    if isinstance(value, str):
        return neutralize_kimi_k3_image_placeholder(value)
    if isinstance(value, list):
        return [neutralize_kimi_k3_image_placeholder_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: neutralize_kimi_k3_image_placeholder_value(item)
            for key, item in value.items()
        }
    return value


class ChatRenderer:
    def __init__(self, config: ChatModelConfig):
        self.config = config

    def render(
        self, request: ChatInput, tools: Optional[List[Dict]], require_reasoning: bool
    ) -> RenderedPrompt:
        if self.config.template_manager.chat_template_name is None:
            return self._apply_jinja_template(request, tools, self.config.is_multimodal)
        return self._apply_conversation_template(
            request, self.config.is_multimodal, require_reasoning
        )

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatInput,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatInput with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.config.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.config.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _prepare_kimi_k3_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatInput,
    ) -> tuple[List[Dict[str, Any]], int, Optional[str]]:
        image_count = 0
        for index, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("text", "input_text"):
                        parts.append(
                            {
                                "type": "text",
                                "text": neutralize_kimi_k3_image_placeholder(
                                    part["text"]
                                ),
                            }
                        )
                    elif part_type in ("image_url", "input_image"):
                        image = part.get("image_url") or {}
                        if isinstance(image, str):
                            image = {"url": image, "detail": part.get("detail")}
                        parts.append({"type": "image_url", "image_url": image})
                        image_count += 1
                message["content"] = parts
            elif isinstance(content, str):
                message["content"] = neutralize_kimi_k3_image_placeholder(content)
            elif content is None:
                message["content"] = ""

            if message.get("role") == "assistant":
                for key in ("reasoning_content", "reasoning"):
                    if key in message:
                        message[key] = neutralize_kimi_k3_image_placeholder_value(
                            message[key]
                        )
                for tool_call in message.get("tool_calls") or []:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, dict)
                        else None
                    )
                    if isinstance(function, dict) and "arguments" in function:
                        function["arguments"] = (
                            neutralize_kimi_k3_image_placeholder_value(
                                function["arguments"]
                            )
                        )

            source = request.messages[index]
            if (
                isinstance(source, ChatCompletionMessageGenericParam)
                and source.role in ("system", "developer")
                and source.tools
            ):
                message["tools"] = [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in source.tools
                ]
            if message.get("role") == "developer":
                message["role"] = "system"

        assistant_prefix = None
        if request.continue_final_message:
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )
        return messages, image_count, assistant_prefix

    def _encode_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatInput,
        thinking_mode: ThinkingMode,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[List[int]]:
        """Encode messages for custom chat_encoding_spec values.

        Returns prompt_ids if handled, None to use default encoding.
        """
        if self.config.chat_encoding_spec == "inkling":
            # Inkling: render messages -> input_ids with framing tokens + ONE placeholder per
            # media (encoding/expansion happens later in InklingMultimodalProcessor). The
            # server's tokenizer is the base tiktoken backend; wrap it so encode_special
            # supplies the framing-token overlay.
            from sglang.srt.parser.inkling_renderer import render_inkling_messages
            from sglang.srt.parser.inkling_tokenizer import (
                CONTENT_TEXT,
                MESSAGE_MODEL,
                InklingTokenizer,
            )

            inkling_tokenizer = InklingTokenizer(tokenizer=self.config.tokenizer)
            reasoning_effort = self._parse_inkling_reasoning_effort(
                request.reasoning_effort
            )
            if reasoning_effort is None:
                reasoning_effort = self.config.inkling_default_reasoning_effort
            assistant_prefix = self._pop_inkling_assistant_prefix(messages, request)
            prompt_ids = render_inkling_messages(
                messages,
                inkling_tokenizer,
                add_generation_prompt=False,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
            if assistant_prefix is not None:
                # Continue the final assistant message inside an OPEN model text
                # block: header + payload, no <|end_message|> and no
                # <|content_model_end_sampling|>, so the model resumes the turn.
                prompt_ids += [
                    inkling_tokenizer.encode_special(MESSAGE_MODEL),
                    inkling_tokenizer.encode_special(CONTENT_TEXT),
                    *inkling_tokenizer.encode_text(assistant_prefix),
                ]
            return prompt_ids
        if self.config.chat_encoding_spec == "kimi_k3":
            messages, image_count, assistant_prefix = self._prepare_kimi_k3_messages(
                messages, request
            )
            template_kwargs = dict(request.chat_template_kwargs or {})
            template_kwargs.pop("tokenize", None)
            template_kwargs.pop("return_dict", None)
            template_kwargs.pop("image_prompts", None)
            if image_count:
                template_kwargs["image_prompts"] = ["<|media_pad|>"] * image_count

            if (
                request.reasoning_effort in ("low", "high", "max")
                and "thinking_effort" not in template_kwargs
            ):
                template_kwargs["thinking_effort"] = request.reasoning_effort
            elif request.reasoning_effort not in (
                None,
                "none",
                "low",
                "high",
                "max",
            ):
                logger.warning(
                    "Kimi K3 does not support reasoning_effort=%r; using the "
                    "encoder default.",
                    request.reasoning_effort,
                )

            effective_tools = request.effective_tools()
            if (
                effective_tools
                and isinstance(request.tool_choice, str)
                and request.tool_choice in ("required", "none")
            ):
                template_kwargs.setdefault("tool_choice", request.tool_choice)
            if request.response_format is not None:
                template_kwargs.setdefault(
                    "response_format",
                    request.response_format.model_dump(
                        exclude_unset=True, by_alias=True
                    ),
                )

            request_tools = (
                [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in request.tools
                ]
                if request.tools
                else None
            )
            prompt_ids = self.config.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=request_tools,
                return_dict=False,
                **template_kwargs,
            )
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
            return prompt_ids
        return None

    @staticmethod
    def _pop_inkling_assistant_prefix(
        messages: List[Dict[str, Any]],
        request: ChatInput,
    ) -> Optional[str]:
        """Extract the trailing assistant text for ``continue_final_message``.

        Only a plain-string assistant message with no tool calls and no
        reasoning content can be continued; anything else renders as a closed
        historical turn. Mutates ``messages`` in place (callers pass a copy).
        """
        if not request.continue_final_message or not messages:
            return None
        last = messages[-1]
        if (
            last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or last.get("tool_calls")
            or last.get("reasoning_content")
        ):
            return None
        messages.pop()
        return last["content"]

    @staticmethod
    def _parse_inkling_reasoning_effort(
        value: Optional[Union[str, float]],
    ) -> Optional[float]:
        """Convert an OpenAI-style reasoning_effort to an Inkling float."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Inkling reasoning_effort must not be a boolean")
        if isinstance(value, (int, float)):
            parsed = float(value)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
                raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
            return parsed
        _EFFORT_MAP = {
            "none": 0.0,
            "minimal": 0.1,
            "low": 0.2,
            "medium": 0.7,
            "high": 0.9,
            "xhigh": 0.99,
            "max": 0.99,
        }
        if value in _EFFORT_MAP:
            return _EFFORT_MAP[value]
        try:
            parsed = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
        return parsed

    @staticmethod
    def _sort_tool_message_run(
        run: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Order a tool-message run by tool_call position.

        Templates that associate results by tool_call_id render the run in
        tool_calls order; sorting the run upfront keeps extraction order and
        placeholder order the same. Runs the template itself would refuse to
        associate (missing/duplicate/unknown ids) are left untouched, as are
        text-only runs, whose order text-only templates may rely on.
        """
        if len(run) < 2:
            return run
        call_ids = [tc.get("id") for tc in tool_calls]
        if any(call_id is None for call_id in call_ids) or len(set(call_ids)) != len(
            call_ids
        ):
            return run
        result_ids = [message.get("tool_call_id") for message in run]
        if any(result_id not in call_ids for result_id in result_ids) or len(
            set(result_ids)
        ) != len(result_ids):
            return run
        has_media = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") in MEDIA_URL_PART_TYPES
                for part in message["content"]
            )
            for message in run
        )
        if not has_media:
            return run
        position = {call_id: index for index, call_id in enumerate(call_ids)}
        return sorted(run, key=lambda message: position[message["tool_call_id"]])

    @classmethod
    def _canonicalize_tool_message_order(
        cls, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        canonical = []
        index = 0
        while index < len(messages):
            message = messages[index]
            canonical.append(message)
            index += 1
            tool_calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not tool_calls:
                continue
            run = []
            while index < len(messages) and messages[index].get("role") in (
                "tool",
                "function",
            ):
                run.append(messages[index])
                index += 1
            canonical.extend(cls._sort_tool_message_run(run, tool_calls))
        return canonical

    def _apply_jinja_template(
        self,
        request: ChatInput,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> RenderedPrompt:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = (
            self.config.template_manager.jinja_template_content_format
        )

        # Try custom encoding first (override in subclass for custom renderers)
        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_DEFAULT_THINKING.get()
        )
        thinking_mode = (
            ThinkingMode.THINKING if thinking_requested else ThinkingMode.CHAT
        )
        messages = [msg.model_dump() for msg in request.messages]
        for message in messages:
            normalize_assistant_tool_call_arguments(
                message, strict=self.config.chat_encoding_spec != "kimi_k3"
            )

        prompt_ids = self._encode_messages(
            copy.deepcopy(messages),
            request,
            thinking_mode,
            tools=tools,
        )

        if prompt_ids is not None:
            engine_prompt = (
                TextPrompt(prompt_ids)
                if isinstance(prompt_ids, str)
                else TokenPrompt(prompt_ids)
            )
            if self.config.chat_encoding_spec in ("inkling", "kimi_k3"):
                for message in request.messages:
                    msg_dict = message.model_dump()
                    if msg_dict.get("content") is None:
                        msg_dict["content"] = ""
                    process_content_for_template_format(
                        msg_dict,
                        "openai",
                        image_data,
                        video_data,
                        audio_data,
                        modalities,
                    )
        elif self.config.chat_encoding_spec is not None:
            # dsv4/dsv32 encoding path
            messages = copy.deepcopy(messages)

            # dsv4/dsv32 are text-only and consume string content; flatten
            # OpenAI parts-list content here so the encoder sees a plain string.
            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    messages[i] = process_content_for_template_format(
                        msg, "string", [], [], [], []
                    )

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.config.chat_encoding_spec == "dsv32",
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]

            # Default encoding (dsv4/dsv32)
            if self.config.chat_encoding_spec == "dsv4":
                effort_source = request.reasoning_effort
                if effort_source is None:
                    env_val = envs.SGLANG_DSV4_REASONING_EFFORT.get()
                    if env_val:
                        effort_source = env_val
                reasoning_effort_profile = self.config.dsv4_reasoning_effort_profile
                assert reasoning_effort_profile is not None
                accepted_efforts = encoding_dsv4.REASONING_EFFORT_PROFILES[
                    reasoning_effort_profile
                ]
                v4_reasoning_effort = (
                    effort_source if effort_source in accepted_efforts else None
                )
                if request.task is not None:
                    encoding_dsv4.attach_task_to_last_user_message(
                        messages, request.task
                    )
                real_input = encoding_dsv4.encode_messages(
                    messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=v4_reasoning_effort,
                    reasoning_effort_profile=reasoning_effort_profile,
                )
                prompt_ids = self.config.tokenizer.encode(real_input)
            else:
                real_input = encoding_dsv32.encode_messages(
                    messages, thinking_mode=thinking_mode
                )
                prompt_ids = self.config.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
            engine_prompt = (
                TextPrompt(prompt_ids)
                if isinstance(prompt_ids, str)
                else TokenPrompt(prompt_ids)
            )
        else:
            if self.config.template_manager.jinja_template_may_reorder_tool_results:
                messages = self._canonicalize_tool_message_order(messages)
            for msg_dict in copy.deepcopy(messages):
                if msg_dict.get("content") is None:
                    msg_dict["content"] = ""

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                processed_msg["content"] = normalize_tool_content(
                    processed_msg["role"], processed_msg.get("content")
                )

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            rc = self.config.template_manager.reasoning_config
            if rc is not None and rc.effort_kwarg is not None:
                if request.reasoning_effort == "low":
                    extra_template_kwargs.setdefault(rc.effort_kwarg, True)
                elif request.reasoning_effort in ("medium", "high", "max"):
                    logger.warning(
                        "Model '%s' supports only 'low' reasoning effort; "
                        "requested '%s' treated as default thinking",
                        self.config.model_name,
                        request.reasoning_effort,
                    )

            # Split apply_chat_template(tokenize=True) into render + encode so we
            # can skip add_special_tokens=False on tokenizers that don't auto-add
            # specials (Kimi-like, OpenAI-chat analogue of #25265). Chat
            # templates already include role/special tokens, so the encode must
            # avoid double BOS on tokenizers that would add it.
            encode_kwargs = (
                {"add_special_tokens": False}
                if self.config.tokenizer_auto_adds_specials
                else {}
            )
            try:
                rendered_prompt = self.config.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
                prompt_ids = self.config.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
            except Exception:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    rendered_prompt = self.config.tokenizer.apply_chat_template(
                        openai_compatible_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        tools=tools,
                        return_dict=False,
                        **extra_template_kwargs,
                    )
                    prompt_ids = self.config.tokenizer.encode(
                        rendered_prompt, **encode_kwargs
                    )
                except _CHAT_TEMPLATE_CLIENT_ERRORS as template_error:
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # and TypeError (e.g., tojson filter on Jinja2 Undefined variables)
                    # should be treated as client errors (400 BadRequest)
                    raise ValueError(str(template_error)) from template_error

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.config.tokenizer.decode(prompt_ids)
            engine_prompt = (
                TextPrompt(
                    prompt,
                    cached_token_ids=prompt_ids if prompt_ids or not prompt else None,
                )
                if is_multimodal
                else TokenPrompt(prompt_ids)
            )

        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return RenderedPrompt(
            prompt=engine_prompt,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
        )

    def _apply_conversation_template(
        self,
        request: ChatInput,
        is_multimodal: bool,
        require_reasoning: bool,
    ) -> RenderedPrompt:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(
            request, self.config.template_manager.chat_template_name
        )

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if require_reasoning and (
                self.config.reasoning_detector is None
                or not self.config.reasoning_detector.thinks_internally
            ):
                # Models with thinks_internally=True think without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        if not is_multimodal:
            prompt_ids = self.config.tokenizer.encode(prompt)

        return RenderedPrompt(
            prompt=TextPrompt(prompt) if is_multimodal else TokenPrompt(prompt_ids),
            template_stop=copy.copy(conv.stop_str or []),
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
        )
