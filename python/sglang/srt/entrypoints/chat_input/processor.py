from __future__ import annotations

import copy
import logging
from typing import Optional

from sglang.srt.entrypoints.chat_input.renderers import ChatRenderer
from sglang.srt.entrypoints.chat_input.schema import ToolChoice
from sglang.srt.entrypoints.chat_input.types import (
    ChatInput,
    ChatModelConfig,
    MessageRenderer,
    PreparedChat,
    RenderedPrompt,
    TokenPrompt,
)
from sglang.srt.entrypoints.chat_input.validation import (
    MediaInputError,
    validate_chat_input,
    validate_media_content,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.parser.hunyuan_reasoning import normalize_hunyuan_reasoning_effort
from sglang.srt.parser.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


class ChatInputProcessor:
    def __init__(
        self, config: ChatModelConfig, renderer: Optional[MessageRenderer] = None
    ):
        self.config = config
        self.renderer = renderer if renderer is not None else ChatRenderer(config)

    def prepare(self, chat_input: ChatInput) -> PreparedChat:
        request = self._resolve_input(chat_input)
        thinking_mode = self._get_reasoning_from_request(request)
        tools, tool_call_constraint, tool_call_stop = self._prepare_tools(
            request, thinking_mode
        )
        if request.input_ids is not None:
            result = RenderedPrompt(prompt=TokenPrompt(request.input_ids))
        else:
            result = self.renderer.render(request, tools, thinking_mode)

        return PreparedChat(
            prompt=result.prompt,
            image_data=result.image_data,
            audio_data=result.audio_data,
            video_data=result.video_data,
            modalities=result.modalities,
            stop=self._resolve_stop(request, result.template_stop, tool_call_stop),
            tool_call_constraint=tool_call_constraint,
            require_reasoning=thinking_mode,
            skip_special_tokens=request.skip_special_tokens,
            reasoning_end_token_ids=self._reasoning_end_token_ids(
                request, thinking_mode
            ),
            chat_template_kwargs=copy.deepcopy(request.chat_template_kwargs),
            reasoning_effort=request.reasoning_effort,
        )

    def _resolve_input(self, chat_input: ChatInput) -> ChatInput:
        request = chat_input.model_copy(deep=True)
        self._validate_media(request)
        error = validate_chat_input(request)
        if error:
            raise ValueError(error)
        if self.config.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.config.default_chat_template_kwargs.items():
                ctk.setdefault(k, copy.deepcopy(v))
            request.chat_template_kwargs = ctk
            effort = ctk.get("reasoning_effort")
            if effort is not None and request.reasoning_effort is None:
                request.reasoning_effort = effort

        normalize_hunyuan_reasoning_effort(
            request,
            self.config.reasoning_parser,
            self.config.template_manager.reasoning_config,
        )

        # GptOss model needs to keep special tokens for harmony parsing
        if self.config.is_gpt_oss or self.config.is_gemma4:
            request.skip_special_tokens = False

        self._patch_reasoning_skip_special_tokens(request)

        if request.effective_tools() and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _prepare_tools(self, request: ChatInput, thinking_mode: bool):
        # SGLang's ReasonerGrammarBackend owns the reasoning prefix
        # when --reasoning-parser is configured, so builtin xgrammar
        # tags must describe only the post-reasoning tool-call suffix.
        xgrammar_reasoning = thinking_mode and (self.config.reasoning_parser is None)
        tool_call_constraint = None

        tools = None
        tool_call_stop = None
        required_parsed_natively = False
        effective_tools = request.effective_tools()
        if effective_tools and request.tool_choice != "none":
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools or []
                    if item.function.name == request.tool_choice.function.name
                ] or None
            elif request.tools:
                tools = [item.model_dump() for item in request.tools]
            if self.config.tool_call_parser:
                parser = FunctionCallParser(
                    effective_tools,
                    self.config.tool_call_parser,
                    tokenizer=self.config.tokenizer,
                )
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    thinking_mode=xgrammar_reasoning,
                )
                required_parsed_natively = parser.detector.parses_required_natively()
                if self.config.chat_encoding_spec == "kimi_k3":
                    tool_call_stop = parser.detector.eot_token
            if (
                tool_call_constraint is None
                and not required_parsed_natively
                and not (
                    self.config.chat_encoding_spec == "kimi_k3"
                    and self.config.tool_call_parser == "kimi_k3"
                )
                and (
                    request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            ):
                json_schema = get_json_schema_constraint(
                    effective_tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        return tools, tool_call_constraint, tool_call_stop

    @staticmethod
    def _resolve_stop(request: ChatInput, template_stop, tool_call_stop):
        if request.input_ids is not None:
            stop = request.stop or []
        elif template_stop is None:
            stop = request.stop
        else:
            stop = copy.copy(template_stop if not request.ignore_eos else [])
            if request.stop:
                if isinstance(request.stop, str):
                    stop.append(request.stop)
                else:
                    stop.extend(request.stop)

        if tool_call_stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            elif stop is None:
                stop = []
            else:
                stop = list(stop)
            if tool_call_stop not in stop:
                stop.append(tool_call_stop)

        return stop

    def _reasoning_end_token_ids(
        self, request: ChatInput, thinking_mode: bool
    ) -> Optional[list[int]]:
        reasoning_end_token_ids = None
        if self.config.reasoning_parser == "k2_horizon" and thinking_mode:
            parser = ReasoningParser(
                model_type=self.config.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=True,
                request=request,
                tokenizer=self.config.tokenizer,
            )
            token_ids = self.config.tokenizer.encode(
                parser.detector.think_end_token,
                add_special_tokens=False,
            )
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(
                    type(token_id) is not int or token_id < 0 for token_id in token_ids
                )
            ):
                raise ValueError(
                    "The selected K2 reasoning terminator could not be encoded"
                )
            reasoning_end_token_ids = list(token_ids)
        return reasoning_end_token_ids

    def _patch_reasoning_skip_special_tokens(self, request: ChatInput) -> None:
        """Keep parser-specific reasoning markers in the decoded text.

        Some reasoning parsers rely on special-token delimiters that would be
        removed during detokenization when ``skip_special_tokens=True``.
        """
        if self.config.reasoning_parser == "apertus2509":
            request.skip_special_tokens = False
        if (
            self.config.reasoning_parser == "kimi_k3"
            or self.config.chat_encoding_spec == "kimi_k3"
        ):
            request.skip_special_tokens = False

        if (
            self.config.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False
        elif self.config.reasoning_parser == "inkling":
            request.skip_special_tokens = False
        elif self.config.reasoning_parser == "muse":
            request.skip_special_tokens = False

    def _get_reasoning_from_request(self, request: ChatInput) -> bool:
        """Determine whether reasoning mode should be enabled for this request.

        NOTE: This is predefined based on model's chat template
        """
        if not self.config.reasoning_parser:
            return False

        if self.config.reasoning_parser == "minimax-m3":
            # M3 template prefills <mm:think> for thinking_mode=enabled, so it never
            # appears in output and reasoning must be forced. Mirrors reasoning_parser.py.
            return (request.chat_template_kwargs or {}).get(
                "thinking_mode"
            ) == "enabled"

        if self.config.reasoning_parser == "hunyuan":
            config = self.config.template_manager.reasoning_config
            if config is not None and config.special_case == "hunyuan_effort":
                return request.reasoning_effort not in ("none", "no_think")
            # Hy3-preview template emits no <think> when reasoning_effort is
            # "no_think" / "none" / unset; forcing reasoning would route all
            # output into reasoning_content.
            return request.reasoning_effort not in (None, "none", "no_think")

        config = self.config.template_manager.reasoning_config
        if config is None:
            # Fallback to parser-level defaults when template toggle config
            # cannot be inferred (e.g., parser-only <think> templates).
            mode = (
                self.config.reasoning_detector.reasoning_default
                if self.config.reasoning_detector is not None
                else None
            )
            if mode is None:
                return False
            if mode == "always":
                return True
            if mode == "mistral":
                return (
                    request.reasoning_effort is not None
                    and request.reasoning_effort != "none"
                )
            if mode in ("thinking", "enable_thinking"):
                return (
                    not request.chat_template_kwargs
                    or request.chat_template_kwargs.get(mode) is not False
                )
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                toggle = mode.replace("explicit_", "")
                return (
                    request.chat_template_kwargs is not None
                    and request.chat_template_kwargs.get(toggle) is True
                )
            logger.warning(
                "Unknown reasoning_default mode '%s', defaulting to reasoning disabled",
                mode,
            )
            return False

        if config.special_case == "always":
            return True

        if config.special_case == "mistral":
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )

        if config.toggle_param is None or config.default_enabled is None:
            return False

        if config.default_enabled:
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get(config.toggle_param) is not False
            )
        return (
            request.chat_template_kwargs is not None
            and request.chat_template_kwargs.get(config.toggle_param) is True
        )

    def _validate_media(self, request: ChatInput) -> None:
        error = validate_media_content(request, self.config.is_multimodal)
        if error:
            raise MediaInputError(error)
