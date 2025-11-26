from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import TypeAdapter

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_hidden_states_from_ret,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser

from .utils import split_text_and_images_messages

try:
    from pydantic import TypeAdapter as _PydanticTypeAdapter  # Pydantic v2

    _PYDANTIC_V2 = True
except Exception:
    _PydanticTypeAdapter = None
    _PYDANTIC_V2 = False
try:
    from pydantic.tools import parse_obj_as as _parse_obj_as
except Exception:
    _parse_obj_as = None

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _clean_text(s: str) -> str:
    # 去掉开头的控制字符、BOM、零宽字符；不改动正常英文/中文内容
    import re

    return re.sub(r"^[\u0000-\u001F\u007F\u200B\uFEFF]+", "", (s or "")).strip()


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if self.default_sampling_params:
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
        logger.warning(
            "[BOOT] model=%s is_multimodal=%s chat_template=%s",
            getattr(self.tokenizer_manager.tokenizer, "name_or_path", None),
            getattr(
                getattr(self.tokenizer_manager, "model_config", None),
                "is_multimodal",
                None,
            ),
            getattr(self.template_manager, "chat_template_name", None),
        )

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _concat_text_from_ret(self, rets: list) -> str:
        """从 generate_request 的返回项中拼出文本；优先用 'text'，否则解码 token ids。"""
        out = []
        tok = self.tokenizer_manager.tokenizer
        for item in rets or []:
            t = item.get("text") if isinstance(item, dict) else None
            if isinstance(t, str) and t:
                out.append(t)
                continue
            ids = None
            if isinstance(item, dict):
                for k in ("output_ids", "token_ids", "output_token_ids"):
                    v = item.get(k)
                    if isinstance(v, (list, tuple)) and v:
                        ids = v
                        break
            if ids and tok is not None:
                try:
                    out.append(tok.decode(ids))
                except Exception:
                    pass
        return "".join(out).strip()

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not request.tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in request.tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        # Validate tool definitions
        for i, tool in enumerate(request.tools or []):
            if tool.function.parameters is None:
                continue
            try:
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ):
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        # 兼容 chat_template_kwargs 中的 reasoning_effort 透传
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)

        # ===== 采样参数：在生成 sampling_params 后做“防复读+限长”的缺省填充 =====
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # 文本/多模态统一走 text；如需预编码，用 input_ids（不要使用自造的 prompt_ids）
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        if lora_path:
            first_adapter = (
                lora_path
                if isinstance(lora_path, str)
                else next((a for a in lora_path if a), None)
            )
            if first_adapter:
                self._validate_lora_enabled(first_adapter)

        imgs = processed_messages.image_data
        mm = ["image"] if imgs else []
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            image_data=imgs,
            modalities=mm,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            return_hidden_states=request.return_hidden_states,
            rid=request.rid,
            extra_key=self._compute_extra_key(request),
            priority=request.priority,
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
        )
        imgs = processed_messages.image_data

        def _count_markers(s: str):
            return {
                "vision_start": s.count("<|vision_start|>"),
                "vision_end": s.count("<|vision_end|>"),
                "image_pad": s.count("<|image_pad|>"),
            }

        markers = _count_markers(processed_messages.prompt if is_multimodal else "")
        img_cnt = (
            0
            if not processed_messages.image_data
            else (
                sum(
                    len(x) if isinstance(x, list) else 1
                    for x in processed_messages.image_data
                )
                if isinstance(processed_messages.image_data, list)
                else 1
            )
        )

        logger.warning(
            "[MM][CHECK] prompt markers=%s, img_cnt=%d, modalities=%s",
            markers,
            img_cnt,
            processed_messages.modalities,
        )

        def _shape(v):
            if v is None:
                return ["None"]
            if isinstance(v, list):
                return ["list", len(v)] + (_shape(v[0]) if v else [])
            if isinstance(v, dict):
                return ["dict", sorted(list(v.keys()))[:5]]  # 打头几个 key
            return [type(v).__name__]

        try:
            # 统计数量 & 第一个条目的关键字段
            img_cnt = 0
            first_item = None
            if isinstance(imgs, list):
                for it in imgs:
                    if isinstance(it, list):
                        img_cnt += len(it)
                        if not first_item and len(it):
                            first_item = it[0]
                    else:
                        img_cnt += 1
                        if not first_item:
                            first_item = it
            elif imgs:
                img_cnt = 1
                first_item = imgs

            logger.info(
                "[MM] summary: is_mm=%s img_cnt=%d modalities=%s shape=%s first_item_keys=%s",
                is_multimodal,
                img_cnt,
                processed_messages.modalities,
                _shape(imgs),
                (
                    list(first_item.keys())
                    if isinstance(first_item, dict)
                    else type(first_item).__name__
                ),
            )
        except Exception:
            logger.exception("[MM] image_data summary failed")

        # --- 强校验与日志 ---
        try:
            img_cnt = 0
            imgs = processed_messages.image_data
            if isinstance(imgs, list):
                for it in imgs:
                    img_cnt += len(it) if isinstance(it, list) else 1
            elif imgs:
                img_cnt = 1
            logger.info(
                "[MM] _convert_to_internal_request: is_mm=%s, img_cnt=%d, modalities=%s",
                is_multimodal,
                img_cnt,
                processed_messages.modalities,
            )
        except Exception:
            pass

        if is_multimodal:
            imgs = processed_messages.image_data
            img_cnt = (
                0
                if not imgs
                else (
                    sum(len(x) if isinstance(x, list) else 1 for x in imgs)
                    if isinstance(imgs, list)
                    else 1
                )
            )
            logger.warning(
                "[MM] _convert_to_internal_request: is_mm=%s img_cnt=%d modalities=%s",
                is_multimodal,
                img_cnt,
                processed_messages.modalities,
            )
            if img_cnt == 0:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400, detail="multimodal request without image_data"
                )

        logger.warning(
            "[MM][FINAL] is_mm=%s, img_cnt=%d, modalities=%s, text_len=%d",
            is_multimodal,
            (
                0
                if not adapted_request.image_data
                else (
                    sum(
                        len(x) if isinstance(x, list) else 1
                        for x in adapted_request.image_data
                    )
                    if isinstance(adapted_request.image_data, list)
                    else 1
                )
            ),
            adapted_request.modalities,
            len(adapted_request.text or ""),
        )
        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        # GptOss model needs to keep special tokens for harmony parsing
        if is_gpt_oss:
            request.skip_special_tokens = False

        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.function.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.function.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice
                )
            # Handle JSON schema constraint directly for required or named tool choice
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                json_schema = get_json_schema_constraint(
                    request.tools, request.tool_choice
                )
                tool_call_constraint = ("json_schema", json_schema)

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        logger.warning(
            "[MM] after _apply_conversation_template: is_multimodal=%s modalities=%s img_cnt=%s",
            is_multimodal,
            result.modalities,
            (
                0
                if not result.image_data
                else (
                    sum(len(x) if isinstance(x, list) else 1 for x in result.image_data)
                    if isinstance(result.image_data, list)
                    else 1
                )
            ),
        )
        # 再打 prompt 头部（排查占位符是否在 prompt 里）
        logger.debug("[PROMPT.head]%s", repr(result.prompt[:400]))
        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        for message in request.messages:
            if message.content is None:
                message.content = ""
            msg_dict = message.model_dump()

            # 用官方 util 解析内容，并侧写 image/audio/modalities
            processed_msg = process_content_for_template_format(
                msg_dict,
                template_content_format,
                image_data,
                video_data,
                audio_data,
                modalities,
            )

            # 工具调用参数转成 dict（保持原逻辑）
            if (
                processed_msg.get("role") == "assistant"
                and "tool_calls" in processed_msg
                and isinstance(processed_msg["tool_calls"], list)
            ):
                for item in processed_msg["tool_calls"]:
                    fn = item.get("function")
                    if fn and isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = orjson.loads(fn["arguments"])
                        except Exception:
                            pass

            openai_compatible_messages.append(processed_msg)

        # assistant 前缀（续写）保持原逻辑
        assistant_prefix = None
        if (
            openai_compatible_messages
            and openai_compatible_messages[-1]["role"] == "assistant"
            and request.continue_final_message
        ):
            assistant_prefix = openai_compatible_messages[-1]["content"]
            openai_compatible_messages = openai_compatible_messages[:-1]

        # 交给 tokenizer.apply_chat_template 处理占位符与工具格式
        try:
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                reasoning_effort=request.reasoning_effort,
                **(request.chat_template_kwargs or {}),
            )
        except Exception:
            tools = (
                [t if "function" in t else {"function": t} for t in (tools or [])]
                if tools
                else None
            )
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                reasoning_effort=request.reasoning_effort,
                **(request.chat_template_kwargs or {}),
            )

        if assistant_prefix:
            encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
            if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
                encoded = encoded[1:]
            prompt_ids += encoded

        if is_multimodal:
            prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=(image_data or None),
            video_data=(video_data or None),
            audio_data=(audio_data or None),
            modalities=(modalities or []),
            stop=stop,
        )

    def _apply_conversation_template(self, request, is_multimodal):
        import json
        import uuid

        req_id = uuid.uuid4().hex[:8]

        # 1) 规范化：把同条 message 内部的 text/image 拆成 纯文本 + 纯图片
        normalized = split_text_and_images_messages(request.messages or [])
        try:
            logger.debug(
                "[%s] step1 normalized=%s",
                req_id,
                json.dumps(normalized, ensure_ascii=False)[:2000],
            )
        except Exception:
            logger.debug("[%s] step1 normalized=<non-json>", req_id)

        # 2) 反序列化为 Pydantic（v2 优先，v1 兜底）
        try:
            adapter = TypeAdapter(List[ChatCompletionMessageParam])
            pyd_msgs = adapter.validate_python(normalized)
        except Exception:
            from pydantic.tools import parse_obj_as

            pyd_msgs = parse_obj_as(List[ChatCompletionMessageParam], normalized)  # type: ignore

        # 3) 若缺 system，补充英文 system 以稳定输出为英文
        if not any(
            (
                getattr(m, "role", None) == "system"
                and isinstance(getattr(m, "content", ""), str)
                and getattr(m, "content").strip()
            )
            for m in pyd_msgs
        ):
            pyd_msgs = [
                ChatCompletionMessageParam(
                    role="system",
                    content="You are a helpful vision-language assistant. Always answer in concise, natural English.",
                )
            ] + list(pyd_msgs)

        # 4) 准备 conv 输入（❗不再改 skip_special_tokens，保持默认即可）

        # 4.1) 模板选择（优先 auto；Qwen 家族兜底）
        if is_multimodal:
            tok = getattr(self.tokenizer_manager, "tokenizer", None)
            name = (getattr(tok, "name_or_path", "") or "").lower() if tok else ""
            if not getattr(self.template_manager, "chat_template_name", None):
                self.template_manager.chat_template_name = "auto"
            if "qwen3-omni" in name:
                self.template_manager.chat_template_name = "qwen3_omni"
            elif "qwen2.5-vl" in name or "qwen2-vl" in name:
                self.template_manager.chat_template_name = "qwen2_vl"

        try:
            request_for_conv = request.model_copy(update={"messages": pyd_msgs})
        except Exception:
            request_for_conv = request.copy(update={"messages": pyd_msgs})

        # 5) 生成 conv
        conv = generate_chat_conv(
            request_for_conv, self.template_manager.chat_template_name
        )
        if conv.image_data:
            conv.modalities = ["image"]
        else:
            conv.modalities = []

        # 6) 获取 prompt + 收尾裁剪
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            prompt = conv.get_prompt()

            if isinstance(conv.stop_str, list):
                for s in conv.stop_str:
                    if prompt.endswith(s):
                        prompt = prompt[: -len(s)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if getattr(conv, "sep", None) and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_enable_thinking_from_request(request):
                prompt += "<think>"

        # ——调试日志：模板、模态、图片数——
        try:
            img_cnt = 0
            if isinstance(conv.image_data, list):
                for it in conv.image_data:
                    img_cnt += len(it) if isinstance(it, list) else 1
            logger.debug(
                "[%s] tpl=%s modalities=%s img_cnt=%d",
                req_id,
                self.template_manager.chat_template_name,
                conv.modalities,
                img_cnt,
            )
            logger.debug(
                "[%s] prompt.head=%s", req_id, prompt[:500].replace("\n", "\\n")
            )
        except Exception:
            logger.debug("[%s] prompt log failed", req_id)

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        # 恢复 stop 的合并逻辑（尊重 ignore_eos 与 request.stop）
        stop = (
            list(conv.stop_str)
            if isinstance(conv.stop_str, list)
            else (
                []
                if request.ignore_eos
                else ([conv.stop_str] if isinstance(conv.stop_str, str) else [])
            )
        )
        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)
        # 8) 文本模型可先编码；多模态不要预编码 ids
        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)
        else:
            prompt_ids = []  # 多模态走 text + image_data 路径

        # CRITICAL FIX: Ensure modality is always 'image' for vision inputs.
        # The test passes 'multi-images' which causes the backend to skip embedding generation.
        if image_data:
            flat_len = (
                sum(len(x) if isinstance(x, list) else 1 for x in image_data)
                if isinstance(image_data, list)
                else 1
            )
            modalities = ["image"] * flat_len

        def _count_markers(s: str):
            return {
                "vision_start": s.count("<|vision_start|>"),
                "vision_end": s.count("<|vision_end|>"),
                "image_pad": s.count("<|image_pad|>"),
                # 若模板用别的占位符，这里继续加
            }

        markers = _count_markers(prompt)
        try:
            img_cnt = 0
            if isinstance(image_data, list):
                for it in image_data:
                    img_cnt += len(it) if isinstance(it, list) else 1
            elif image_data:
                img_cnt = 1
            logger.warning(
                "[MM][CHECK] prompt markers=%s, img_cnt=%d, modalities=%s",
                markers,
                img_cnt,
                modalities,
            )
            logger.debug("[MM][PROMPT.preview]%s", repr(prompt[:800]))
        except Exception:
            logger.exception("[MM] marker check failed")

        image_data = conv.image_data if conv.image_data else None

        if image_data:
            n = (
                sum(len(x) if isinstance(x, list) else 1 for x in image_data)
                if isinstance(image_data, list)
                else 1
            )
            modalities = ["image"] * n  # ← 关键：对齐图片张数，全部写成 'image'
        else:
            modalities = []

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=conv.modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat completion request"""
        return StreamingResponse(
            self._generate_chat_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                completion_tokens[index] = content["meta_info"]["completion_tokens"]
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    choice_logprobs = self._process_streaming_logprobs(
                        content, n_prev_tokens.get(index, 0)
                    )
                    n_prev_tokens[index] = len(
                        content["meta_info"]["output_token_logprobs"]
                    )

                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta

                # Handle reasoning content
                if self.reasoning_parser and request.separate_reasoning:
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # Add usage stats if continuous_usage_stats is enabled
                        if (
                            request.stream_options
                            and request.stream_options.continuous_usage_stats
                        ):
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle tool calls
                if (
                    request.tool_choice != "none"
                    and request.tools
                    and self.tool_call_parser
                ):
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                    ):
                        if chunk:
                            yield chunk

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type is not None and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk

                else:
                    # Regular content
                    if delta:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # Add usage stats if continuous_usage_stats is enabled
                        if (
                            request.stream_options
                            and request.stream_options.continuous_usage_stats
                        ):
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if request.stream_options and request.stream_options.include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        # 1) 先把 rets 合成为字符串（无论 text 还是 ids 都能解码出来）
        content = self._concat_text_from_ret(ret)

        content = _clean_text(content)

        # 3) 回写到 ret[0]['text']，确保下游 _build_chat_response 拿到 str
        if not ret:
            ret = [{}]
        if not isinstance(ret[0], dict):
            ret[0] = {"text": content or ""}
        else:
            ret[0]["text"] = content or ""

        # 4) 原有构建逻辑
        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )
        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        choices = []

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_enable_thinking_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if (
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
            ):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - Non-streaming (use_token_index=True): uses token_idx for full data
                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        if self.tool_call_parser != "kimi_k2":
            # A simple uuid is sufficient for all models except for Kimi-K2.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        else:
            # Align with Kimi-K2 format: functions.{name}:{index}
            # Kimi-K2 allows multiple tool_calls in one message; SGLang sets call_item.tool_index to the *local* position inside that message.
            # Therefore, the index must be corrected by using `history_tool_calls_cnt + call_item.tool_index` to ensure globally unique and properly ordered.
            tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt+call_item.tool_index}"
            logger.debug(
                f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
            )
            return tool_call_id

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response"""

        # Handle required or named tool choice
        if tool_choice == "required" or (
            isinstance(tool_choice, ToolChoice) and tool_choice.type == "function"
        ):
            # Set finish reason to tool_calls since we're processing tool calls
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                # For required tool choice, we expect a JSON array of tool calls
                tool_call_data = orjson.loads(text)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    # Create a ToolCallItem from the JSON data
                    call_info = ToolCallItem(
                        tool_index=i,  # Use the loop index as tool_index
                        name=tool["name"],
                        parameters=json.dumps(tool["parameters"], ensure_ascii=False),
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=json.dumps(
                                    tool["parameters"], ensure_ascii=False
                                ),
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except json.JSONDecodeError as e:
                logger.error(f"Tool call parsing error: {e}")
                return ToolCallProcessingResult(None, text, finish_reason)

        # Use parser since output is not constrained by JSON schema
        parser = FunctionCallParser(tools, self.tool_call_parser)
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=getattr(call_info, "tool_index", None),
                            function=FunctionResponse(
                                name=call_info.name, arguments=call_info.parameters
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, text, finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                # Return error but don't fail the whole request
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self, content: Dict[str, Any], n_prev_token: int
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_enable_thinking_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
        return idx

    def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> bool:
        """Extracts the 'enable_thinking' flag from request chat_template_kwargs.

        NOTE: This parameter is only useful for models that support enable_thinking
        flag, such as Qwen3.

        Args:
            request_obj: The request object (or an item from a list of requests).
        Returns:
            The boolean value of 'enable_thinking' if found, otherwise False.
        """
        if hasattr(request, "chat_template_kwargs") and request.chat_template_kwargs:
            # For Qwen3 models, `enable_thinking` is supported.
            if self.reasoning_parser in ["qwen3", "glm45"]:
                return request.chat_template_kwargs.get("enable_thinking", False)
            # For DeepSeek-V3.1 models, `thinking` is supported.
            elif self.reasoning_parser in ["deepseek-v3"]:
                return request.chat_template_kwargs.get("thinking", False)
            else:
                return False
        return False

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
    ):
        """Process tool calls in streaming response"""
        if index not in parser_dict:
            # Use JSON detector directly for required or named tool choice
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=request.tools,
                    tool_call_parser=self.tool_call_parser,
                )

        parser = parser_dict[index]

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, request.tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)

        # Yield normal text
        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if request.stream_options and request.stream_options.continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if request.stream_options and request.stream_options.continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
