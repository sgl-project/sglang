from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jinja2
import msgspec
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingChatMessage,
    EmbeddingMessages,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    MultimodalEmbeddingInput,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.utils import convert_embeds_to_tensors
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.parser.conversation import (
    generate_chat_conv,
    generate_embedding_convs,
)
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

_MEDIA_PART_TYPES = frozenset({"image_url", "video_url", "audio_url"})
_MEDIA_PART_MODALITY = {
    "image_url": "image",
    "video_url": "video",
    "audio_url": "audio",
}
_EMBEDDING_PART_TYPES = _MEDIA_PART_TYPES | {"text"}
# The roles generate_chat_conv renders; it rejects any other role.
_CONVERSATION_TEMPLATE_ROLES = frozenset({"system", "user", "assistant"})


class _RenderedConversation(msgspec.Struct, frozen=True, kw_only=True):
    prompt: str
    # Media in content-part order, matching the placeholders in `prompt`.
    image_data: List[Any]
    video_data: List[Any]
    audio_data: List[Any]


def _split_conversations(
    messages: EmbeddingMessages,
) -> tuple[bool, List[List[EmbeddingChatMessage]]]:
    batched = len(messages) > 0 and isinstance(messages[0], list)
    return batched, (messages if batched else [messages])


def _has_media(conversation: List[EmbeddingChatMessage]) -> bool:
    return any(
        part.type in _MEDIA_PART_TYPES
        for message in conversation
        if isinstance(message.content, list)
        for part in message.content
    )


def _render_with_conversation_template(
    conversation: List[EmbeddingChatMessage], *, template_name: str
) -> _RenderedConversation:
    for message in conversation:
        if message.role not in _CONVERSATION_TEMPLATE_ROLES:
            raise ValueError(
                f"role {message.role!r} is not supported by the "
                f"{template_name!r} conversation template"
            )
    chat_request = ChatCompletionRequest(
        messages=[message.model_dump(exclude_none=True) for message in conversation]
    )
    conv = generate_chat_conv(chat_request, template_name)
    return _RenderedConversation(
        prompt=conv.get_prompt(),
        image_data=conv.image_data,
        video_data=conv.video_data,
        audio_data=conv.audio_data,
    )


class OpenAIServingEmbedding(OpenAIServingBase):
    """Handler for v1/embeddings requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "embd-"

    def _validate_request(self, request: EmbeddingRequest) -> Optional[str]:
        """Validate that the input is not empty or whitespace only."""
        if request.encoding_format not in ("float", "base64"):
            return (
                "encoding_format must be either 'float' or 'base64', "
                f"got {request.encoding_format!r}"
            )

        if request.messages is not None:
            return self._validate_messages(request.messages)

        if not (input := request.input):
            return "Input cannot be empty"

        # Handle single string
        if isinstance(input, str):
            if not input.strip():
                return "Input cannot be empty or whitespace only"
            return None

        # Handle list inputs
        if isinstance(input, list):
            if len(input) == 0:
                return "Input cannot be empty"

            # Check first element to determine type
            first_item = input[0]

            if isinstance(first_item, str):
                # List of strings
                for i, item in enumerate(input):
                    if not isinstance(item, str):
                        return "All items in input list must be strings"
                    if not item.strip():
                        return f"Input at index {i} cannot be empty or whitespace only"
            elif isinstance(first_item, int):
                # List of integers (token IDs)
                for i, item in enumerate(input):
                    if not isinstance(item, int):
                        return "All items in input list must be integers"
                    if item < 0:
                        return f"Token ID at index {i} must be non-negative"
        return None

    def _validate_messages(self, messages: EmbeddingMessages) -> Optional[str]:
        _, conversations = _split_conversations(messages)
        for conversation in conversations:
            if not conversation:
                return "Each conversation in messages needs at least one message"
            for message in conversation:
                if not isinstance(message.content, list):
                    continue
                for part in message.content:
                    if part.type not in _EMBEDDING_PART_TYPES:
                        return (
                            f"Content part type {part.type!r} is not supported "
                            "for embeddings"
                        )
                    if part.type not in _MEDIA_PART_TYPES:
                        continue
                    model_config = self.tokenizer_manager.model_config
                    if not model_config.is_multimodal:
                        return (
                            "Model only supports text input; "
                            f"received unsupported content type {part.type!r}."
                        )
                    # The model's processor would drop media it has no encoder for.
                    supported = (
                        model_config.is_audio_understandable_model
                        if part.type == "audio_url"
                        # Video goes through the vision tower.
                        else model_config.is_image_understandable_model
                    )
                    if not supported:
                        return (
                            f"Model does not support "
                            f"{_MEDIA_PART_MODALITY[part.type]} input; "
                            f"received unsupported content type {part.type!r}."
                        )
        return None

    def _convert_to_internal_request(
        self,
        request: EmbeddingRequest,
        raw_request: Request = None,
    ) -> tuple[EmbeddingReqInput, EmbeddingRequest]:
        """Convert OpenAI embedding request to internal format"""
        prompt = request.input

        if request.messages is not None:
            prompt_kwargs = self._chat_prompt_kwargs(request)
        elif isinstance(prompt, str):
            # Single string input
            prompt_kwargs = {"text": prompt}
        elif isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], str):
                prompt_kwargs = {"text": prompt}
            elif len(prompt) > 0 and isinstance(prompt[0], MultimodalEmbeddingInput):
                # Handle multimodal embedding inputs
                texts = []
                images = []
                videos = []
                for item in prompt:
                    texts.append(item.text)
                    images.append(item.image if item.image is not None else None)
                    videos.append(item.video if item.video is not None else None)

                # Precedence: a SGLang-registered conversation template wins
                # over the tokenizer's own HF Jinja template when both exist.
                generate_prompts = []
                if self.template_manager.chat_template_name is not None:
                    convs = generate_embedding_convs(
                        texts, images, videos, self.template_manager.chat_template_name
                    )
                    for conv in convs:
                        generate_prompts.append(conv.get_prompt())
                elif self._has_jinja_chat_template():
                    generate_prompts = self._apply_jinja_template_to_embedding_inputs(
                        texts, images, videos
                    )
                else:
                    generate_prompts = [
                        text if text is not None else "padding" for text in texts
                    ]

                if len(generate_prompts) == 1:
                    prompt_kwargs = {
                        "text": generate_prompts[0],
                        "image_data": images[0],
                        "video_data": videos[0],
                    }
                else:
                    prompt_kwargs = {
                        "text": generate_prompts,
                        "image_data": images,
                        "video_data": videos,
                    }
            else:
                # List of integers (token IDs) or empty list
                prompt_kwargs = {"input_ids": prompt}
        else:
            # Other types (should not happen but handle gracefully)
            prompt_kwargs = {"input_ids": prompt}

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)

        # Validate pairing: both or neither must be provided
        if (
            request.embed_overrides is not None
            and request.embed_override_token_id is None
        ):
            raise ValueError(
                "embed_override_token_id is required when embed_overrides is provided"
            )
        if (
            request.embed_override_token_id is not None
            and request.embed_overrides is None
        ):
            raise ValueError(
                "embed_override_token_id requires embed_overrides to be provided"
            )

        # Convert float lists to tensors; position resolution is deferred
        # to the tokenizer manager (after tokenization for text inputs).
        embed_overrides = convert_embeds_to_tensors(request.embed_overrides)

        adapted_request = EmbeddingReqInput(
            **prompt_kwargs,
            rid=request.rid,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            dimensions=request.dimensions,
            lora_path=lora_path,
            embed_override_token_id=request.embed_override_token_id,
            embed_overrides=embed_overrides,
        )

        return adapted_request, request

    def _apply_jinja_template_to_embedding_inputs(
        self,
        texts: List[Optional[str]],
        images: List[Optional[str]],
        videos: List[Optional[str]],
    ) -> List[str]:
        """Render each multimodal embedding input through the tokenizer's Jinja chat template.

        Image/video bytes are threaded to the engine separately via
        ``EmbeddingReqInput.image_data``/``video_data``; this method only produces
        the prompt string. ``text=None`` emits no text chunk (no ``"padding"``
        literal). Jinja failures are re-raised as ``ValueError`` so the caller
        returns HTTP 400 instead of 500.
        """
        prompts: List[str] = []
        template_content_format = self.template_manager.jinja_template_content_format

        for text, image, video in zip(texts, images, videos):
            content_parts = []
            if image is not None:
                content_parts.append({"type": "image_url", "image_url": {"url": image}})
            if video is not None:
                content_parts.append({"type": "video_url", "video_url": {"url": video}})
            if text is not None:
                content_parts.append({"type": "text", "text": text})

            msg_dict = {
                "role": "user",
                "content": content_parts if content_parts else "",
            }
            # Empty list args: this helper is only used to normalize the content
            # shape (e.g. image_url -> image); real payloads ride on the outer
            # images/videos lists, not EmbeddingReqInput fields derived here.
            processed_msg = process_content_for_template_format(
                msg_dict,
                template_content_format,
                image_data=[],
                video_data=[],
                audio_data=[],
                modalities=[],
            )
            prompts.append(
                self._apply_chat_template(
                    [processed_msg], add_generation_prompt=True, template_kwargs={}
                )
            )

        return prompts

    def _has_jinja_chat_template(self) -> bool:
        return (
            self.tokenizer_manager.tokenizer is not None
            and getattr(self.tokenizer_manager.tokenizer, "chat_template", None)
            is not None
        )

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        template_kwargs: Dict[str, Any],
    ) -> str:
        """Render `messages` with the tokenizer's Jinja chat template.

        Jinja failures are re-raised as ``ValueError`` so the caller returns
        HTTP 400 instead of 500.
        """
        # Request options win over chat_template_kwargs with the same name, as in vLLM.
        render_kwargs = {
            **template_kwargs,
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                messages, **render_kwargs
            )
        except jinja2.TemplateError as template_error:
            location = getattr(template_error, "lineno", None)
            name = getattr(template_error, "name", None)
            suffix = ""
            if name or location:
                suffix = f" (template={name or '<unknown>'}, line={location})"
            raise ValueError(f"{template_error}{suffix}") from template_error
        except (TypeError, KeyError, AttributeError) as template_error:
            raise ValueError(
                f"Failed to render chat template for embedding input: {template_error}"
            ) from template_error

    def _chat_prompt_kwargs(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Render each conversation in `request.messages` into one engine prompt."""
        batched, conversations = _split_conversations(request.messages)
        rendered = [
            self._render_conversation(
                conversation,
                add_generation_prompt=request.add_generation_prompt,
                template_kwargs=request.chat_template_kwargs or {},
            )
            for conversation in conversations
        ]
        prompts = [conversation.prompt for conversation in rendered]

        if not self.tokenizer_manager.model_config.is_multimodal:
            # Chat templates write their own special tokens, so none are added here.
            input_ids = [
                self.tokenizer_manager.tokenizer.encode(
                    prompt, add_special_tokens=False
                )
                for prompt in prompts
            ]
            return {"input_ids": input_ids if batched else input_ids[0]}

        # The MM processor tokenizes the text so it can expand the placeholders.
        media = {
            "image_data": [conversation.image_data for conversation in rendered],
            "video_data": [conversation.video_data for conversation in rendered],
            "audio_data": [conversation.audio_data for conversation in rendered],
        }
        prompt_kwargs: Dict[str, Any] = {"text": prompts if batched else prompts[0]}
        for key, per_conversation in media.items():
            if not any(per_conversation):
                continue
            if batched:
                prompt_kwargs[key] = [items or None for items in per_conversation]
            else:
                prompt_kwargs[key] = per_conversation[0]
        return prompt_kwargs

    def _render_conversation(
        self,
        conversation: List[EmbeddingChatMessage],
        *,
        add_generation_prompt: bool,
        template_kwargs: Dict[str, Any],
    ) -> _RenderedConversation:
        # Same precedence as the multimodal `input` path above.
        template_name = self.template_manager.chat_template_name
        if template_name is not None:
            return _render_with_conversation_template(
                conversation, template_name=template_name
            )
        if not self._has_jinja_chat_template():
            raise ValueError(
                "messages input needs a chat template, and this model has none; "
                "start the server with --chat-template"
            )
        return self._render_with_jinja_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=template_kwargs,
        )

    def _render_with_jinja_template(
        self,
        conversation: List[EmbeddingChatMessage],
        *,
        add_generation_prompt: bool,
        template_kwargs: Dict[str, Any],
    ) -> _RenderedConversation:
        content_format = self.template_manager.jinja_template_content_format
        # A "string" template keeps only the text parts, which would drop the media.
        if content_format == "string" and _has_media(conversation):
            raise ValueError(
                "This model's chat template renders text only, so messages with "
                "image, video or audio content cannot be embedded"
            )
        image_data: List[Any] = []
        video_data: List[Any] = []
        audio_data: List[Any] = []
        messages = []
        for message in conversation:
            msg_dict = message.model_dump()
            if msg_dict["content"] is None:
                msg_dict["content"] = ""
            messages.append(
                process_content_for_template_format(
                    msg_dict,
                    content_format,
                    image_data=image_data,
                    video_data=video_data,
                    audio_data=audio_data,
                    modalities=[],
                )
            )
        return _RenderedConversation(
            prompt=self._apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                template_kwargs=template_kwargs,
            ),
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: EmbeddingRequest,
        raw_request: Request,
    ) -> Union[EmbeddingResponse, ErrorResponse, ORJSONResponse]:
        """Handle the embedding request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_embedding_response(ret, request.encoding_format)
        return response

    @staticmethod
    def _serialize_embedding(
        embedding: List[float], encoding_format: str
    ) -> Union[List[float], str]:
        if encoding_format == "float":
            return embedding

        # OpenAI-compatible base64 embeddings contain contiguous little-endian
        # float32 values. Explicit packing prevents the wire format from
        # depending on the host's native byte order or Python float width.
        encoded = struct.pack(f"<{len(embedding)}f", *embedding)
        return base64.b64encode(encoded).decode("ascii")

    def _build_embedding_response(
        self, ret: List[Dict[str, Any]], encoding_format: str = "float"
    ) -> EmbeddingResponse:
        """Build the embedding response"""
        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding_objects.append(
                EmbeddingObject(
                    embedding=self._serialize_embedding(
                        ret_item["embedding"], encoding_format
                    ),
                    index=idx,
                )
            )
            # Handle missing prompt_tokens gracefully
            meta_info = ret_item.get("meta_info", {})
            prompt_tokens += meta_info.get("prompt_tokens", 0)

        return EmbeddingResponse(
            data=embedding_objects,
            model=self.tokenizer_manager.model_path,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
