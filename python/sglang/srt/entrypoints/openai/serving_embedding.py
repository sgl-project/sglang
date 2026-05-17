from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jinja2
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
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
from sglang.srt.parser.conversation import generate_embedding_convs
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


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

    def _convert_to_internal_request(
        self,
        request: EmbeddingRequest,
        raw_request: Request = None,
    ) -> tuple[EmbeddingReqInput, EmbeddingRequest]:
        """Convert OpenAI embedding request to internal format"""
        prompt = request.input

        if isinstance(prompt, str):
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
                elif (
                    self.tokenizer_manager.tokenizer is not None
                    and getattr(self.tokenizer_manager.tokenizer, "chat_template", None)
                    is not None
                ):
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
            try:
                prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    [processed_msg],
                    tokenize=False,
                    add_generation_prompt=True,
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
            prompts.append(prompt)

        return prompts

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

        response = self._build_embedding_response(ret)
        return response

    def _build_embedding_response(self, ret: List[Dict[str, Any]]) -> EmbeddingResponse:
        """Build the embedding response"""
        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding_objects.append(
                EmbeddingObject(
                    embedding=ret_item["embedding"],
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
