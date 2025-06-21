from typing import Any, Dict, List, Optional, Union

from fastapi import Request

from sglang.srt.conversation import generate_embedding_convs
from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    MultimodalEmbeddingInput,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput


class OpenAIServingEmbedding(OpenAIServingBase):
    """Handler for embedding requests"""

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
                        return f"All items in input list must be strings"
                    if not item.strip():
                        return f"Input at index {i} cannot be empty or whitespace only"
            elif isinstance(first_item, int):
                # List of integers (token IDs)
                for i, item in enumerate(input):
                    if not isinstance(item, int):
                        return f"All items in input list must be integers"
                    if item < 0:
                        return f"Token ID at index {i} must be non-negative"
        return None

    def _convert_to_internal_request(
        self,
        request: EmbeddingRequest,
    ) -> tuple[EmbeddingReqInput, EmbeddingRequest]:
        """Convert OpenAI embedding request to internal format"""
        prompt = request.input

        if isinstance(prompt, str):
            # Single string input
            prompt_kwargs = {"text": prompt}
        elif isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], str):
                # List of strings - if it's a single string in a list, treat as single string
                if len(prompt) == 1:
                    prompt_kwargs = {"text": prompt[0]}
                else:
                    prompt_kwargs = {"text": prompt}
            elif len(prompt) > 0 and isinstance(prompt[0], MultimodalEmbeddingInput):
                # Handle multimodal embedding inputs
                texts = []
                images = []
                for item in prompt:
                    # Use padding for text if None - this could be improved
                    texts.append(item.text if item.text is not None else "padding")
                    images.append(item.image if item.image is not None else None)

                generate_prompts = []
                # Check if we have a chat template for multimodal embeddings
                chat_template_name = getattr(
                    self.tokenizer_manager, "chat_template_name", None
                )
                if chat_template_name is not None:
                    convs = generate_embedding_convs(texts, images, chat_template_name)
                    for conv in convs:
                        generate_prompts.append(conv.get_prompt())
                else:
                    generate_prompts = texts

                if len(generate_prompts) == 1:
                    prompt_kwargs = {
                        "text": generate_prompts[0],
                        "image_data": images[0],
                    }
                else:
                    prompt_kwargs = {
                        "text": generate_prompts,
                        "image_data": images,
                    }
            else:
                # List of integers (token IDs) or empty list
                prompt_kwargs = {"input_ids": prompt}
        else:
            # Other types (should not happen but handle gracefully)
            prompt_kwargs = {"input_ids": prompt}

        adapted_request = EmbeddingReqInput(
            **prompt_kwargs,
        )

        return adapted_request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: EmbeddingRequest,
        raw_request: Request,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
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
