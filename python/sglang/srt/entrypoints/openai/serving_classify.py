from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIServingClassify(OpenAIServingBase):
    """Handler for v1/classify requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.id2label = self._get_id2label_mapping()
        self.model_name = (
            self.tokenizer_manager.served_model_name
            if self.tokenizer_manager.served_model_name
            else self.tokenizer_manager.server_args.model_path
        )
        if not self.id2label:
            raise ValueError("id2label mapping is missing")

    def _request_id_prefix(self) -> str:
        return "classify-"

    def _convert_to_internal_request(
        self,
        request: ClassifyRequest,
        raw_request: Request = None,
    ) -> tuple[EmbeddingReqInput, ClassifyRequest]:
        """Convert OpenAI embedding request to internal format"""
        prompt = request.input

        if isinstance(prompt, str):
            # Single string input
            prompt_kwargs = {"text": prompt}
        elif isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], str):
                prompt_kwargs = {"text": prompt}
            else:
                # List of integers (token IDs) or empty list
                prompt_kwargs = {"input_ids": prompt}
        else:
            # Other types (should not happen but handle gracefully)
            prompt_kwargs = {"input_ids": prompt}

        adapted_request = EmbeddingReqInput(
            **prompt_kwargs,
            rid=request.rid,
            priority=request.priority,
        )

        return adapted_request, request

    def _validate_request(self, request: ClassifyRequest) -> Optional[str]:
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

    def _get_id2label_mapping(self) -> Optional[Dict[int, str]]:
        """Get id2label mapping from model config."""
        try:
            hf_config = self.tokenizer_manager.model_config.hf_config
            # Check for id2label in hf_config
            if hf_config.id2label:
                return hf_config.id2label
            # Check for num_labels and create default mapping if needed
            if hasattr(hf_config, "num_labels") and hf_config.num_labels:
                num_labels = hf_config.num_labels
                # Create default mapping: {0: "LABEL_0", 1: "LABEL_1", ...}
                return {i: f"LABEL_{i}" for i in range(num_labels)}

        except Exception as e:
            logger.warning(f"Failed to get id2label mapping: {e}")

        return None

    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: ClassifyRequest,
        raw_request: Request,
    ) -> Union[ClassifyResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming classification request."""
        # Generate request ID

        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_classify_response(ret)
        return response

    def _build_classify_response(self, ret: List[Dict[str, Any]]) -> ClassifyResponse:
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        created_time = int(time.time())
        classify_objects = []
        prompt_tokens = 0
        total_latency = 0.0

        for i, item in enumerate(ret):
            embedding = item.get("embedding", [])
            meta_info = item.get("meta_info", {})

            prompt_tokens += meta_info.get("prompt_tokens", 0)
            total_latency += meta_info.get("e2e_latency", 0.0)

            if embedding:
                try:
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                    probs = F.softmax(embedding_tensor, dim=0).tolist()

                    predicted_class = torch.argmax(embedding_tensor).item()

                    label = self.id2label[predicted_class]

                except Exception as e:
                    logger.error(f"Error processing embedding for item {i}: {e}")
                    probs = [1.0]
                    label = "Default"
            else:
                probs = [1.0]
                label = "Default"

            classify_obj = {
                "index": i,
                "label": label,
                "probs": probs,
                "num_classes": len(probs),
            }
            classify_objects.append(classify_obj)

        response = {
            "id": request_id,
            "object": "list",
            "created": created_time,
            "model": self.model_name,
            "data": classify_objects,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
                "completion_tokens": 0,
                "prompt_tokens_details": None,
            },
        }

        return ClassifyResponse(**response)
