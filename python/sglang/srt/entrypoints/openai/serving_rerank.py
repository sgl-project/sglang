import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    RerankResponse,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput

logger = logging.getLogger(__name__)


class OpenAIServingRerank(OpenAIServingBase):
    """Handler for /v1/rerank requests"""

    # NOTE: /v1/rerank is not an official OpenAI endpoint. This module may be moved
    # to another module in the future.

    def _request_id_prefix(self) -> str:
        return "rerank-"

    def _validate_request(self, request: V1RerankReqInput) -> Optional[str]:
        """Validate rerank request format and content"""
        if not request.query:
            return "Query cannot be empty"

        if isinstance(request.query, str):
            if not request.query.strip():
                return "Query cannot be empty or whitespace only"

        if not request.documents:
            return "Documents cannot be empty"

        for doc in request.documents:
            if not doc:
                return "Each document must be a non-empty string"
            if isinstance(doc, str) and not doc.strip():
                return "Each document cannot be empty or whitespace only"

        return None

    def _convert_to_internal_request(
        self, request: V1RerankReqInput
    ) -> tuple[EmbeddingReqInput, V1RerankReqInput]:
        """Convert OpenAI rerank request to internal embedding format"""
        # Create pairs of [query, document] for each document
        pairs = []
        for doc in request.documents:
            pairs.append([request.query, doc])

        adapted_request = EmbeddingReqInput(
            text=pairs,
            is_cross_encoder_request=True,
        )

        return adapted_request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: V1RerankReqInput,
        raw_request: Request,
    ) -> Union[List[RerankResponse], ErrorResponse, ORJSONResponse]:
        """Handle the rerank request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()

        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        responses = self._build_rerank_response(ret, request)
        return responses

    def _build_rerank_response(
        self, ret: List[Dict[str, Any]], request: V1RerankReqInput
    ) -> List[RerankResponse]:
        """Build the rerank response from generation results"""
        responses = []
        for idx, ret_item in enumerate(ret):
            responses.append(
                RerankResponse(
                    score=ret_item["embedding"],
                    document=request.documents[idx],
                    index=idx,
                    meta_info=ret_item["meta_info"],
                )
            )

        # Sort by score in descending order (highest relevance first)
        responses.sort(key=lambda x: x.score, reverse=True)

        return responses
