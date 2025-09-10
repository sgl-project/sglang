import logging
from typing import Union

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    ScoringRequest,
    ScoringResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

logger = logging.getLogger(__name__)


class OpenAIServingScore(OpenAIServingBase):
    """Handler for /v1/score requests"""

    # NOTE: /v1/rerank is not an official OpenAI endpoint. This module may be moved
    # to another module in the future.

    def _request_id_prefix(self) -> str:
        return "score-"

    def _convert_to_internal_request(
        self,
        request: ScoringRequest,
    ) -> tuple[ScoringRequest, ScoringRequest]:
        """Convert OpenAI scoring request to internal format"""
        # For scoring, we pass the request directly as the tokenizer_manager
        # has a specialized score_request method that doesn't use GenerateReqInput

        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: ScoringRequest,
        request: ScoringRequest,
        raw_request: Request,
    ) -> Union[ScoringResponse, ErrorResponse]:
        """Handle the scoring request"""
        try:
            # Use tokenizer_manager's score_request method directly
            scores = await self.tokenizer_manager.score_request(
                query=request.query,
                items=request.items,
                label_token_ids=request.label_token_ids,
                apply_softmax=request.apply_softmax,
                item_first=request.item_first,
                request=raw_request,
            )

            # Create response with just the scores, without usage info
            response = ScoringResponse(
                scores=scores,
                model=request.model,
            )
            return response

        except ValueError as e:
            return self.create_error_response(str(e))
