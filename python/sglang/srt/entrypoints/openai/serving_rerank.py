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

QWEN3_YES_TOKEN_ID = 9693
QWEN3_NO_TOKEN_ID = 2152


def _is_qwen3_reranker_template(chat_template: str) -> bool:
    # Heuristic: our bundled template includes these key phrases.
    if not chat_template:
        return False
    t = chat_template.lower()
    return ('answer can only be "yes" or "no"' in t) or (
        "answer can only be" in t and '"yes"' in t and '"no"' in t
    )


def _qwen3_rerank_score(p_yes: float, p_no: float) -> float:
    denom = p_yes + p_no
    if denom <= 0.0:
        return 0.0
    return p_yes / denom


def _render_jinja_chat_template(
    chat_template: str,
    *,
    query: str,
    document: str,
    instruct: Optional[str],
) -> str:
    """Render a loaded Jinja chat template for Qwen3 reranker prompts."""
    try:
        import jinja2  # Lazy import: server env should provide this dependency.
    except ModuleNotFoundError as e:
        raise ValueError(
            "Rendering Qwen3 reranker prompts requires `jinja2`. "
            "Please install it in your runtime environment (e.g., `pip install jinja2`)."
        ) from e

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        undefined=jinja2.Undefined,
    )
    template = env.from_string(chat_template)
    render_kwargs = {
        "messages": [
            {"role": "user", "content": query},
            {"role": "user", "content": document},
        ]
    }
    # Only pass instruct when explicitly provided; template uses `default(...)`
    # which works only when the variable is undefined (not None).
    if instruct:
        render_kwargs["instruct"] = instruct
    return template.render(**render_kwargs)


class OpenAIServingRerank(OpenAIServingBase):
    """Handler for /v1/rerank requests"""

    def __init__(self, tokenizer_manager, template_manager=None):
        super().__init__(tokenizer_manager)
        # TemplateManager is optional; rerank uses tokenizer.chat_template today.
        # Keeping this explicit makes the dependency clear and supports future extensions.
        self.template_manager = template_manager

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
        self,
        request: V1RerankReqInput,
        raw_request: Request = None,
    ) -> tuple[Union[EmbeddingReqInput, V1RerankReqInput], V1RerankReqInput]:
        """
        Convert OpenAI rerank request to internal format.

        - For cross-encoder rerank models: adapt into `EmbeddingReqInput` pairs.
        - For Qwen3 reranker (decoder-only): keep the request and score via
          `tokenizer_manager.score_prompts(...)` in the handler.
        """
        chat_template = self.tokenizer_manager.tokenizer.chat_template
        # Only treat as Qwen3 reranker when the chat template matches.
        # `is_generation` alone is too broad and can break non-Qwen3 generation models.
        if isinstance(chat_template, str) and _is_qwen3_reranker_template(
            chat_template
        ):
            return request, request

        # Cross-encoder rerank: Create pairs of [query, document] for each document.
        pairs = [[request.query, doc] for doc in request.documents]
        adapted_request = EmbeddingReqInput(text=pairs, is_cross_encoder_request=True)
        return adapted_request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: Union[EmbeddingReqInput, V1RerankReqInput],
        request: V1RerankReqInput,
        raw_request: Request,
    ) -> Union[List[RerankResponse], ErrorResponse, ORJSONResponse]:
        """Handle the rerank request"""
        # Qwen3 reranker path (decoder-only scoring).
        chat_template = getattr(self.tokenizer_manager.tokenizer, "chat_template", None)
        if isinstance(chat_template, str) and _is_qwen3_reranker_template(
            chat_template
        ):
            # Qwen3 reranker relies on decoder-only logprobs. If the server is launched
            # with --is-embedding, model_config.is_generation is typically False and
            # logprob scoring is not supported.
            if not self.tokenizer_manager.model_config.is_generation:
                return self.create_error_response(
                    "Detected Qwen3 reranker chat template, but the server is not in generation mode. "
                    "Please relaunch without --is-embedding for Qwen3-Reranker models."
                )
            try:
                prompts = [
                    _render_jinja_chat_template(
                        chat_template,
                        query=request.query,
                        document=doc,
                        instruct=getattr(request, "instruct", None),
                    )
                    for doc in request.documents
                ]

                probs = await self.tokenizer_manager.score_prompts(
                    prompts,
                    label_token_ids=[QWEN3_YES_TOKEN_ID, QWEN3_NO_TOKEN_ID],
                    apply_softmax=False,
                    request=raw_request,
                )
                scores = [_qwen3_rerank_score(p[0], p[1]) for p in probs]
            except ValueError as e:
                return self.create_error_response(str(e))
            except Exception as e:
                # Includes template rendering errors from jinja2.
                return self.create_error_response(str(e))

            responses = self._build_rerank_response(scores, request)
            return responses

        # Default cross-encoder rerank path (existing behavior).
        try:
            if not isinstance(adapted_request, EmbeddingReqInput):
                raise ValueError(
                    "Invalid rerank request adaptation. "
                    "If you are serving a decoder-only reranker (e.g., Qwen3-Reranker), "
                    "please provide the corresponding --chat-template and launch without --is-embedding."
                )
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
        self, ret: Union[List[Dict[str, Any]], List[float]], request: V1RerankReqInput
    ) -> List[RerankResponse]:
        """Build the rerank response from generation results"""
        responses = []
        for idx, item in enumerate(ret):
            if isinstance(item, dict):
                score_val = item.get("embedding")
                # Some rerank/reward models return scalar score as embedding[0].
                if isinstance(score_val, list):
                    if len(score_val) == 0 or not isinstance(
                        score_val[0], (int, float)
                    ):
                        raise ValueError(
                            f"Invalid embedding score for rerank at index {idx}: {score_val!r}"
                        )
                    score_val = float(score_val[0])
                responses.append(
                    RerankResponse(
                        score=float(score_val),
                        document=(
                            request.documents[idx] if request.return_documents else None
                        ),
                        index=idx,
                        meta_info=item.get("meta_info"),
                    )
                )
            else:
                responses.append(
                    RerankResponse(
                        score=float(item),
                        document=(
                            request.documents[idx] if request.return_documents else None
                        ),
                        index=idx,
                    )
                )

        # Sort by score in descending order (highest relevance first)
        responses.sort(key=lambda x: x.score, reverse=True)

        # Apply top_n limit if specified
        if request.top_n is not None and request.top_n > 0:
            responses = responses[: request.top_n]

        return responses
