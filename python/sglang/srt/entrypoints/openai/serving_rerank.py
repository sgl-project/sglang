import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ErrorResponse,
    RerankContent,
    RerankResponse,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)

QWEN3_YES_TOKEN_ID = 9693
QWEN3_NO_TOKEN_ID = 2152


def _is_qwen3_reranker_template(chat_template: str) -> bool:
    """Detect if the chat template is for Qwen3 text-only reranker."""
    if not chat_template:
        return False
    t = chat_template.lower()
    return ('answer can only be "yes" or "no"' in t) or (
        "answer can only be" in t and '"yes"' in t and '"no"' in t
    )


def _is_qwen3_vl_reranker_template(chat_template: str) -> bool:
    """Detect if the chat template is for Qwen3-VL multimodal reranker.

    VL reranker templates use `query` and `document` as jinja variables
    and include vision token placeholders for image/video support.
    """
    if not chat_template:
        return False
    t = chat_template.lower()
    # Check for reranker phrase (yes/no judgment)
    has_reranker_phrase = ('answer can only be "yes" or "no"' in t) or (
        "answer can only be" in t and '"yes"' in t and '"no"' in t
    )
    # Check for vision token placeholders (unique to VL templates)
    has_vision_tokens = "<|vision_start|>" in t or "<|image_pad|>" in t
    return has_reranker_phrase and has_vision_tokens


def _is_qwen3_vl_model(model_path: str) -> bool:
    """Check if the model is a Qwen3-VL model based on model path."""
    if not model_path:
        return False
    model_lower = model_path.lower()
    return "qwen3-vl" in model_lower or "qwen3vl" in model_lower


def _qwen3_rerank_score(p_yes: float, p_no: float) -> float:
    denom = p_yes + p_no
    if denom <= 0.0:
        return 0.0
    return p_yes / denom


def _render_jinja_chat_template(
    chat_template: str,
    *,
    query: RerankContent,
    document: RerankContent,
    instruct: Optional[str],
) -> str:
    """Render a loaded Jinja chat template for Qwen3 reranker prompts (text-only)."""
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

    # For text-only template, extract text content
    query_text = query if isinstance(query, str) else _extract_text_from_content(query)
    doc_text = (
        document if isinstance(document, str) else _extract_text_from_content(document)
    )

    render_kwargs = {
        "messages": [
            {"role": "user", "content": query_text},
            {"role": "user", "content": doc_text},
        ]
    }
    # Only pass instruct when explicitly provided; template uses `default(...)`
    # which works only when the variable is undefined (not None).
    if instruct:
        render_kwargs["instruct"] = instruct
    return template.render(**render_kwargs)


def _render_vl_jinja_template(
    chat_template: str,
    *,
    query: List[Dict[str, Any]],
    document: List[Dict[str, Any]],
    instruct: Optional[str],
) -> str:
    """Render a loaded Jinja chat template for Qwen3-VL reranker prompts (multimodal).

    The template expects `query` and `document` as lists of content parts,
    where each part has a `type` field (text, image, video) and corresponding data.
    """
    try:
        import jinja2
    except ModuleNotFoundError as e:
        raise ValueError(
            "Rendering Qwen3-VL reranker prompts requires `jinja2`. "
            "Please install it in your runtime environment (e.g., `pip install jinja2`)."
        ) from e

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        undefined=jinja2.Undefined,
    )
    template = env.from_string(chat_template)

    render_kwargs = {
        "query": query,
        "document": document,
    }
    if instruct:
        render_kwargs["instruct"] = instruct
    return template.render(**render_kwargs)


def _extract_text_from_content(content: RerankContent) -> str:
    """Extract text from multimodal content."""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if isinstance(part, ChatCompletionMessageContentTextPart):
            texts.append(part.text)
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    return " ".join(texts)


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

        - For Qwen3-VL reranker (multimodal decoder-only): keep the request.
        - For Qwen3 reranker (text-only decoder-only): keep the request and score via
          `tokenizer_manager.score_prompts(...)` in the handler.
        - For cross-encoder rerank models: adapt into `EmbeddingReqInput` pairs.
        """
        chat_template = self.tokenizer_manager.tokenizer.chat_template
        model_path = getattr(self.tokenizer_manager.model_config, "model_path", "")

        # Check if this is a multimodal request
        is_multimodal = request.is_multimodal()

        # Check if this is a Qwen3-VL model
        is_vl_model = _is_qwen3_vl_model(model_path)

        # Check if using VL reranker template
        is_vl_template = isinstance(
            chat_template, str
        ) and _is_qwen3_vl_reranker_template(chat_template)

        # For VL template/model or multimodal requests with reranker template, keep as is
        if (
            is_vl_template
            or is_vl_model
            or (
                is_multimodal
                and isinstance(chat_template, str)
                and _is_qwen3_reranker_template(chat_template)
            )
        ):
            return request, request

        # Only treat as Qwen3 text-only reranker when the chat template matches
        # and it's not a VL template (VL templates are handled above).
        if isinstance(chat_template, str) and _is_qwen3_reranker_template(
            chat_template
        ):
            return request, request

        # Cross-encoder rerank: Create pairs of [query, document] for each document.
        # Note: Cross-encoder only supports text-only content
        if is_multimodal:
            # Extract text for cross-encoder (multimodal not supported)
            query_text = _extract_text_from_content(request.query)
            doc_texts = [_extract_text_from_content(doc) for doc in request.documents]
            pairs = [[query_text, doc] for doc in doc_texts]
        else:
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
        chat_template = getattr(self.tokenizer_manager.tokenizer, "chat_template", None)
        model_path = getattr(self.tokenizer_manager.model_config, "model_path", "")

        # Check if this is a multimodal request
        is_multimodal = request.is_multimodal()

        # Check if this is a Qwen3-VL model (for multimodal reranking)
        is_vl_model = _is_qwen3_vl_model(model_path)

        # Check if using VL reranker template (expects query/document format, not messages)
        is_vl_template = isinstance(
            chat_template, str
        ) and _is_qwen3_vl_reranker_template(chat_template)

        # Qwen3-VL reranker path (decoder-only scoring with query/document template format)
        # Trigger if: VL template OR VL model OR (multimodal request with reranker template)
        if (
            is_vl_template
            or is_vl_model
            or (
                is_multimodal
                and isinstance(chat_template, str)
                and _is_qwen3_reranker_template(chat_template)
            )
        ):
            return await self._handle_vl_reranker_request(
                request, raw_request, chat_template or ""
            )

        # Qwen3 text-only reranker path (decoder-only scoring).
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

    async def _handle_vl_reranker_request(
        self,
        request: V1RerankReqInput,
        raw_request: Request,
        _chat_template: str,
    ) -> Union[List[RerankResponse], ErrorResponse]:
        """Handle multimodal VL reranker request using chat completion with logprobs."""
        if not self.tokenizer_manager.model_config.is_generation:
            return self.create_error_response(
                "Detected Qwen3-VL reranker, but the server is not in generation mode. "
                "Please relaunch without --is-embedding for Qwen3-VL-Reranker models."
            )

        try:
            scores = []
            instruct = getattr(request, "instruct", None)

            for doc in request.documents:
                # Build multimodal content lists and render prompt using jinja template
                query_content, doc_content, image_data, video_data = (
                    self._build_vl_reranker_content(
                        query=request.query,
                        document=doc,
                    )
                )

                # Render the chat template directly with query/document variables
                prompt = _render_vl_jinja_template(
                    chat_template=_chat_template,
                    query=query_content,
                    document=doc_content,
                    instruct=instruct,
                )

                # Create generate request with logprobs
                gen_request = GenerateReqInput(
                    text=prompt,
                    image_data=image_data if image_data else None,
                    video_data=video_data if video_data else None,
                    sampling_params={
                        "max_new_tokens": 1,
                        "temperature": 0,
                    },
                    return_logprob=True,
                    top_logprobs_num=50,  # Get enough logprobs to find yes/no tokens
                    logprob_start_len=0,
                )

                # Execute generation request
                ret = await self.tokenizer_manager.generate_request(
                    gen_request, raw_request
                ).__anext__()

                # Extract yes/no probabilities from logprobs
                score = self._extract_score_from_logprobs(ret)
                scores.append(score)

            responses = self._build_rerank_response(scores, request)
            return responses

        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.exception("Error handling VL reranker request")
            return self.create_error_response(str(e))

    def _build_vl_reranker_content(
        self,
        query: RerankContent,
        document: RerankContent,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
        """Build content lists for VL reranker request.

        Returns:
            Tuple of (query_content, document_content, image_data, video_data)
            where query_content and document_content are lists suitable for jinja template.
        """
        image_data = []
        video_data = []

        # Build query content list
        query_content = self._content_to_template_list(query, image_data, video_data)

        # Build document content list
        doc_content = self._content_to_template_list(document, image_data, video_data)

        return query_content, doc_content, image_data, video_data

    def _content_to_template_list(
        self,
        content: RerankContent,
        image_data: List[str],
        video_data: List[str],
    ) -> List[Dict[str, Any]]:
        """Convert RerankContent to a list format suitable for jinja template."""
        result = []

        if isinstance(content, str):
            result.append({"type": "text", "text": content})
            return result

        for part in content:
            if isinstance(part, ChatCompletionMessageContentTextPart):
                result.append({"type": "text", "text": part.text})
            elif isinstance(part, ChatCompletionMessageContentImagePart):
                if part.image_url:
                    image_data.append(part.image_url.url)
                    result.append({"type": "image"})
            elif isinstance(part, ChatCompletionMessageContentVideoPart):
                if part.video_url:
                    video_data.append(part.video_url.url)
                    result.append({"type": "video"})
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    result.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if url:
                        image_data.append(url)
                        result.append({"type": "image"})
                elif part_type == "video_url":
                    video_url = part.get("video_url", {})
                    if isinstance(video_url, dict):
                        url = video_url.get("url")
                    else:
                        url = video_url
                    if url:
                        video_data.append(url)
                        result.append({"type": "video"})

        return result

    def _extract_score_from_logprobs(self, ret: Dict[str, Any]) -> float:
        """Extract reranking score from generation response with logprobs."""
        import math

        # Get logprobs from the response
        meta_info = ret.get("meta_info", {})
        input_top_logprobs = meta_info.get("input_top_logprobs", [])
        output_top_logprobs = meta_info.get("output_top_logprobs", [])

        # Use input_top_logprobs at the last position - this gives us the model's
        # prediction for what token comes next after processing the full prompt
        # (which should be "yes" or "no")
        if input_top_logprobs:
            top_logprobs = input_top_logprobs[-1]
        elif output_top_logprobs:
            top_logprobs = output_top_logprobs[0]
        else:
            logger.warning("No logprobs found in response, returning 0.0")
            return 0.0

        # Find yes and no token probabilities
        p_yes = 0.0
        p_no = 0.0

        if isinstance(top_logprobs, dict):
            # Format: {token_id: logprob}
            for token_id, logprob in top_logprobs.items():
                token_id_int = int(token_id) if isinstance(token_id, str) else token_id
                if token_id_int == QWEN3_YES_TOKEN_ID:
                    p_yes = math.exp(logprob)
                elif token_id_int == QWEN3_NO_TOKEN_ID:
                    p_no = math.exp(logprob)
        elif isinstance(top_logprobs, list):
            # Format: list of tuples (logprob, token_id, token_text) or list of dicts
            for item in top_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    # SGLang format: (logprob, token_id, token_text)
                    logprob, token_id = item[0], item[1]
                    if token_id == QWEN3_YES_TOKEN_ID:
                        p_yes = math.exp(logprob)
                    elif token_id == QWEN3_NO_TOKEN_ID:
                        p_no = math.exp(logprob)
                elif isinstance(item, dict):
                    token_id = item.get("token_id", item.get("id"))
                    logprob = item.get("logprob", item.get("logprobs"))
                    if token_id == QWEN3_YES_TOKEN_ID:
                        p_yes = math.exp(logprob) if logprob else 0.0
                    elif token_id == QWEN3_NO_TOKEN_ID:
                        p_no = math.exp(logprob) if logprob else 0.0

        return _qwen3_rerank_score(p_yes, p_no)

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
