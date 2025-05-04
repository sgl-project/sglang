# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

import torch
from fastapi import Request
from torch.nn import CosineSimilarity

from sglang.srt.openai_api.protocol import (
    EmbeddingRequest,
    ErrorResponse,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
    UsageInfo,
)


class ServingScores:

    def __init__(self, tokenizer_manager) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.model_config = tokenizer_manager.model_config

    async def _embedding_score(
        self, model: str, texts_1: list[str], texts_2: list[str], raw_request: Request
    ) -> ScoreResponse:

        from sglang.srt.openai_api.adapter import v1_embedding_request

        scorer = CosineSimilarity(0)

        input_texts = texts_1 + texts_2
        all_requests = [EmbeddingRequest(input=input_texts, model=model)]

        adapted_request, request = v1_embedding_request(
            all_requests, self.tokenizer_manager
        )

        ret = await self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()

        score_data = []
        embeddings = [None] * len(input_texts)

        emb_texts_1 = []
        emb_texts_2 = []

        prompt_tokens = 0

        for i in range(0, len(texts_1)):
            emb = ret[i]["embedding"]
            assert emb is not None
            emb_texts_1.append(emb)
            prompt_tokens += ret[i]["meta_info"]["prompt_tokens"]

        for i in range(len(texts_1), len(embeddings)):
            emb = ret[i]["embedding"]
            assert emb is not None
            emb_texts_2.append(emb)
            prompt_tokens += ret[i]["meta_info"]["prompt_tokens"]

        if len(emb_texts_1) == 1:
            emb_texts_1 = emb_texts_1 * len(emb_texts_2)

        for idx, (emb_1, emb_2) in enumerate(zip(emb_texts_1, emb_texts_2)):
            pair_score = scorer(torch.tensor(emb_1), torch.tensor(emb_2))
            _score_data = ScoreResponseData(index=idx, score=pair_score.item())
            score_data.append(_score_data)

        score_response = ScoreResponse(
            model=model, data=score_data, usage=UsageInfo(prompt_tokens=prompt_tokens)
        )

        return score_response

    async def _cross_encoding_score(
        self, model: str, texts_1: list[str], texts_2: list[str], raw_request: Request
    ) -> ScoreResponse:

        from sglang.srt.openai_api.adapter import v1_embedding_request

        engine_prompts = []

        if len(texts_1) == 1:
            texts_1 = texts_1 * len(texts_2)

        input_pairs = [(t1, t2) for t1, t2 in zip(texts_1, texts_2)]

        for t1, t2 in input_pairs:
            request_prompt = f"{t1}{self.tokenizer_manager.tokenizer.sep_token}{t2}"
            engine_prompt = EmbeddingRequest(input=request_prompt, model=model)

            engine_prompts.append(engine_prompt)

        final_res_batch = [None] * len(engine_prompts)
        prompt_tokens = 0

        for i, engine_prompt in enumerate(engine_prompts):

            adapted_request, request = v1_embedding_request(
                [engine_prompt], self.tokenizer_manager
            )

            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()

            final_res_batch[i] = ret["embedding"]
            prompt_tokens += ret["meta_info"]["prompt_tokens"]

        score_data = []
        embed = [out for out in final_res_batch if out is not None]

        for idx, emb in enumerate(embed):
            _score_data = ScoreResponseData(index=idx, score=emb)
            score_data.append(_score_data)

        score_response = ScoreResponse(
            model=model, data=score_data, usage=UsageInfo(prompt_tokens=prompt_tokens)
        )

        return score_response

    async def _run_scoring(
        self, request: ScoreRequest, raw_request: Request
    ) -> ScoreResponse:

        if isinstance(request.text_1, str):
            request.text_1 = [request.text_1]

        if isinstance(request.text_2, str):
            request.text_2 = [request.text_2]

        if self.model_config.is_cross_encoder_model:
            return await self._cross_encoding_score(
                model=request.model,
                texts_1=request.text_1,
                texts_2=request.text_2,
                raw_request=raw_request,
            )

        else:
            return await self._embedding_score(
                model=request.model,
                texts_1=request.text_1,
                texts_2=request.text_2,
                raw_request=raw_request,
            )

    async def do_rerank(
        self, request: RerankRequest, raw_request: Optional[Request] = None
    ) -> Union[RerankResponse, ErrorResponse]:
        """
        Rerank API based on JinaAI's rerank API; implements the same
        API interface. Designed for compatibility with off-the-shelf
        tooling, since this is a common standard for reranking APIs

        See example client implementations at
        https://github.com/infiniflow/ragflow/blob/main/rag/llm/rerank_model.py
        numerous clients use this standard.
        """

        documents = request.documents
        top_n = request.top_n if request.top_n > 0 else len(documents)

        score_request = ScoreRequest(
            text_1=request.query, text_2=documents, model=request.model
        )
        final_res_batch = await self._run_scoring(score_request, raw_request)

        return self.request_output_to_rerank_response(
            final_res_batch,
            request.model,
            documents,
            top_n,
        )

    def request_output_to_rerank_response(
        self,
        final_res_batch: ScoreResponse,
        model_name: str,
        documents: list[str],
        top_n: int,
    ) -> RerankResponse:
        """
        Convert the output of do_rank to a RerankResponse
        """
        results: list[RerankResult] = []
        num_prompt_tokens = final_res_batch.usage.prompt_tokens
        for idx, final_res in enumerate(final_res_batch.data):

            result = RerankResult(
                index=idx,
                document=RerankDocument(text=documents[idx]),
                relevance_score=final_res.score,
            )
            results.append(result)

        # sort by relevance, then return the top n if set
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if top_n < len(documents):
            results = results[:top_n]

        return RerankResponse(
            model=model_name,
            results=results,
            usage=RerankUsage(total_tokens=num_prompt_tokens),
        )
