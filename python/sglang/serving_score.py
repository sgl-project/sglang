# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

from fastapi import Request
from sglang.srt.openai_api.protocol import (ErrorResponse, RerankDocument,
                                              RerankRequest, RerankResponse,
                                              RerankResult, RerankUsage,
                                              ScoreRequest, ScoreResponse,
                                              ScoreResponseData, UsageInfo,
                                              EmbeddingRequest)
from torch.nn import CosineSimilarity
from concurrent.futures import ThreadPoolExecutor
from sglang.utils import make_async

class ServingScores:

    def __init__(self, tokenizer_manager) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.model_config = tokenizer_manager.model_config
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)


    async def _embedding_score(
        self,
        model: str,
        texts_1: list[str],
        texts_2: list[str],
        raw_request: Request
    ) -> ScoreResponse:

        from sglang.srt.openai_api.adapter import v1_embedding_request

        scorer = CosineSimilarity(0)

        input_texts = texts_1 + texts_2
        all_requests = [EmbeddingRequest(input=input_texts, model=model)]

        adapted_request, request = v1_embedding_request(all_requests, self.tokenizer_manager)
        
        ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        
        
        score_data = []
        embeddings = [None] * len(input_texts)

        emb_texts_1 = []
        emb_texts_2 = []

        prompt_tokens = 0

        for i in range(0, len(texts_1)):
            emb = ret[i]['embedding']
            assert emb is not None
            emb_texts_1.append(emb)
            prompt_tokens += ret[i]["meta_info"]["prompt_tokens"]

        for i in range(len(texts_1), len(embeddings)):
            emb = ret[i]['embedding']
            assert emb is not None
            emb_texts_2.append(emb)
            prompt_tokens += ret[i]["meta_info"]["prompt_tokens"]

        if len(emb_texts_1) == 1:
            emb_texts_1 = emb_texts_1 * len(emb_texts_2)

        for idx, (emb_1, emb_2) in enumerate(zip(emb_texts_1, emb_texts_2)):
            pair_score = scorer(emb_1, emb_2)
            _score_data = ScoreResponseData(index = idx,
                                            score = pair_score.item())
            score_data.append(_score_data)
        
        score_response = ScoreResponse(
            model = model,
            data = score_data,
            usage = UsageInfo(prompt_tokens=prompt_tokens)

        )


        return score_response

    async def _cross_encoding_score(
        self,
        model: str,
        texts_1: list[str],
        texts_2: list[str],
        raw_request: Request
    ) -> ScoreResponse:

        from sglang.srt.openai_api.adapter import v1_embedding_request

        request_prompts: list[str] = []
        engine_prompts = []

        if len(texts_1) == 1:
            texts_1 = texts_1 * len(texts_2)

        input_pairs = [(t1, t2) for t1, t2 in zip(texts_1, texts_2)]

        tokenize_async = make_async(self.tokenizer_manager.tokenizer.__call__,
                                    executor=self._tokenizer_executor)

        tokenization_kwargs = tokenization_kwargs or {}
        tokenized_prompts = await asyncio.gather(
            *(tokenize_async(text=t1, text_pair=t2, **tokenization_kwargs)
              for t1, t2 in input_pairs))

        for prompt_inputs, (t1, t2) in zip(tokenized_prompts, input_pairs):

            request_prompt = f"{t1}{self.tokenizer_manager.tokenizer.sep_token}{t2}"

            input_ids = prompt_inputs["input_ids"]
            engine_prompt = EmbeddingRequest(input=input_ids, model=model)

            request_prompts.append(request_prompt)
            engine_prompts.append(engine_prompt)
        
        adapted_request, request = v1_embedding_request(engine_prompts, self.tokenizer_manager)
        
        ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()

        # Non-streaming response
        final_res_batch = [None] * len(engine_prompts)

        prompt_tokens = 0
        for i in len(engine_prompts):
            final_res_batch[i] = ret[i]["embedding"]
            prompt_tokens += ret[i]["meta_info"]["prompt_tokens"]

        score_data = []
        embed = [out for out in final_res_batch if out is not None]
        for idx, emb in enumerate(embed):
            
            _score_data = ScoreResponseData(index = idx,
                                            score = emb.item())
            score_data.append(_score_data)
        
        score_response = ScoreResponse(
            model = model,
            data = score_data,
            usage = UsageInfo(prompt_tokens=prompt_tokens)

        )


        return score_response

    async def _run_scoring(
        self, request: ScoreRequest, raw_request: Request
    ) -> ScoreResponse:

        if isinstance(request.text_1, str):
            request.text_1 = [request.text_1]
        
        if isinstance(request.text_2, str):
            request.text_2 = [request.text_2]

        
        if self.model_config.is_cross_encoder:
            return await self._cross_encoding_score(
                model = request.model,
                texts_1=request.text_1,
                texts_2=request.text_2
            )

        else:
            return await self._embedding_score(
                model = request.model,
                texts_1=request.text_1,
                texts_2=request.text_2,
                raw_request=raw_request
            )

    

    async def do_rerank(
        self,
        request: RerankRequest,
        raw_request: Optional[Request] = None
    ) -> Union[RerankResponse, ErrorResponse]:
        """
        Rerank API based on JinaAI's rerank API; implements the same
        API interface. Designed for compatibility with off-the-shelf
        tooling, since this is a common standard for reranking APIs

        See example client implementations at
        https://github.com/infiniflow/ragflow/blob/main/rag/llm/rerank_model.py
        numerous clients use this standard.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"rerank-{self._base_request_id(raw_request)}"
        documents = request.documents
        top_n = request.top_n if request.top_n > 0 else len(documents)

        try:
            final_res_batch = await self._run_scoring(
                request.query,
                documents,
                request,
                request_id,
                raw_request,
                request.truncate_prompt_tokens,
            )
            return self.request_output_to_rerank_response(
                final_res_batch,
                request_id,
                self._get_model_name(request.model),
                documents,
                top_n,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    # def request_output_to_rerank_response(
    #         self, final_res_batch: list[PoolingRequestOutput], request_id: str,
    #         model_name: str, documents: list[str],
    #         top_n: int) -> RerankResponse:
    #     """
    #     Convert the output of do_rank to a RerankResponse
    #     """
    #     results: list[RerankResult] = []
    #     num_prompt_tokens = 0
    #     for idx, final_res in enumerate(final_res_batch):
    #         classify_res = ScoringRequestOutput.from_base(final_res)

    #         result = RerankResult(
    #             index=idx,
    #             document=RerankDocument(text=documents[idx]),
    #             relevance_score=classify_res.outputs.score,
    #         )
    #         results.append(result)
    #         prompt_token_ids = final_res.prompt_token_ids
    #         num_prompt_tokens += len(prompt_token_ids)

    #     # sort by relevance, then return the top n if set
    #     results.sort(key=lambda x: x.relevance_score, reverse=True)
    #     if top_n < len(documents):
    #         results = results[:top_n]

    #     return RerankResponse(
    #         id=request_id,
    #         model=model_name,
    #         results=results,
    #         usage=RerankUsage(total_tokens=num_prompt_tokens))
