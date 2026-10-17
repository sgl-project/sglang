# SPDX-License-Identifier: Apache-2.0
"""Prefill-only Valen decisions with the released model's prompt and answers."""

import asyncio
from contextlib import aclosing

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.valen.media import encode_valen_prompt
from sglang.srt.entrypoints.valen.protocol import ValenRequest
from sglang.srt.entrypoints.valen.responses import valen_answers
from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
from sglang.srt.layers.valen_layout import pack_valen_layout
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.runtime_context import get_mm, get_schedule


class ValenServing:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    async def handle_request(self, request: ValenRequest, raw_request=None):
        manager = self.tokenizer_manager
        try:
            if not manager.model_config.is_valen:
                raise ValueError("/v1/valen requires a Valen checkpoint")
            if manager.tokenizer is None:
                raise ValueError("Valen requires the server tokenizer")
            if request.model and ":" in request.model:
                raise ValueError("Valen adapter requests are not supported")
            limit = min(
                manager.model_config.hf_config.valen_max_length,
                max_joint_prompt_tokens(
                    context_len=manager.context_len,
                    num_reserved_tokens=manager.num_reserved_tokens,
                    max_req_input_len=manager.max_req_input_len,
                    max_prefill_tokens=get_schedule().max_prefill_tokens,
                ),
            )
            media_limits = get_mm().limit_mm_data_per_request

            def compile_request():
                return encode_valen_prompt(
                    manager.tokenizer,
                    getattr(manager, "processor", None),
                    request.state,
                    {qid: q.model_dump() for qid, q in request.questions.items()},
                    media_kwargs=manager.model_config.hf_config.valen_media_kwargs,
                    max_length=limit,
                    media_limits=media_limits,
                )

            processed = await asyncio.to_thread(compile_request)
            encoded = processed.encoded
            obj = EmbeddingReqInput(
                input_ids=encoded.input_ids,
                valen_layout=pack_valen_layout(encoded.layout),
                image_data=processed.image_data,
                video_data=processed.video_data,
            )
            async with aclosing(manager.generate_request(obj, raw_request)) as results:
                result = await results.__anext__()
            answers = valen_answers(
                encoded.questions, result["embedding"], request.temperature
            )
            tokens = result["meta_info"]["prompt_tokens"]
            return ORJSONResponse(
                content={
                    "model": manager.served_model_name,
                    "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "internal_usage": {"compute_tokens": tokens},
                }
            )
        except ValueError as error:
            return ORJSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error",
                        "code": 400,
                    }
                },
            )

    async def handle_systemone(self, request, raw_request=None):
        try:
            if request.chat_template_kwargs:
                raise ValueError(
                    "Valen uses the trained chat template without chat_template_kwargs"
                )
            if not isinstance(request.state, str):
                raise ValueError(
                    "Use /v1/valen for Valen messages; System One state must be text"
                )
            questions = {}
            for qid, q in request.questions.items():
                item = q.model_dump(exclude_none=True)
                if q.type == "noul" and q.criteria is not None:
                    if any(v is not None for v in q.criteria.model_dump().values()):
                        raise ValueError(
                            "Valen uses its trained fixed Noul descriptions"
                        )
                questions[qid] = item
            state = request.state
            if request.images:
                state = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": state},
                                *(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image.url
                                            if hasattr(image, "url")
                                            else image
                                        },
                                    }
                                    for image in request.images
                                ),
                            ],
                        }
                    ]
                }
            native = ValenRequest(model=request.model, state=state, questions=questions)
            return await self.handle_request(native, raw_request)
        except ValueError as error:
            return ORJSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error",
                        "code": 400,
                    }
                },
            )
