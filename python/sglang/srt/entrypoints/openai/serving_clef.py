"""Joint Clef requests using SGLang's prefill-only GPU execution path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from fastapi.responses import ORJSONResponse

from sglang.srt.layers.clef import validate_record
from sglang.srt.layers.clef_reference import encode_record, systemone_answer
from sglang.srt.managers.io_struct import GenerateReqInput


def decision_record(request: Any) -> dict:
    """Translate the decisions wire schema to Clef's unchanged native schema."""
    questions = {}
    for question in request.questions:
        native = {"instructions": question.question}
        if question.type == "yes_no":
            native["type"] = "noul"
            native["criteria"] = {
                key: value
                for key, value in (("true", question.yes), ("false", question.no))
                if value is not None
            }
        elif question.type == "choice":
            native.update(
                type="choice",
                criteria={
                    option.name: option.description for option in question.options
                },
            )
        else:
            native.update(type="score", criteria=question.levels)
        questions[question.id] = native
    return {"state": request.input, "questions": questions}


async def handle_clef_request(
    serving: Any, request: Any, raw_request: Any
) -> ORJSONResponse:
    """Answer all fields from one joint prompt."""
    manager = serving.tokenizer_manager
    # Custom server/template configurations are unsupported by the fixed encoder.
    error = serving._validate_server(request.model)
    if error:
        raise ValueError(error)
    if request.images or request.chat_template_kwargs:
        raise ValueError(
            "Clef currently supports text only and its fixed native prompt"
        )
    decisions = serving.route == "/v1/decisions"
    if decisions:
        if request.temperature != 1 or request.prompt_format_version not in (None, 1):
            raise ValueError("Clef requires temperature=1 and prompt_format_version=1")
        record = decision_record(request)
    else:
        record = {
            "state": request.state,
            "questions": {
                key: value.model_dump(exclude_unset=True)
                for key, value in request.questions.items()
            },
        }
    # The reference truncates state at max_length. Encode unbounded, then reject:
    # every question must retain its original spans and complete state context.
    encoded = await asyncio.to_thread(
        encode_record, manager.tokenizer, record, max_length=2**63 - 1
    )
    token_ids = list(encoded.input_ids)
    if len(token_ids) + manager.num_reserved_tokens >= manager.context_len:
        raise ValueError(
            f"Clef prompt has {len(token_ids)} tokens and exceeds context length {manager.context_len}"
        )
    validate_record(encoded, token_ids)
    internal = GenerateReqInput(
        input_ids=token_ids,
        sampling_params={
            "max_new_tokens": 0,
            "temperature": 0,
            "custom_params": {"clef_record": json.dumps(asdict(encoded))},
        },
    )
    result = None
    async for response in manager.generate_request(internal, raw_request):
        result = response
    if result is None:
        raise RuntimeError("Clef worker returned no result")
    metadata = result["meta_info"]
    finish_reason = metadata.get("finish_reason") or {}
    if finish_reason.get("type") == "abort":
        return serving.create_error_response(
            message=finish_reason.get("message") or "Clef decision request aborted.",
            err_type="RequestAborted",
            status_code=finish_reason.get("status_code") or 503,
        )
    values = metadata.get("clef_probabilities")
    execution = metadata.get("clef_execution")
    if not values or len(values) != 1 or not execution:
        raise RuntimeError("Clef worker did not execute its trained joint head")
    probabilities = values[0]
    if not decisions:
        return ORJSONResponse(
            {
                "model": manager.served_model_name,
                "answers": {
                    key: systemone_answer(question, probabilities[key])
                    for key, question in record["questions"].items()
                },
                "usage": {"input_tokens": len(token_ids), "output_tokens": 0},
                "execution": execution[0],
            }
        )
    answers = {}
    for question in request.questions:
        probs = probabilities[question.id]
        answer = {"type": question.type, "probabilities": probs}
        if question.type == "yes_no":
            answer["probabilities"] = {"yes": probs["true"], "no": probs["false"]}
        elif question.type == "choice":
            answer["choice"] = max(
                (option.name for option in question.options), key=probs.__getitem__
            )
        else:
            answer["score"] = sum(int(level) * value for level, value in probs.items())
        if request.return_prompt_token_ids:
            answer["prompt_token_ids"] = token_ids
        # No vocabulary labels/readout mass exist for the trained joint head.
        answers[question.id] = answer
    return ORJSONResponse(
        {
            "object": "decisions",
            "model": request.model,
            "prompt_format_version": 1,
            "answers": answers,
            "execution": execution[0],
            "usage": {
                "prompt_tokens": len(token_ids),
                "completion_tokens": 0,
                "total_tokens": len(token_ids),
            },
        }
    )
