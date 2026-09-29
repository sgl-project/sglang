"""Jev handler for decision model checkpoints: every field of a request is read from one prefill of the family's prompt."""

from __future__ import annotations

import math
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.decision.families import FAMILIES, detect_family
from sglang.srt.entrypoints.decision.families.base import (
    DecisionInputError,
    DecisionPrompt,
)
from sglang.srt.entrypoints.decision.protocol import (
    JevAnswer,
    JevCalibration,
    JevRequest,
    JevResponse,
    JevUsage,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import _CHAT_TEMPLATE_CLIENT_ERRORS
from sglang.srt.entrypoints.systemone.serving import (
    choice_confidence,
    score_confidence,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import get_exec, get_serving, get_spec

# Names the TypeSafe SDKs send by default, answered by the served checkpoint.
MODEL_ALIASES = ("jev-latest", "jev-preview")
_SUBJECT = "Decision model requests"


class DecisionModelServing(OpenAIServingBase):
    def __init__(self, tokenizer_manager):
        super().__init__(tokenizer_manager)
        self.family = detect_family(tokenizer_manager.tokenizer)

    def _request_id_prefix(self) -> str:
        return "decision-model-"

    async def handle_request(self, request: JevRequest, raw_request: Request):
        error = self._validate_request(request)
        if error is not None:
            return self.create_error_response(error)
        error = self.check_model(request.model)
        if error is not None:
            return self.create_error_response(
                message=error,
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND.value,
            )
        try:
            self.family.validate(request)
        except DecisionInputError as e:
            return _unprocessable(e)
        return await super().handle_request(request, raw_request)

    def _validate_request(self, request: JevRequest) -> Optional[str]:
        if not self.tokenizer_manager.is_generation:
            return f"{_SUBJECT} require a generation model"
        if self.family is None:
            names = ", ".join(family.name for family in FAMILIES)
            return (
                f"{_SUBJECT} require a checkpoint of a supported decision model "
                f"family ({names}), and the served model is not one"
            )
        if get_serving().chat_template is not None:
            return f"{_SUBJECT} render the checkpoint's own chat template, so --chat-template is not supported"
        # Marker readout needs prefill-only requests, which speculative decoding disables.
        if get_spec().speculative_algorithm is not None:
            return f"{_SUBJECT} do not support speculative decoding"
        if get_exec().features.enable_mis:
            return f"{_SUBJECT} do not support --enable-mis"
        if get_exec().dllm.dllm_algorithm is not None:
            return f"{_SUBJECT} do not support --dllm-algorithm"
        if get_serving().allow_auto_truncate:
            return f"{_SUBJECT} do not support --allow-auto-truncate, which can cut off readout positions"
        return None

    def check_model(self, model: Optional[str]) -> Optional[str]:
        served = self.tokenizer_manager.served_model_name
        if model is None or model == served or model in MODEL_ALIASES:
            return None
        return f"the model {model!r} is not served, use {served!r} or one of {list(MODEL_ALIASES)}"

    def _convert_to_internal_request(
        self,
        request: JevRequest,
        raw_request: Request = None,
    ) -> Tuple[GenerateReqInput, Tuple[JevRequest, DecisionPrompt]]:
        try:
            prompt = self.family.encode(request)
        except _CHAT_TEMPLATE_CLIENT_ERRORS as e:
            raise ValueError(f"the chat template failed: {e}") from e
        candidates = dict.fromkeys(
            token_id for field in prompt.fields for token_id in field.candidate_ids
        )
        # The tokenizer manager resolves readout positions on the expanded prompt
        # and refuses prompts longer than one prefill chunk.
        adapted = GenerateReqInput(
            text=prompt.text,
            input_ids=prompt.input_ids,
            image_data=prompt.images or None,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=0,
            token_ids_logprob=list(candidates),
            readout_anchor=prompt.readout_anchor,
            stream=False,
        )
        return adapted, (request, prompt)

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        processed: Tuple[JevRequest, DecisionPrompt],
        raw_request: Request,
    ) -> ORJSONResponse:
        request, prompt = processed
        result = await self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
        rows = result["meta_info"].get("input_token_ids_logprobs") or []
        if len(rows) != len(prompt.fields):
            raise RuntimeError(
                f"expected {len(prompt.fields)} readouts, got {len(rows)}"
            )
        temperature = 1.0 if request.temperature is None else request.temperature
        answers = {}
        for field, row in zip(prompt.fields, rows):
            logprobs = {token_id: logprob for logprob, token_id, _ in row}
            answers[field.name] = build_answer(
                kind=request.questions[field.name].type,
                options=field.options,
                probabilities=softmax(
                    [logprobs[token_id] for token_id in field.candidate_ids],
                    temperature,
                ),
            )
        response = JevResponse(
            model=self.tokenizer_manager.served_model_name,
            answers=answers,
            usage=JevUsage(
                input_tokens=result["meta_info"]["prompt_tokens"],
                output_tokens=len(answers),
                decision_count=len(answers),
            ),
            calibration=(
                None
                if request.temperature is None
                else JevCalibration(temperature=temperature)
            ),
        )
        return ORJSONResponse(content=response.model_dump(exclude_none=True))


def _unprocessable(error: DecisionInputError) -> ORJSONResponse:
    # The FastAPI detail list that request validation returns on these routes.
    detail = [
        {"type": "value_error", "loc": list(error.loc), "msg": f"Value error, {error}"}
    ]
    return ORJSONResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value, content={"detail": detail}
    )


def softmax(logprobs: List[float], temperature: float) -> List[float]:
    # Full-vocabulary logprobs share one normalizer, so this is a softmax of the candidate logits.
    maximum = max(logprobs)
    weights = [math.exp((value - maximum) / temperature) for value in logprobs]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def build_answer(
    kind: str, options: List[Tuple[str, str]], probabilities: List[float]
) -> JevAnswer:
    if not all(math.isfinite(p) for p in probabilities):
        raise RuntimeError("the readout produced non-finite probabilities")
    labels = [label for label, _ in options]
    by_label: Dict[str, float] = dict(zip(labels, probabilities))
    # Ties go to the smaller label, as in the official argmax.
    decision = min(labels, key=lambda label: (-by_label[label], label))
    if kind == "choice":
        return JevAnswer(
            type=kind,
            probabilities=by_label,
            decision=decision,
            confidence=choice_confidence(probabilities),
            choice=decision,
        )
    if kind == "noul":
        return JevAnswer(
            type=kind,
            probabilities=by_label,
            decision=decision,
            confidence=by_label[decision],
            noul=by_label["yes"],
        )
    return JevAnswer(
        type=kind,
        probabilities=by_label,
        decision=decision,
        confidence=score_confidence(probabilities),
        score=math.fsum(float(label) * p for label, p in by_label.items()),
        legend=dict(options),
    )
