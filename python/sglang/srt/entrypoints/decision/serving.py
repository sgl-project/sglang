from __future__ import annotations

import asyncio
import math
from http import HTTPStatus
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.decision.families import FAMILIES, detect_family
from sglang.srt.entrypoints.decision.families.base import (
    DecisionInputError,
    DecisionPrompt,
)
from sglang.srt.entrypoints.decision.images import normalize_images
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
    _choice_confidence,
    _score_confidence,
)
from sglang.srt.runtime_context import get_serving, get_spec

# Names the TypeSafe SDKs send by default, answered by the served checkpoint.
MODEL_ALIASES = ("jev-latest", "jev-preview")
_SUBJECT = "Decision model requests"


class TrainedDecisions(OpenAIServingBase):
    name = "trained"
    request_model = JevRequest

    def __init__(
        self, *, tokenizer_manager, validate_server: Callable[[str], Optional[str]]
    ):
        super().__init__(tokenizer_manager)
        self.family = detect_family(tokenizer_manager.tokenizer)
        self.validate_server = validate_server

    @staticmethod
    def detect(tokenizer) -> bool:
        return detect_family(tokenizer) is not None

    def _request_id_prefix(self) -> str:
        return "decision-model-"

    async def handle(self, request: JevRequest, raw_request: Request):
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
            if request.images:
                # Decoding up to 32 MiB of images would stall the event loop.
                images = await asyncio.to_thread(normalize_images, request.images)
                request = request.model_copy(update={"images": images})
            self.family.validate(request)
        except DecisionInputError as e:
            return _unprocessable(e)
        return await super().handle_request(request, raw_request)

    def _validate_request(self, request: JevRequest) -> Optional[str]:
        if self.family is None:
            names = ", ".join(family.name for family in FAMILIES)
            return (
                f"{_SUBJECT} require a checkpoint of a supported decision model "
                f"family ({names}), and the served model is not one"
            )
        error = self.validate_server(self.tokenizer_manager.served_model_name)
        if error is not None:
            return error
        if get_serving().chat_template is not None:
            return f"{_SUBJECT} render the checkpoint's own chat template, so --chat-template is not supported"
        # Marker readout needs prefill-only requests, which speculative decoding disables.
        if get_spec().speculative_algorithm is not None:
            return f"{_SUBJECT} do not support speculative decoding"
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
    ) -> Tuple[DecisionPrompt, JevRequest]:
        try:
            return self.family.encode(request), request
        except _CHAT_TEMPLATE_CLIENT_ERRORS as e:
            raise ValueError(f"the chat template failed: {e}") from e

    async def _handle_non_streaming_request(
        self,
        prompt: DecisionPrompt,
        request: JevRequest,
        raw_request: Request,
    ) -> ORJSONResponse:
        temperature = 1.0 if request.temperature is None else request.temperature
        result = await self.tokenizer_manager.score_readouts(
            input_ids=prompt.input_ids,
            text=prompt.text,
            image_data=prompt.images or None,
            readout_anchor=prompt.readout_anchor,
            label_token_ids=[field.candidate_ids for field in prompt.fields],
            temperature=temperature,
            request=raw_request,
        )
        if temperature != 1.0:
            error = _calibration_error(
                prompt.fields, result.token_logprobs, result.scores
            )
            if error is not None:
                return _unprocessable(error)
        answers = {
            field.name: build_answer(
                kind=request.questions[field.name].type,
                options=field.options,
                probabilities=probabilities,
            )
            for field, probabilities in zip(prompt.fields, result.scores)
        }
        response = JevResponse(
            model=self.tokenizer_manager.served_model_name,
            answers=answers,
            usage=JevUsage(
                input_tokens=result.prompt_tokens,
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


PROMPT_SOURCES = (TrainedDecisions,)


def _unprocessable(error: DecisionInputError) -> ORJSONResponse:
    detail = [
        {"type": "value_error", "loc": list(error.loc), "msg": f"Value error, {error}"}
    ]
    return ORJSONResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value, content={"detail": detail}
    )


def _calibration_error(fields, token_logprobs, scaled) -> Optional[DecisionInputError]:
    """Temperature scaling preserves the argmax in exact arithmetic; refuse when rounding breaks that."""
    for field, logprobs, probabilities in zip(fields, token_logprobs, scaled):
        labels = [label for label, _ in field.options]
        if _decide(labels, _uncalibrated(logprobs)) != _decide(labels, probabilities):
            return DecisionInputError(
                f"the temperature changes the decision of {field.name!r} through "
                "floating-point rounding",
                ("body", "temperature"),
            )
    return None


def _uncalibrated(logprobs: List[float]) -> List[float]:
    # Term for term the temperature-1 softmax of score_readouts.
    maximum = max(logprobs)
    weights = [math.exp(logprob - maximum) for logprob in logprobs]
    total = sum(weights)
    return [weight / total for weight in weights]


def _decide(labels: List[str], probabilities: List[float]) -> str:
    by_label = dict(zip(labels, probabilities))
    # Ties go to the smaller label, as in the official argmax.
    return min(labels, key=lambda label: (-by_label[label], label))


def build_answer(
    kind: str, options: List[Tuple[str, str]], probabilities: List[float]
) -> JevAnswer:
    if not all(math.isfinite(p) for p in probabilities):
        raise RuntimeError("the readout produced non-finite probabilities")
    labels = [label for label, _ in options]
    by_label: Dict[str, float] = dict(zip(labels, probabilities))
    decision = _decide(labels, probabilities)
    if kind == "choice":
        return JevAnswer(
            type=kind,
            probabilities=by_label,
            decision=decision,
            confidence=_choice_confidence(probabilities),
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
        confidence=_score_confidence(probabilities),
        score=math.fsum(float(label) * p for label, p in by_label.items()),
        legend=dict(options),
    )
