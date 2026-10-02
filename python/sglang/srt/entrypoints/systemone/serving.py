"""Handler for the System One compatible decision API, on the /v1/decisions scoring path."""

from __future__ import annotations

import math
import string
from typing import Any, Dict, Iterator, List, Optional, Tuple

import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.serving_decisions import (
    OpenAIServingDecisions,
    QuestionView,
    default_labels,
    label_context,
    label_mass,
    label_token_id,
    render_text,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceAnswer,
    SystemOneChoiceQuestion,
    SystemOneNoulAnswer,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneUsage,
)

# Beyond A to Z, every option gets a two-letter label, in this fixed order.
_PAIR_LABELS = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


class SystemOneServing(OpenAIServingDecisions):
    """Answers System One questions with the rendering, label checks, and scoring of /v1/decisions."""

    route = "/v1/systemone"

    def _request_id_prefix(self) -> str:
        return "systemone-"

    def _validate_request(self, request: SystemOneRequest) -> Optional[str]:
        return self._validate_server(request.model) or self._validate_reasoning(
            request.chat_template_kwargs
        )

    def _convert_to_internal_request(
        self,
        request: SystemOneRequest,
        raw_request: Request = None,
    ) -> Tuple[
        Iterator[Tuple[List[int], List[int]]],
        Tuple[SystemOneRequest, List[QuestionView]],
    ]:
        views = [_view(question) for question in request.questions.values()]
        # Lazy, so the async handler can yield to other requests between questions.
        return self._encoded_systemone_questions(request, views), (request, views)

    def _encoded_systemone_questions(
        self, request: SystemOneRequest, views: List[QuestionView]
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Prompt and label ids for each question, in request order."""
        text = render_text(request.state)
        chat_template_kwargs = self._chat_template_kwargs(request.chat_template_kwargs)
        pair_labels = None
        for question_id, view in zip(request.questions, views):
            try:
                labels = default_labels(view)
                if view.kind == "choice" and len(view.names) > len(labels):
                    if pair_labels is None:
                        pair_labels = self._pair_labels(chat_template_kwargs)
                    if len(view.names) > len(pair_labels):
                        raise ValueError(
                            f"it has {len(view.names)} options, but the served "
                            "tokenizer and chat template can label at most "
                            f"{max(len(pair_labels), len(string.ascii_uppercase))}"
                        )
                    labels = pair_labels[: len(view.names)]
                if view.kind == "score":
                    _check_legend(question_id, view)
                encoded = self._encode_question(
                    text=text,
                    view=view,
                    labels=labels,
                    chat_template_kwargs=chat_template_kwargs,
                )
            except ValueError as e:
                raise ValueError(f"question {question_id!r}: {e}") from e
            yield encoded

    def _pair_labels(self, chat_template_kwargs: Dict[str, Any]) -> List[str]:
        """Two-letter labels that are distinct single tokens at the answer position."""
        if not self.added_tokens:
            raise ValueError(
                "more than 26 options needs added tokens the server can read "
                "from the tokenizer, and the served tokenizer reports none"
            )
        tokenizer = self.tokenizer_manager.tokenizer
        contexts = []
        for message in ("x", "y"):
            prompt = self._apply_chat_template(message, chat_template_kwargs)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            contexts.append(
                label_context(tokenizer, prompt, prompt_ids, self.added_tokens)
            )
        (text, text_ids, shortcut), other = contexts
        # Labels must be checked on text that excludes the message. Otherwise
        # every label re-encodes the whole input on the event loop, which is
        # unbounded work for hundreds of options.
        if not (shortcut and other[2] and other[0] == text):
            raise ValueError(
                "more than 26 options needs an added token between the message "
                "and the answer position, after which the text tokenizes the same on "
                "its own, which this tokenizer and chat template do not provide"
            )
        labels, label_ids = [], set()
        for label in _PAIR_LABELS:
            token_id = label_token_id(tokenizer, text, text_ids, label)
            if token_id is not None and token_id not in label_ids:
                labels.append(label)
                label_ids.add(token_id)
        # Each final prompt is still checked as a whole in _encode_question.
        return labels

    async def _handle_non_streaming_request(
        self,
        adapted_request: Iterator[Tuple[List[int], List[int]]],
        processed: Tuple[SystemOneRequest, List[QuestionView]],
        raw_request: Request,
    ) -> ORJSONResponse:
        request, views = processed
        _, _, result = await self._score(
            adapted_request=adapted_request, raw_request=raw_request
        )
        answers = {}
        for i, question_id in enumerate(request.questions):
            answers[question_id] = _answer(
                view=views[i],
                probabilities=result.scores[i],
                mass=label_mass(result.token_logprobs[i]),
                question_id=question_id,
            )
        response = SystemOneResponse(
            # The served model answered, whatever name the request used.
            model=self.tokenizer_manager.served_model_name,
            answers=answers,
            usage=SystemOneUsage(input_tokens=result.prompt_tokens),
        )
        return ORJSONResponse(content=response.model_dump())


def _view(question: SystemOneQuestion) -> QuestionView:
    if isinstance(question, SystemOneChoiceQuestion):
        return QuestionView(
            kind="choice",
            question=question.instructions,
            names=list(question.criteria),
            details=list(question.criteria.values()),
        )
    if isinstance(question, SystemOneNoulQuestion):
        criteria = question.criteria
        return QuestionView(
            kind="yes_no",
            question=question.instructions,
            names=["yes", "no"],
            details=[
                criteria.true if criteria else None,
                criteria.false if criteria else None,
            ],
        )
    return QuestionView(
        kind="score",
        question=question.instructions,
        names=[str(level) for level in range(len(question.criteria))],
        details=list(question.criteria),
    )


def _answer(
    view: QuestionView,
    probabilities: List[float],
    mass: float,
    question_id: str,
):
    if not all(math.isfinite(value) for value in [*probabilities, mass]):
        # A server fault, reported as 500 rather than as a client error.
        raise RuntimeError(f"question {question_id!r} scored non-finite values")
    # Reported as scored, like /v1/decisions, and normalized only for confidence.
    if view.kind == "yes_no":
        return SystemOneNoulAnswer(noul=probabilities[0], x_label_mass=mass)
    probabilities_by_name = dict(zip(view.names, probabilities))
    if view.kind == "choice":
        return SystemOneChoiceAnswer(
            choice=view.names[probabilities.index(max(probabilities))],
            confidence=_choice_confidence(_normalized(probabilities)),
            probabilities=probabilities_by_name,
            x_label_mass=mass,
        )
    return SystemOneScoreAnswer(
        score=math.fsum(i * p for i, p in enumerate(probabilities)),
        confidence=_score_confidence(_normalized(probabilities)),
        legend=_legend(view),
        probabilities=probabilities_by_name,
        x_label_mass=mass,
    )


def _legend(view: QuestionView) -> Dict[str, Any]:
    return dict(zip(view.names, view.details))


def _check_legend(question_id: str, view: QuestionView) -> None:
    """Refuse, before scoring, levels that the response cannot echo in its legend."""
    answer = SystemOneScoreAnswer(
        score=0.0,
        confidence=0.0,
        legend=_legend(view),
        probabilities={},
        x_label_mass=0.0,
    )
    response = SystemOneResponse(
        model="", answers={question_id: answer}, usage=SystemOneUsage(input_tokens=0)
    )
    try:
        # The same encoder and options as the real response.
        ORJSONResponse(content=response.model_dump())
    except orjson.JSONEncodeError as e:
        raise ValueError(f"a level cannot be returned in the legend: {e}") from e


def _normalized(probabilities: List[float]) -> List[float]:
    total = math.fsum(probabilities)
    if total <= 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [p / total for p in probabilities]


def _choice_confidence(q: List[float]) -> float:
    """How far the top option stands above a uniform guess, from 0 to 1."""
    n = len(q)
    if n == 1:
        return 1.0
    return min(1.0, max(0.0, (n * max(q) - 1) / (n - 1)))


def _score_confidence(q: List[float]) -> float:
    """One minus the spread around the top level relative to a uniform spread, floored at 0."""
    n = len(q)
    if n == 1:
        return 1.0
    top = q.index(max(q))
    spread = math.fsum(p * abs(i - top) for i, p in enumerate(q))
    uniform_spread = math.fsum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform_spread)
