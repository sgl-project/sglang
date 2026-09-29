"""The Intern-Decision family: checkpoints that ship a <decision> token, with the prompt of compile_row in internlm/Intern-Decision src/inputs/schema.py."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.decision.families.base import (
    DecisionField,
    DecisionInputError,
    DecisionPrompt,
)

if TYPE_CHECKING:
    from sglang.srt.entrypoints.decision.protocol import JevRequest

DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."
# The checkpoint renders its trained layout only with these template arguments.
CHAT_TEMPLATE_KWARGS = {
    "tokenize": False,
    "add_generation_prompt": False,
    "enable_thinking": False,
    "add_vision_id": True,
}


class CompiledDecision(msgspec.Struct, frozen=True):
    messages: List[Dict[str, Any]]
    fields: List[str]
    # Per field: the original option labels and their descriptions, in symbol order.
    options: Dict[str, List[Tuple[str, str]]]


def question_options(question: Mapping[str, Any]) -> List[Tuple[str, str]]:
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        return [(str(key), str(value)) for key, value in criteria.items()]
    descriptions = criteria if isinstance(criteria, Mapping) else {}
    yes = next(
        (
            str(descriptions[k])
            for k in descriptions
            if str(k).lower() in {"yes", "true", "1"}
        ),
        NOUL_YES,
    )
    no = next(
        (
            str(descriptions[k])
            for k in descriptions
            if str(k).lower() in {"no", "false", "0"}
        ),
        NOUL_NO,
    )
    return [("no", no), ("yes", yes)]


def compile_decision(
    state: Any, questions: Mapping[str, Mapping[str, Any]]
) -> CompiledDecision:
    """Chat messages whose assistant skeleton holds one decision marker per field, in field order."""
    fields: List[str] = []
    options: Dict[str, List[Tuple[str, str]]] = {}
    schema_lines: List[str] = []
    for field, question in questions.items():
        field_options = question_options(question)
        if not field_options:
            raise DecisionInputError(
                f"question {field!r} has no options", ("body", "questions", field)
            )
        if len(field_options) > len(ANSWER_SYMBOLS):
            raise DecisionInputError(
                f"supply 1 to {len(ANSWER_SYMBOLS)} options, the number of "
                f"single-token answer symbols, got {len(field_options)}",
                ("body", "questions", field, "criteria"),
            )
        fields.append(field)
        options[field] = field_options
        schema_lines.append(f"{field}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(ANSWER_SYMBOLS, field_options):
            schema_lines.append(f"    {symbol} = {value}: {description}")

    rendered_state = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=False)
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        "## State\n"
        f"{rendered_state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    if DECISION_TOKEN in user_text:
        raise DecisionInputError(
            f"the reserved decision marker {DECISION_TOKEN} appears in the request"
        )
    skeleton = json.dumps(
        dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return CompiledDecision(messages=messages, fields=fields, options=options)


class InternDecisionFamily:
    name = "Intern-Decision"
    answer_symbols = ANSWER_SYMBOLS

    @staticmethod
    def detect(tokenizer: Any) -> bool:
        return (
            isinstance(tokenizer, PreTrainedTokenizerBase)
            and DECISION_TOKEN in tokenizer.get_added_vocab()
            and _symbol_ids(tokenizer) is not None
        )

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.marker_id: int = tokenizer.get_added_vocab()[DECISION_TOKEN]
        self.symbol_ids: List[int] = _symbol_ids(tokenizer)

    def validate(self, request: JevRequest) -> None:
        _compile(request)

    def encode(self, request: JevRequest) -> DecisionPrompt:
        compiled = _compile(request)
        text = self.tokenizer.apply_chat_template(
            compiled.messages, **CHAT_TEMPLATE_KWARGS
        )
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # The distribution predicting a marker slot is read one position before it.
        positions = [
            i - 1 for i, token in enumerate(input_ids) if token == self.marker_id
        ]
        if len(positions) != len(compiled.fields) or min(positions) < 0:
            raise ValueError(
                f"the rendered prompt has {len(positions)} decision markers for "
                f"{len(compiled.fields)} fields"
            )
        fields = [
            DecisionField(
                name=name,
                options=compiled.options[name],
                candidate_ids=self.symbol_ids[: len(compiled.options[name])],
            )
            for name in compiled.fields
        ]
        return DecisionPrompt(
            input_ids=input_ids, fields=fields, readout_positions=positions
        )


def _compile(request: JevRequest) -> CompiledDecision:
    questions = {
        name: question.model_dump(include={"type", "instructions", "criteria"})
        for name, question in request.questions.items()
    }
    return compile_decision(request.state, questions)


def _symbol_ids(tokenizer: PreTrainedTokenizerBase) -> Optional[List[int]]:
    encoded = [tokenizer.encode(s, add_special_tokens=False) for s in ANSWER_SYMBOLS]
    if any(len(ids) != 1 for ids in encoded):
        return None
    return [ids[0] for ids in encoded]
