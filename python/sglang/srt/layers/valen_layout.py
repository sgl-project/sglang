# SPDX-License-Identifier: Apache-2.0
"""Versioned, request-local readouts for Valen shared-state inference."""

from __future__ import annotations

from typing import Sequence, Tuple

import msgspec

QUESTION_TYPES = ("choice", "noul", "score")
LAYOUT_VERSION = 1


class ValenQuestionLayout(msgspec.Struct, frozen=True):
    kind: str
    instruction_span: Tuple[int, int]
    candidate_spans: Tuple[Tuple[int, int], ...]
    decision_position: int


class ValenLayout(msgspec.Struct, frozen=True):
    prompt_length: int
    context_span: Tuple[int, int]
    questions: Tuple[ValenQuestionLayout, ...]


def validate_valen_layout(layout: ValenLayout, num_tokens: int) -> None:
    """Reject truncated, overlapping or mismatched layouts before scheduling.

    Positions address the final, expanded sequence. Unlike Clef's layout,
    the context span includes media tokens and must never be shifted whole.
    """
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError("Valen needs a nonempty prompt")
    if layout.prompt_length != num_tokens:
        raise ValueError("Valen layout does not match the final prompt length")
    start, end = layout.context_span
    if start != 0 or not 0 < end < num_tokens:
        raise ValueError("Invalid Valen context span")
    if not layout.questions or len(layout.questions) > num_tokens:
        raise ValueError("Invalid Valen question count")
    schema_end = end
    for question in layout.questions:
        if question.kind not in QUESTION_TYPES:
            raise ValueError("Unknown Valen question type")
        count = len(question.candidate_spans)
        minimum = 1 if question.kind == "choice" else 2
        if not minimum <= count <= 255 or (question.kind == "noul" and count != 2):
            raise ValueError("Invalid Valen candidate count")
        a, b = question.instruction_span
        if not schema_end <= a < b <= num_tokens:
            raise ValueError("Invalid or overlapping Valen instruction span")
        schema_end = b
        for a, b in question.candidate_spans:
            if not schema_end <= a < b <= num_tokens:
                raise ValueError("Invalid or overlapping Valen candidate span")
            schema_end = b
    last_decision = schema_end - 1
    for question in layout.questions:
        if not last_decision < question.decision_position < num_tokens:
            raise ValueError("Valen readouts must follow all schemas in order")
        last_decision = question.decision_position


def pack_valen_layout(layout: ValenLayout) -> list[int]:
    """Compact IPC representation; independent of Clef's decision_layout."""
    validate_valen_layout(layout, layout.prompt_length)
    values = [
        LAYOUT_VERSION,
        layout.prompt_length,
        *layout.context_span,
        len(layout.questions),
    ]
    for question in layout.questions:
        values.extend(
            (
                QUESTION_TYPES.index(question.kind),
                *question.instruction_span,
                question.decision_position,
                len(question.candidate_spans),
            )
        )
        for span in question.candidate_spans:
            values.extend(span)
    return values


def parse_valen_layout(values: Sequence[int], num_tokens: int) -> ValenLayout:
    if len(values) < 5 or any(type(value) is not int for value in values):
        raise ValueError("Valen layout must be a list of integers")
    version, prompt_length, start, end, count = values[:5]
    if version != LAYOUT_VERSION:
        raise ValueError("Unsupported Valen layout version")
    if prompt_length != num_tokens or not 1 <= count <= num_tokens:
        raise ValueError("Valen layout does not match the final prompt length")
    cursor, questions = 5, []
    for _ in range(count):
        if cursor + 5 > len(values):
            raise ValueError("Truncated Valen question layout")
        kind, a, b, decision, options = values[cursor : cursor + 5]
        cursor += 5
        if not 0 <= kind < len(QUESTION_TYPES) or not 1 <= options <= 255:
            raise ValueError("Invalid Valen question layout")
        if cursor + 2 * options > len(values):
            raise ValueError("Truncated Valen candidate layout")
        spans = tuple(
            (values[i], values[i + 1]) for i in range(cursor, cursor + 2 * options, 2)
        )
        cursor += 2 * options
        questions.append(
            ValenQuestionLayout(QUESTION_TYPES[kind], (a, b), spans, decision)
        )
    if cursor != len(values):
        raise ValueError("Valen layout has trailing values")
    layout = ValenLayout(prompt_length, (start, end), tuple(questions))
    validate_valen_layout(layout, num_tokens)
    return layout


def rebase_valen_context(layout: ValenLayout, context_length: int) -> ValenLayout:
    """Expand a prefix context without shifting its start or pooling padding.

    Call only when preprocessing changes the prefix and preserves the schema
    suffix token-for-token. This is not a replacement for video/MRoPE validation.
    """
    validate_valen_layout(layout, layout.prompt_length)
    if type(context_length) is not int or context_length < layout.context_span[1]:
        raise ValueError("Valen context expansion cannot remove tokens")
    shift = context_length - layout.context_span[1]
    questions = tuple(
        ValenQuestionLayout(
            q.kind,
            tuple(p + shift for p in q.instruction_span),
            tuple((a + shift, b + shift) for a, b in q.candidate_spans),
            q.decision_position + shift,
        )
        for q in layout.questions
    )
    result = ValenLayout(layout.prompt_length + shift, (0, context_length), questions)
    validate_valen_layout(result, result.prompt_length)
    return result
