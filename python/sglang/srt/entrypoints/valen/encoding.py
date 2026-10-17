# SPDX-License-Identifier: Apache-2.0
"""Compile Valen's trained suffix after an already processed state context.

The media/tokenizer frontend owns the context. Passing its expanded token IDs
avoids processing images or videos twice and keeps all layout positions final.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import msgspec

from sglang.srt.layers.valen_layout import (
    ValenLayout,
    ValenQuestionLayout,
    validate_valen_layout,
)

NOUL_OPTIONS = (
    ("true", "True / 是：满足问题中的条件。"),
    ("false", "False / 否：不满足问题中的条件。"),
)


class ValenQuestion(msgspec.Struct, frozen=True):
    qid: str
    kind: str
    keys: tuple[str, ...]
    descriptions: tuple[str, ...]


class ValenEncodedPrompt(msgspec.Struct, frozen=True):
    input_ids: list[int]
    layout: ValenLayout
    questions: tuple[ValenQuestion, ...]


def _control_tokens(tokenizer):
    # HF's special-token map can omit controls that are registered as special
    # only in tokenizer.json (including Qwen's im_start). Consult the added-token
    # metadata as well, without rejecting ordinary, non-special added words.
    tokens = set(tokenizer.all_special_tokens)
    for token in getattr(tokenizer, "added_tokens_decoder", {}).values():
        if token.special:
            tokens.add(token.content)
    return tokens


def valen_options(question: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    if not isinstance(question, Mapping):
        raise ValueError("Valen questions must be objects")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Valen instructions must be nonempty text")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "noul":
        return NOUL_OPTIONS
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= 255:
            raise ValueError("Valen Choice needs 1–255 candidates")
        options = tuple(criteria.items())
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
            raise ValueError("Valen Score needs 2–255 ordered descriptions")
        options = tuple((str(i), text) for i, text in enumerate(criteria))
    else:
        raise ValueError("Unknown Valen question type")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(text, str)
        or not text.strip()
        for key, text in options
    ):
        raise ValueError("Valen candidate keys and descriptions must be nonempty text")
    return options


def encode_valen_schema(
    tokenizer: Any,
    context_ids: Sequence[int],
    questions: Mapping[str, Mapping[str, Any]],
    max_length: int = 32768,
) -> ValenEncodedPrompt:
    if not context_ids or any(
        type(token) is not int or token < 0 for token in context_ids
    ):
        raise ValueError("Valen context must contain processed token IDs")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Valen needs at least one question")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Valen max_length must be a positive integer")
    base_length, suffix = len(context_ids), []

    def append(text: str, span: tuple[int, int] | None = None):
        start = base_length + len(suffix)
        if span is None:
            suffix.extend(tokenizer.encode(text, add_special_tokens=False))
            return None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        suffix.extend(encoded["input_ids"])
        indices = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < span[1] and b > span[0]
        ]
        if not indices:
            raise ValueError("Empty Valen role token span")
        return start + indices[0], start + indices[-1] + 1

    metadata, layouts = [], []
    control_tokens = _control_tokens(tokenizer)
    append("<|im_start|>user\n")
    for index, (qid, question) in enumerate(questions.items()):
        options = valen_options(question)
        for value in (
            question["instructions"],
            *(v for option in options for v in option),
        ):
            if any(token in value for token in control_tokens):
                raise ValueError("Reserved tokenizer control token in Valen schema")
        prefix = f"Question {index + 1}\nTask: {question['type']}\nQuestion: "
        instruction = append(
            prefix + question["instructions"] + "\nCandidates:\n",
            (len(prefix), len(prefix) + len(question["instructions"])),
        )
        spans = []
        for key, description in options:
            prefix = key + ": " if question["type"] != "score" else ""
            spans.append(
                append(
                    prefix + description, (len(prefix), len(prefix) + len(description))
                )
            )
            append("\n")
        layouts.append(
            ValenQuestionLayout(question["type"], instruction, tuple(spans), -1)
        )
        metadata.append(
            ValenQuestion(
                qid,
                question["type"],
                tuple(k for k, _ in options),
                tuple(v for _, v in options),
            )
        )
    append("<|im_end|>\n<|im_start|>assistant\n")
    completed = []
    for index, layout in enumerate(layouts):
        append(f"Question {index + 1} Decision:")
        completed.append(
            ValenQuestionLayout(
                layout.kind,
                layout.instruction_span,
                layout.candidate_spans,
                base_length + len(suffix) - 1,
            )
        )
        append("\n")
    ids = list(context_ids) + suffix
    if len(ids) > max_length:
        raise ValueError(
            f"Valen prompt has {len(ids)} tokens, exceeds max_length={max_length}; no truncation"
        )
    layout = ValenLayout(len(ids), (0, base_length), tuple(completed))
    validate_valen_layout(layout, len(ids))
    return ValenEncodedPrompt(ids, layout, tuple(metadata))


def valen_messages(tokenizer, state, *, allow_media=False):
    """Normalize state without mutating it or accepting injected control tokens."""
    control_tokens = _control_tokens(tokenizer)
    if isinstance(state, str):
        messages = [{"role": "user", "content": state}]
    else:
        messages = deepcopy(state.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("Valen state needs nonempty messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError("Unsupported Valen message role")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if not isinstance(content, list):
            raise ValueError("Valen message content must be text or a content list")
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Valen content items must be objects")
            kind = item.get("type")
            if allow_media and kind in {"image_url", "video_url"}:
                source = item.get(kind)
                url = source.get("url") if isinstance(source, dict) else None
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(f"Valen {kind} needs a nonempty url")
                # HF consumes image/video items; the URL itself is not prompt text.
                item.clear()
                item.update(type=kind.removesuffix("_url"), url=url)
                continue
            if kind != "text":
                raise ValueError(
                    "Use the Valen multimodal processor for image/video content"
                )
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("Valen text content must be a string")
            if any(token in text for token in control_tokens):
                raise ValueError("Reserved tokenizer control token in Valen state")
    return messages


def encode_valen_text(tokenizer, state):
    """Match the training compiler's add_generation_prompt=False text context."""
    messages = valen_messages(tokenizer, state)
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
