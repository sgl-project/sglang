"""The prompts that a Kev checkpoint's pointer head reads.

Kev answers /v1/systemone questions from the backbone's hidden states instead
of from generated text. Each question is its own prompt, the shared state
followed by the question and its options between reserved delimiter tokens,
and the head scores each option at its closing token. This follows encode and
to_record in the checkpoint's reference code, kev/model.py and kev/api.py.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRequest,
)
from sglang.srt.layers.joint_schema_head import (
    QUESTION_TYPES,
    LayoutQuestion,
    pack_decision_layout,
)

# Rarely used Qwen tokens that the checkpoint reads as state, question, option,
# option end, and decide delimiters.
_DELIMITERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
# Text that looks like a delimiter is rewritten, so callers cannot forge one.
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
_ESCAPED = "<¦\\1¦>"
# The reference server's limits: the state with its delimiter, and one prompt.
MAX_STATE_TOKENS = 65536
MAX_ROW_TOKENS = MAX_STATE_TOKENS + 8192


def _render(value: Any, indent: int = 0) -> str:
    """Flattens JSON into the labeled text the checkpoint was trained on."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {_render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        (
            f"{pad}{key}:\n{_render(x, indent + 1)}"
            if isinstance(x, (dict, list))
            else f"{pad}{key}: {_render(x)}"
        )
        for key, x in value.items()
    )


def _option(name: str, description: Any) -> str:
    if description is None or description == "":
        return name
    return f"{name}: {_render(description)}"


def kev_options(question: SystemOneQuestion) -> List[str]:
    """Each option's text, in the order the head scores them."""
    if isinstance(question, SystemOneNoulQuestion):
        criteria = question.criteria
        return [
            _option("no", criteria.false if criteria else None),
            _option("yes", criteria.true if criteria else None),
        ]
    if isinstance(question, SystemOneChoiceQuestion):
        return [_option(name, detail) for name, detail in question.criteria.items()]
    return [_render(level) for level in question.criteria]


def encode_kev(
    tokenizer: Any, request: SystemOneRequest, max_prompt_tokens: int
) -> List[Tuple[List[int], List[int]]]:
    """Each question's prompt ids and decision layout, in request order."""

    def tokens(value: Any) -> List[int]:
        text = _SPECIAL.sub(_ESCAPED, _render(value))
        return tokenizer.encode(text, add_special_tokens=False)

    state_id, question_id, option_id, close_id, decide_id = (
        tokenizer.convert_tokens_to_ids(token) for token in _DELIMITERS
    )
    prefix = [state_id] + tokens(request.state)
    if len(prefix) > MAX_STATE_TOKENS:
        raise ValueError(
            f"the state is {len(prefix)} tokens with its delimiter, over the "
            f"{MAX_STATE_TOKENS} token limit of this checkpoint"
        )
    limit = min(MAX_ROW_TOKENS, max_prompt_tokens)
    rows = []
    for name, question in request.questions.items():
        input_ids = prefix + [question_id] + tokens(question.instructions)
        option_spans = []
        for option in kev_options(question):
            start = len(input_ids)
            input_ids += [option_id] + tokens(option) + [close_id]
            option_spans.append((start, len(input_ids)))
        input_ids.append(decide_id)
        if len(input_ids) > limit:
            raise ValueError(
                f"question {name!r} needs a {len(input_ids)} token prompt, but "
                f"this server scores at most {limit} in one prefill"
            )
        decide = len(input_ids) - 1
        layout = pack_decision_layout(
            len(input_ids),
            [
                LayoutQuestion(
                    QUESTION_TYPES.index(question.type),
                    (decide, decide + 1),
                    tuple(option_spans),
                )
            ],
        )
        rows.append((input_ids, layout))
    return rows
