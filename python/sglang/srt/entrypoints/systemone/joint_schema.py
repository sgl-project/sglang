"""The prompt that a Clef checkpoint's joint schema head reads.

Clef answers /v1/systemone questions from the backbone's hidden states instead
of from generated text. One prompt holds the state and every question with its
allowed options, and the head scores the span of each option. This follows
encode_record in the checkpoint's joint_schema_model.py, piece by piece, since
each piece is tokenized on its own.
"""

from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

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

_SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
_PREFIX = f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
_SUFFIX = (
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "JOINT SCHEMA DECISIONS:"
)
_NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def joint_schema_options(question: SystemOneQuestion) -> List[Tuple[str, Any]]:
    """Each option's name and description, in the order the head scores them."""
    if isinstance(question, SystemOneNoulQuestion):
        criteria = dict(_NOUL_CRITERIA)
        if question.criteria is not None:
            # A description sent as null drops the default, as in encode_record.
            criteria.update(question.criteria.model_dump(exclude_unset=True))
        return [(name, criteria[name]) for name in ("true", "false")]
    if isinstance(question, SystemOneChoiceQuestion):
        return sorted(question.criteria.items(), key=lambda item: item[0])
    return [(str(level), detail) for level, detail in enumerate(question.criteria)]


def encode_joint_schema(
    tokenizer: Any,
    request: SystemOneRequest,
    max_length: int,
    image_token_counts: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """The prompt ids and their decision layout.

    Each image's placeholder expands to its count in ``image_token_counts``, and
    the state is truncated so that the expanded prompt has at most
    ``max_length`` tokens, as encode_record budgets the expanded media.
    """

    def tokens(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    schema = tokens("\n\nSCHEMA FIELDS:\n")
    questions = []
    for index, (question_id, question) in enumerate(request.questions.items()):
        schema += tokens(
            f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {question.type}\n"
            "INSTRUCTION: "
        )
        instructions = question.instructions
        if instructions is None or instructions == "":
            instructions = question_id
        question_start = len(schema)
        schema += tokens(_render(instructions))
        question_span = (question_start, len(schema))
        schema += tokens("\nALLOWED OPTIONS:\n")
        option_spans = []
        for option_index, (name, description) in enumerate(
            joint_schema_options(question)
        ):
            schema += tokens(f"OPTION {option_index + 1}: ")
            option_start = len(schema)
            semantics = {"option_id": name}
            if description is not None:
                semantics["description"] = description
            schema += tokens(_render(semantics))
            option_spans.append((option_start, len(schema)))
            schema += tokens("\n")
        schema += tokens("END FIELD\n")
        questions.append(
            (QUESTION_TYPES.index(question.type), question_span, option_spans)
        )

    prefix = tokens(_PREFIX)
    if request.images:
        prefix += tokens(_IMAGE * len(request.images) + "\n")
    suffix = tokens(_SUFFIX)
    # Each image's single <|image_pad|> becomes that image's tokens.
    expansion = sum(count - 1 for count in image_token_counts)
    fixed_length = len(prefix) + expansion + len(schema) + len(suffix)
    if fixed_length > max_length:
        raise ValueError(
            f"the questions need {fixed_length} prompt tokens before the state, "
            f"but at most {max_length} fit the context length"
        )
    state = tokens(_render(request.state))[: max_length - fixed_length]
    offset = len(prefix) + len(state)
    input_ids = prefix + state + schema + suffix
    layout = pack_decision_layout(
        len(input_ids),
        [
            LayoutQuestion(
                question_type,
                (question_span[0] + offset, question_span[1] + offset),
                tuple((start + offset, end + offset) for start, end in option_spans),
            )
            for question_type, question_span, option_spans in questions
        ],
    )
    return input_ids, layout
