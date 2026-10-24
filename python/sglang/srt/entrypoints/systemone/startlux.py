"""The prompt format and read-out of StartLux-Decision checkpoints.

A StartLux-Decision checkpoint answers a typed question by reading the next-token
probabilities of the option letters at the last prompt position, so nothing is
generated for a one-pass answer. The wording below is the format the checkpoints
were trained on: changing any of it changes the answers, which is why it is
rendered here rather than by the /v1/decisions prompt of PROMPT_FORMAT_VERSION.
"""

from __future__ import annotations

import json
import math
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Part of the trained format: the system message of every question.
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."

LETTERS = string.ascii_uppercase
QUESTION_TYPES = ("choice", "score", "noul")
# A rendered question kind to the checkpoint's temperature_by_type key. The yes or no
# question of this API is the checkpoint's "noul" type.
TEMPERATURE_KEYS = {"choice": "choice", "yes_no": "noul", "score": "score"}
# One letter per option, A to Z.
MAX_OPTIONS = 26
MAX_LEVELS = 10
# JSON arrays at least this long carry an "_index" on every element.
ANNOTATE_MIN = 8
NOUL_INSTRUCTIONS = "Which answer fits the evidence?"
# The extra last option of a choice question that also reads a none candidate.
NONE_OPTION = "None of the above"
# The end of every one-pass prompt, where the chat template switches thinking off.
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"

_BARE = re.compile(
    r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I
)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    """Long arrays carry their element index, so the model can refer to items by position."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [
                (
                    {"_index": i, **annotate_indices(v, min_len)}
                    if isinstance(v, dict)
                    else {"_index": i, "value": annotate_indices(v, min_len)}
                )
                for i, v in enumerate(x)
            ]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def state_text(state: Any) -> str:
    """The evidence as text: a string as it is, any other JSON value as compact JSON."""
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def value_text(value: Any) -> Optional[str]:
    """An option description or instruction as text, or None when it is empty."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.strip()
    return text or None


def _norm(value: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value)).strip().lower()


def score_keys(criteria: Dict[Any, Any]) -> List[Any]:
    """Level order of a score question: numeric keys ascending, otherwise as given."""
    try:
        return sorted(criteria, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(criteria)


def from_systemone(state: Any, spec: Dict[str, Any], allow_none: bool = False) -> Dict:
    """One /v1/systemone question -> the rendered question of the trained format.

    Score level ids are "0" to "n-1", lowest first. ``allow_none`` appends the extra
    last option NONE_OPTION to a choice question, which the read-out scores like any
    other option.
    """
    kind = spec.get("type", "choice")
    criteria = spec.get("criteria")
    question = value_text(spec.get("instructions")) or ""
    if kind == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = {str(option): None for option in criteria}
        if not isinstance(criteria, dict):
            raise ValueError(
                "a choice question needs criteria: {option: description or null} "
                "or a list of options"
            )
        options = [
            {"id": str(name), "criterion": value_text(description)}
            for name, description in criteria.items()
        ]
        if allow_none:
            if any(_norm(o["id"]) == _norm(NONE_OPTION) for o in options):
                raise ValueError(
                    f'reading the "{NONE_OPTION}" candidate adds an option of that '
                    "name itself: remove the option of that name, or leave the none "
                    "read-out out"
                )
            options.append({"id": NONE_OPTION, "criterion": None})
    elif kind == "score":
        if isinstance(criteria, dict):
            criteria = [criteria[key] for key in score_keys(criteria)]
        if not isinstance(criteria, (list, tuple)):
            raise ValueError(
                "a score question needs criteria: a list of levels, lowest first"
            )
        options = [
            {"id": str(i), "criterion": value_text(level)}
            for i, level in enumerate(criteria)
        ]
    elif kind == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        options = [
            {"id": "true", "criterion": value_text(criteria.get("true"))},
            {"id": "false", "criterion": value_text(criteria.get("false"))},
        ]
        if not question:
            question = NOUL_INSTRUCTIONS
    else:
        raise ValueError(f"unknown question type {kind!r}: use choice, noul or score")
    row = {
        "type": kind,
        "state": state_text(state),
        "instructions": question,
        "options": options,
    }
    if kind == "choice" and allow_none:
        row["none_option"] = True
    return row


def validate_row(row: Dict[str, Any]) -> List[str]:
    """Check a rendered question and return its option ids."""
    kind = row.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {kind!r}")
    if not isinstance(row.get("state"), str):
        raise ValueError("the state must be a string after rendering")
    if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
        raise ValueError("instructions are required for choice and score questions")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(
            f"a question needs 2 to {MAX_OPTIONS} options in one pass, got {options!r}"
        )
    ids = []
    for option in options:
        option_id = option.get("id")
        if (
            not isinstance(option_id, str)
            or not option_id.strip()
            or "\n" in option_id
            or "\r" in option_id
            or len(option_id) > 200
        ):
            raise ValueError(
                f"bad option id {option_id!r}: ids must be non-empty single-line "
                "strings of at most 200 characters"
            )
        criterion = option.get("criterion")
        if criterion is not None and not isinstance(criterion, str):
            raise ValueError("an option description must be a string or null")
        ids.append(option_id)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate option id")
    if kind == "noul" and set(ids) != {"true", "false"}:
        raise ValueError("yes/no options must be 'true' and 'false'")
    if kind == "score" and not 2 <= len(ids) <= MAX_LEVELS:
        raise ValueError(
            f"a score question needs 2 to {MAX_LEVELS} levels, got {len(ids)}"
        )
    return ids


def option_lines(row: Dict[str, Any]) -> List[str]:
    """The lettered option lines of a rendered question, in the order of its options."""
    order = [option["id"] for option in row["options"]]
    criteria = {option["id"]: option.get("criterion") for option in row["options"]}
    kind = row["type"]
    # Position of the extra none option, which the bare-id rule ignores.
    extra = len(order) - 1 if row.get("none_option") else None
    listed = order if extra is None else order[:extra]
    hide = (
        kind == "choice"
        and all(_BARE.match(name) for name in listed)
        and all(criteria[name] for name in listed)
    )
    lines = []
    for position, option_id in enumerate(order):
        name = ("yes" if option_id == "true" else "no") if kind == "noul" else option_id
        criterion = criteria[option_id]
        if hide and position != extra:
            body = criterion
        elif (
            criterion is None
            or not criterion.strip()
            or _norm(criterion) == _norm(name)
        ):
            body = name
        else:
            body = f"{name}: {criterion}"
        lines.append(f"{LETTERS[position]}) {body}")
    return lines


def messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """The chat messages, system then user turn, of a rendered question."""
    validate_row(row)
    state = row["state"] if row["state"].strip() else "(none)"
    content = (
        "Evidence:\n"
        + state
        + "\n\nQuestion: "
        + row["instructions"].strip()
        + "\nOptions:\n"
        + "\n".join(option_lines(row))
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def question_content(row: Dict[str, Any]) -> str:
    """The user message of a rendered question, for a chat template applied separately."""
    validate_row(row)
    return messages(row)[1]["content"]


def letter_token_ids(tokenizer: Any) -> List[int]:
    """Check that every letter is one token right after the thinking-off prompt, and return their ids."""
    probe = messages(
        {
            "type": "noul",
            "state": "s",
            "instructions": "q",
            "options": [
                {"id": "true", "criterion": None},
                {"id": "false", "criterion": None},
            ],
        }
    )
    text = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError(
            "the tokenizer's chat template lacks the thinking-off assistant prefix "
            f"{THINK_OFF_SUFFIX!r} this checkpoint was trained with"
        )
    ids = tokenizer.encode(text, add_special_tokens=False)
    letters = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if (
            len(encoded) != 1
            or tokenizer.encode(text + letter, add_special_tokens=False)
            != ids + encoded
        ):
            raise ValueError(
                f"the letter {letter} is not a separate single token after the prompt"
            )
        letters.append(encoded[0])
    if len(set(letters)) != len(LETTERS):
        raise ValueError("the letter token ids are not distinct")
    return letters


def probabilities(logprobs: Sequence[float], temperature: float) -> List[float]:
    """softmax(logprobs / temperature) over the options of one read-out."""
    top = max(logprobs)
    weights = [math.exp((value - top) / temperature) for value in logprobs]
    total = sum(weights)
    return [weight / total for weight in weights]


def split_none(p_read: Sequence[float]) -> Tuple[List[float], float, bool]:
    """Split a read-out whose last option is the none candidate.

    Returns the listed options renormalized to sum to 1, the none probability before
    that, and whether the none candidate is strictly more likely than every listed
    option (a tie goes to the listed option).
    """
    listed, none = list(p_read[:-1]), p_read[-1]
    total = sum(listed)
    renormalised = (
        [value / total for value in listed]
        if total > 0
        else [1.0 / len(listed)] * len(listed)
    )
    return renormalised, none, none > max(listed)


def choice_confidence(p_read: Sequence[float]) -> float:
    """How far the top probability sits above an even split, from 0 (even) to 1."""
    count = len(p_read)
    if count < 2:
        return 1.0
    return min(1.0, max(0.0, (count * max(p_read) - 1) / (count - 1)))


def score_confidence(p_read: Sequence[float]) -> float:
    """1 minus the distance in levels from the most likely one, relative to an even spread."""
    count = len(p_read)
    top = max(range(count), key=p_read.__getitem__)
    spread = sum(value * abs(i - top) for i, value in enumerate(p_read))
    even = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return max(0.0, 1 - spread / even) if even else 1.0
