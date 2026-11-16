import functools
import json
from typing import List, Sequence, Tuple

from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
)


class AnswerFrame:
    """Native message framing a reasoning format wraps around constrained output.

    A frame owns the whole turn: an optional thinking message, then the answer
    message holding the constrained content.
    """

    framed_key_types: Tuple[str, ...] = ()

    def framing_token_ids(self) -> List[int]:
        """Token IDs that may only match as frame tokens, never as content text."""
        raise NotImplementedError()

    def ends_in_open_answer_block(self, input_ids: Sequence[int]) -> bool:
        """Whether the prompt already opened the answer block (continuation)."""
        raise NotImplementedError()

    def structural_tag(
        self, content: dict, *, in_answer_block: bool, max_think_tokens: int
    ) -> dict:
        """XGrammar structural tag framing ``content``; ``max_think_tokens < 0``
        leaves thinking unbounded."""
        raise NotImplementedError()

    def owns_turn(self, structural_tag: str) -> bool:
        """Whether this structural tag already frames the whole turn itself."""
        raise NotImplementedError()


@functools.cache
def _canonical_inkling_auto_tool_schema() -> dict:
    # Lazy: inkling_detector imports xgrammar unconditionally, and XPU
    # installs may not ship it.
    from sglang.srt.function_call.inkling_detector import InklingDetector

    return (
        InklingDetector().get_auto_tool_call_structural_tag().model_dump(by_alias=True)
    )


def _token(name: str) -> dict:
    return {"type": "token", "token": INKLING_SPECIAL_TOKEN_IDS[name]}


class InklingAnswerFrame(AnswerFrame):
    framed_key_types = ("json", "regex", "ebnf")

    def framing_token_ids(self) -> List[int]:
        # Inkling's control tokens decode to their literal spellings, so text
        # positions such as JSON strings would otherwise admit <|end_message|>.
        return [
            token_id
            for token, token_id in INKLING_SPECIAL_TOKEN_IDS.items()
            if token != CONTENT_MODEL_END_SAMPLING
        ]

    def ends_in_open_answer_block(self, input_ids: Sequence[int]) -> bool:
        end_message = INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE]
        last_end = next(
            (
                i
                for i in range(len(input_ids) - 1, -1, -1)
                if input_ids[i] == end_message
            ),
            -1,
        )
        return list(input_ids[last_end + 1 : last_end + 3]) == [
            INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL],
            INKLING_SPECIAL_TOKEN_IDS[CONTENT_TEXT],
        ]

    def structural_tag(
        self, content: dict, *, in_answer_block: bool, max_think_tokens: int
    ) -> dict:
        elements = [content, _token(END_MESSAGE)]
        if in_answer_block:
            return {
                "type": "structural_tag",
                "format": {"type": "sequence", "elements": elements},
            }
        thinking = {
            "type": "any_tokens",
            "exclude_tokens": [INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE]],
        }
        if max_think_tokens >= 0:
            thinking["max_tokens"] = max_think_tokens
        thinking_message = {
            "type": "sequence",
            "elements": [
                _token(MESSAGE_MODEL),
                _token(CONTENT_THINKING),
                thinking,
                _token(END_MESSAGE),
            ],
        }
        elements = [
            {"type": "optional", "content": thinking_message},
            _token(MESSAGE_MODEL),
            _token(CONTENT_TEXT),
            *elements,
        ]
        return {
            "type": "structural_tag",
            "format": {"type": "sequence", "elements": elements},
        }

    def owns_turn(self, structural_tag: str) -> bool:
        # Only the canonical auto-tool schema owns the full turn; user restrictions may not.
        return json.loads(structural_tag) == _canonical_inkling_auto_tool_schema()
