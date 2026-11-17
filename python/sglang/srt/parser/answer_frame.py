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
        self,
        content: dict,
        *,
        in_answer_block: bool,
        max_think_tokens: int,
        allows_tools: bool = False,
    ) -> dict:
        """XGrammar structural tag framing ``content``; ``max_think_tokens < 0``
        leaves thinking unbounded. With ``allows_tools`` the turn may end in tool
        calls instead of the answer."""
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
        self,
        content: dict,
        *,
        in_answer_block: bool,
        max_think_tokens: int,
        allows_tools: bool = False,
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


# Muse Glimmer / Muse Spark Open (shared tokenizer). The generation prompt ends at
# "<|start|>assistant"; the model writes " to=<recipient><|message|>" itself.
_MUSE_EOS, _MUSE_EOM, _MUSE_EOT, _MUSE_START, _MUSE_MESSAGE = (
    200001,
    200007,
    200008,
    200022,
    200023,
)
# <|start|>assistant to=user<|message|>
_MUSE_OPEN_ANSWER = [_MUSE_START, 140680, 328, 76976, _MUSE_MESSAGE]


def _muse_const(value: str) -> dict:
    return {"type": "const_string", "value": value}


def _muse_token(token_id: int) -> dict:
    return {"type": "token", "token": token_id}


# Recipients the frame handles itself; any other recipient is a tool.
_MUSE_ROLES = ("user", "self", "commentary")


def _muse_tool_recipient_ebnf() -> str:
    """`` to=<name>``: any recipient except a role, so a turn may end in tool
    calls but never reach the ``to=user`` answer unconstrained."""
    alts = []

    def walk(prefix: str) -> None:
        nexts = sorted(
            {
                w[len(prefix)]
                for w in _MUSE_ROLES
                if w.startswith(prefix) and w != prefix
            }
        )
        literal = json.dumps(prefix) + " " if prefix else ""
        alts.append(f"{literal}[^{''.join(nexts)} \\t\\n\\r<] rest")
        if prefix and prefix not in _MUSE_ROLES:
            alts.append(json.dumps(prefix))
        for c in nexts:
            walk(prefix + c)

    walk("")
    return (
        f'root ::= " to=" name\nname ::= {" | ".join(alts)}\nrest ::= [^ \\t\\n\\r<]*\n'
    )


class MuseAnswerFrame(AnswerFrame):
    """Any number of ``to=self`` (reasoning) and ``to=commentary`` (visible
    preamble) messages, then the ``to=user`` answer holding the constrained
    content or, when the request offers tools, tool calls; <|eot|> is a stop
    token and closes the turn."""

    framed_key_types = ("json", "regex", "ebnf")

    def framing_token_ids(self) -> List[int]:
        return [_MUSE_START, _MUSE_MESSAGE, _MUSE_EOM, _MUSE_EOT, _MUSE_EOS]

    def ends_in_open_answer_block(self, input_ids: Sequence[int]) -> bool:
        return list(input_ids[-len(_MUSE_OPEN_ANSWER) :]) == _MUSE_OPEN_ANSWER

    def structural_tag(
        self,
        content: dict,
        *,
        in_answer_block: bool,
        max_think_tokens: int,
        allows_tools: bool = False,
    ) -> dict:
        if in_answer_block:
            elements = [content]
        else:
            body = {
                "type": "any_tokens",
                "exclude_tokens": [_MUSE_START, _MUSE_MESSAGE, _MUSE_EOM, _MUSE_EOT],
            }
            if max_think_tokens >= 0:
                body["max_tokens"] = max_think_tokens
            private = {
                "type": "or",
                "elements": [_muse_const(" to=self"), _muse_const(" to=commentary")],
            }
            tool = {"type": "grammar", "grammar": _muse_tool_recipient_ebnf()}
            tool_body = {
                "type": "any_tokens",
                "exclude_tokens": [_MUSE_START, _MUSE_MESSAGE, _MUSE_EOM, _MUSE_EOT],
            }

            def message(recipient: dict, message_body: dict) -> list:
                return [recipient, _muse_token(_MUSE_MESSAGE), message_body]

            next_message = [
                _muse_token(_MUSE_EOM),
                _muse_token(_MUSE_START),
                _muse_const("assistant"),
            ]
            answer = {
                "type": "sequence",
                "elements": [
                    _muse_const(" to=user"),
                    _muse_token(_MUSE_MESSAGE),
                    content,
                ],
            }
            # A turn that calls tools ends in tool calls, with reasoning or
            # commentary allowed between them; only the answer is constrained.
            tool_calls = {
                "type": "sequence",
                "elements": [
                    *message(tool, tool_body),
                    {
                        "type": "star",
                        "content": {
                            "type": "sequence",
                            "elements": [
                                *next_message,
                                {"type": "or", "elements": [tool, private]},
                                _muse_token(_MUSE_MESSAGE),
                                tool_body,
                            ],
                        },
                    },
                ],
            }
            elements = [
                {
                    "type": "star",
                    "content": {
                        "type": "sequence",
                        "elements": [*message(private, body), *next_message],
                    },
                },
                {"type": "or", "elements": [answer, tool_calls]}
                if allows_tools
                else answer,
            ]
        return {
            "type": "structural_tag",
            "format": {"type": "sequence", "elements": elements},
        }

    def owns_turn(self, structural_tag: str) -> bool:
        return False
