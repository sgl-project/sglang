import json
from functools import lru_cache

import torch
from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    GrammarFormat,
    SequenceFormat,
    StarFormat,
    TokenFormat,
)

from sglang.srt.constrained.base_grammar_backend import (
    InvalidGrammarObject,
    _grammar_key_contains_nul,
)
from sglang.srt.constrained.json_schema_validation import validate_xgrammar_json_schema
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.environ import envs
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    INKLING_CONTROL_TOKENS,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
)

_TEXT = INKLING_SPECIAL_TOKEN_IDS[CONTENT_TEXT]
_END = INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE]
_PROTOCOL_IDS = frozenset(INKLING_SPECIAL_TOKEN_IDS.values())


@lru_cache(maxsize=128)
def json_response_grammar(schema: str, *, continue_final_message: bool = False) -> str:
    if _grammar_key_contains_nul("json", schema):
        raise ValueError("NUL bytes (\\u0000) are not allowed in JSON schemas")
    validate_xgrammar_json_schema(json.loads(schema))
    try:
        json_grammar = Grammar.from_json_schema(
            schema, any_whitespace=True, max_whitespace_cnt=16
        )
    except RuntimeError as error:
        raise ValueError(f"Invalid JSON schema: {error}") from error
    body = GrammarFormat(grammar=str(json_grammar))
    end = TokenFormat(token=INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE])
    final = SequenceFormat(elements=[body, end])
    if not continue_final_message:
        message = TokenFormat(token=INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL])
        thinking = SequenceFormat(
            elements=[
                message,
                TokenFormat(token=INKLING_SPECIAL_TOKEN_IDS[CONTENT_THINKING]),
                AnyTextFormat(excludes=sorted(INKLING_CONTROL_TOKENS)),
                end,
            ]
        )
        final = SequenceFormat(
            elements=[
                StarFormat(content=thinking),
                message,
                TokenFormat(token=INKLING_SPECIAL_TOKEN_IDS[CONTENT_TEXT]),
                final,
            ]
        )
    return str(Grammar.from_structural_tag(StructuralTag(format=final)))


def constrain_json_response(
    sampling_params: dict,
    *,
    assistant_prefix: str | None = None,
    input_ids: list[int] | None = None,
    enable_strict_thinking: bool = False,
    grammar_backend: str = "xgrammar",
    separate_reasoning: bool = True,
) -> None:
    schema = sampling_params.get("json_schema")
    if schema is None:
        return
    if grammar_backend != "xgrammar":
        raise ValueError(
            "Inkling JSON response formats require --grammar-backend xgrammar."
        )
    if not separate_reasoning:
        raise ValueError(
            "Inkling JSON response formats require separate_reasoning=true."
        )
    custom_params = sampling_params.get("custom_params")
    if custom_params is not None and "thinking_budget" in custom_params:
        raise ValueError(
            "Inkling JSON response formats do not support thinking_budget."
        )
    if enable_strict_thinking and envs.SGLANG_MAX_THINK_TOKENS.get() >= 0:
        raise ValueError(
            "Inkling JSON response formats do not support the server's "
            "SGLANG_MAX_THINK_TOKENS limit."
        )

    if any(
        sampling_params.get(key) is not None
        for key in ("regex", "ebnf", "structural_tag")
    ):
        raise ValueError(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set."
        )
    if assistant_prefix:
        raise ValueError(
            "Inkling JSON response formats cannot continue a nonempty assistant "
            "message. Use an empty assistant message or start a new response."
        )
    if input_ids is not None:
        final_header = [
            INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL],
            INKLING_SPECIAL_TOKEN_IDS[CONTENT_TEXT],
        ]
        if input_ids[-2:] == final_header:
            assistant_prefix = ""
        elif assistant_prefix is not None or not (
            not input_ids
            or input_ids[-1] == INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE]
            or input_ids[-2:]
            == [
                INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
                INKLING_SPECIAL_TOKEN_IDS[CONTENT_MODEL_END_SAMPLING],
            ]
        ):
            raise ValueError(
                "Inkling JSON response formats require input_ids ending at a "
                "closed message or an empty assistant text header."
            )

    sampling_params["custom_params"] = {
        **(custom_params or {}),
        "__sglang_inkling_json": assistant_prefix is not None,
    }


def compile_json_response(backend, key_string: str):
    schema, continuation = json.loads(key_string)
    if not isinstance(schema, str) or type(continuation) is not bool:
        raise ValueError("Invalid Inkling JSON grammar key")
    try:
        grammar = json_response_grammar(schema, continue_final_message=continuation)
    except ValueError as error:
        return InvalidGrammarObject(str(error))
    result = backend.dispatch_full_assistant_ebnf(grammar)
    if isinstance(result, InvalidGrammarObject):
        return result
    return InklingJsonGrammar(
        result,
        backend.tokenizer_info.decoded_vocab,
        continue_final_message=continuation,
    )


class InklingJsonGrammar(XGrammarGrammar):
    def __init__(
        self,
        grammar: XGrammarGrammar,
        decoded_vocab: list[bytes],
        *,
        continue_final_message: bool = False,
    ) -> None:
        super().__init__(
            grammar.matcher,
            grammar.vocab_size,
            grammar.ctx,
            grammar.override_stop_tokens,
            grammar.key_string,
            grammar.grammar_stats,
        )
        self.decoded_vocab = decoded_vocab
        self.continue_final_message = continue_final_message
        self._json_states = [(continue_final_message, False, False)]

    def accept_token(self, token: int) -> None:
        if self.is_terminated():
            return
        in_json, in_string, escaped = self._json_states[-1]
        if in_string and token in _PROTOCOL_IDS:
            raise ValueError("Inkling framing tokens cannot appear inside JSON strings")
        super().accept_token(token)
        if token == _TEXT:
            in_json = True
        elif token == _END:
            in_json = False
        elif in_json:
            for byte in self.decoded_vocab[token]:
                if escaped:
                    escaped = False
                elif byte == ord('"'):
                    in_string = not in_string
                elif in_string and byte == ord("\\"):
                    escaped = True
        self._json_states.append((in_json, in_string, escaped))

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        super().fill_vocab_mask(vocab_mask, idx)
        if self._json_states[-1][1]:
            set_token_filter_torch(
                vocab_mask,
                list(_PROTOCOL_IDS),
                idx,
                is_allowed=False,
                reset_vocab_mask=False,
            )

    def rollback(self, k: int) -> None:
        if k == 0:
            return
        if not 0 < k < len(self._json_states):
            raise ValueError("Cannot roll back beyond accepted tokens")
        super().rollback(k)
        del self._json_states[-k:]

    def copy(self) -> "InklingJsonGrammar":
        return InklingJsonGrammar(
            super().copy(),
            self.decoded_vocab,
            continue_final_message=self.continue_final_message,
        )

    def try_jump_forward(self, tokenizer):
        return None

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
        raise RuntimeError("Inkling JSON grammars do not support jump-forward")
