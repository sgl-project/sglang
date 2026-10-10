from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.decision.protocol import JevRequest


class DecisionInputError(ValueError):
    def __init__(self, message: str, loc: Tuple[Any, ...] = ("body",)):
        super().__init__(message)
        self.loc = loc


class DecisionField(msgspec.Struct, frozen=True):
    name: str
    options: List[Tuple[str, str]]
    candidate_ids: List[int]


class DecisionPrompt(msgspec.Struct, frozen=True):
    input_ids: Optional[List[int]]
    text: Optional[str]
    images: List[str]
    # Field i is read at the i-th anchor occurrence, so markers must follow field order.
    fields: List[DecisionField]
    readout_anchor: Tuple[int, int]


class DecisionModelFamily(Protocol):
    name: str
    answer_symbols: str

    @staticmethod
    def detect(tokenizer: Any) -> bool: ...

    def __init__(self, tokenizer: Any) -> None: ...

    def validate(self, request: JevRequest) -> None:
        """Raise DecisionInputError for a request this family cannot render."""
        ...

    def encode(self, request: JevRequest) -> DecisionPrompt: ...
