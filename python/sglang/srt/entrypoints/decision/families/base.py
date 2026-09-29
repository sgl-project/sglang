"""What a decision model family provides to the family-agnostic Jev serving path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Protocol, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.decision.protocol import JevRequest


class DecisionInputError(ValueError):
    """A request the family cannot render, answered with 422 at `loc`."""

    def __init__(self, message: str, loc: Tuple[Any, ...] = ("body",)):
        super().__init__(message)
        self.loc = loc


class DecisionField(msgspec.Struct, frozen=True):
    name: str
    # Option labels and descriptions, in the order of candidate_ids.
    options: List[Tuple[str, str]]
    # The token that selects each option at the field's readout position.
    candidate_ids: List[int]


class DecisionPrompt(msgspec.Struct, frozen=True):
    input_ids: List[int]
    # In request order; each field is answered by the next-token scores at its position.
    fields: List[DecisionField]
    readout_positions: List[int]


class DecisionModelFamily(Protocol):
    name: str
    answer_symbols: str

    @staticmethod
    def detect(tokenizer: Any) -> bool:
        """Whether the served tokenizer belongs to a checkpoint of this family."""
        ...

    def __init__(self, tokenizer: Any) -> None: ...

    def validate(self, request: JevRequest) -> None:
        """Raise DecisionInputError for a request this family cannot render."""
        ...

    def encode(self, request: JevRequest) -> DecisionPrompt:
        """The prompt ids and readout positions of a validated request."""
        ...
