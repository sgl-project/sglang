"""What a decision model family provides to the family-agnostic Jev serving path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

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
    # Token ids of a text-only prompt, or the rendered text that the multimodal
    # processor expands around the images.
    input_ids: Optional[List[int]]
    text: Optional[str]
    # Data URLs, in placeholder order.
    images: List[str]
    # In request order; field i is answered at the i-th anchor readout position.
    fields: List[DecisionField]
    # (token_id, offset), resolved against the expanded prompt by the tokenizer manager.
    readout_anchor: Tuple[int, int]


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
        """The prompt and readout anchor of a validated request."""
        ...
