from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Union

from pydantic import model_validator

from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    _check_equal_lengths,
    _FrozenBase,
)


class SGLangSeqId(NamedTuple):
    rid: str


class PositionalSeqId(NamedTuple):
    step: int
    seq_index: int


SeqId = Union[SGLangSeqId, PositionalSeqId]


@dataclass(frozen=True)
class TokenAlignerStepAux:
    """Normalized auxiliary tensors for a single step (framework-agnostic)."""

    input_ids: list[int]  # [num_tokens]
    positions: list[int]  # [num_tokens]
    seq_lens: list[int]  # [num_seqs]
    seq_ids: list[SeqId]  # [num_seqs] — sequence identity

    def __post_init__(self) -> None:
        _check_equal_lengths(input_ids=self.input_ids, positions=self.positions)
        _check_equal_lengths(seq_lens=self.seq_lens, seq_ids=self.seq_ids)

        token_count: int = sum(self.seq_lens)
        if token_count != len(self.input_ids):
            raise ValueError(
                f"sum(seq_lens)={token_count} != len(input_ids)={len(self.input_ids)}"
            )


@dataclass(frozen=True)
class TokenAlignerGlobalAux:
    """Auxiliary tensors for one side across all steps + side-level metadata."""

    step_auxs: dict[int, TokenAlignerStepAux]
    framework: str  # "sglang" | "megatron"
    layout: TokenLayout
    thd_seq_lens_by_step: Optional[dict[int, list[int]]] = field(default=None)


class TokenLocator(_FrozenBase):
    """Locates tokens within a multi-step tensor store.

    token i is at tensor_of_step[steps[i]][token_index_in_step[i]].
    """

    steps: list[int]
    token_index_in_step: list[int]

    def __add__(self, other: TokenLocator) -> TokenLocator:
        return TokenLocator(
            steps=self.steps + other.steps,
            token_index_in_step=self.token_index_in_step + other.token_index_in_step,
        )


class TokenAlignerSeqInfo(_FrozenBase):
    """Information for a sequence, containing information to locate all the tokens inside the sequence."""

    # All these fields are of shape (num_tokens_in_seq,)
    input_ids: list[int]
    positions: list[int]
    locator: TokenLocator

    @model_validator(mode="after")
    def _validate_fields(self) -> TokenAlignerSeqInfo:
        n: int = len(self.input_ids)
        _check_equal_lengths(
            input_ids=self.input_ids,
            positions=self.positions,
            locator_steps=self.locator.steps,
            locator_token_index_in_step=self.locator.token_index_in_step,
        )

        if self.positions != list(range(n)):
            raise ValueError(
                f"positions must be [0, 1, ..., {n - 1}], got {self.positions}"
            )

        return self

    def __add__(self, other: TokenAlignerSeqInfo) -> TokenAlignerSeqInfo:
        return TokenAlignerSeqInfo(
            input_ids=self.input_ids + other.input_ids,
            positions=self.positions + other.positions,
            locator=self.locator + other.locator,
        )


class TokenAlignerSeqsInfo(_FrozenBase):
    """All sequences for one side across all steps."""

    sequences: dict[SeqId, TokenAlignerSeqInfo]
    layout: TokenLayout


class TokenAlignerPlan(_FrozenBase):
    """Token alignment plan. locators.x[i] and locators.y[i] correspond to the same logical token."""

    locators: Pair[TokenLocator]
    layouts: Pair[TokenLayout]

    @model_validator(mode="after")
    def _validate_fields(self) -> TokenAlignerPlan:
        _check_equal_lengths(
            locators_x_steps=self.locators.x.steps,
            locators_x_token_index_in_step=self.locators.x.token_index_in_step,
            locators_y_steps=self.locators.y.steps,
            locators_y_token_index_in_step=self.locators.y.token_index_in_step,
        )
        return self
