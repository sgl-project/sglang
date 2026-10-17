"""Per-token outputs of one LLM/VLM response."""

from __future__ import annotations

from typing import Mapping, Optional

import msgspec
import numpy as np

from radix_transfer_plugins.schema import RawField


class TokenOutputStash(msgspec.Struct, frozen=True, kw_only=True):
    """Field names are the stored names. Dtypes follow what the writer produced and are
    recorded in the ref; today the first four are int32 and the logprobs float32."""

    # [num_tokens, num_layers, top_k]
    routed_experts: Optional[np.ndarray] = None
    # [num_tokens, num_layers, top_k]
    indexer_topk: Optional[np.ndarray] = None
    # [num_tokens]: how many allowed token ids each output token's sampling mask holds
    output_token_sampling_mask_lengths: Optional[np.ndarray] = None
    # [sum(lengths)]
    output_token_sampling_mask_token_ids: Optional[np.ndarray] = None
    # [sum(lengths)] when every allowed token has a logprob, else [num_tokens]
    output_token_sampling_logprobs: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return all(value is None for value in self._arrays().values())

    def to_fields(self) -> dict[str, RawField]:
        return {
            name: _to_raw_field(array)
            for name, array in self._arrays().items()
            if array is not None
        }

    @classmethod
    def from_fields(cls, fields: Mapping[str, RawField]) -> TokenOutputStash:
        unknown = set(fields) - set(cls.__struct_fields__)
        if unknown:
            raise ValueError(f"unknown token output fields: {sorted(unknown)}")
        return cls(**{name: _from_raw_field(field) for name, field in fields.items()})

    def _arrays(self) -> dict[str, Optional[np.ndarray]]:
        return msgspec.structs.asdict(self)


def _to_raw_field(array: np.ndarray) -> RawField:
    array = np.ascontiguousarray(array)
    return RawField(
        dtype=array.dtype.name,
        shape=tuple(array.shape),
        data=memoryview(array.reshape(-1).view(np.uint8)),
    )


def _from_raw_field(field: RawField) -> np.ndarray:
    return np.frombuffer(field.data, dtype=np.dtype(field.dtype)).reshape(field.shape)
