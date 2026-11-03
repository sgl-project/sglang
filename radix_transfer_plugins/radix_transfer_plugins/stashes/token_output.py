"""Per-token outputs of one LLM/VLM response."""

from __future__ import annotations

import re
from typing import Mapping, Optional

import msgspec
import numpy as np

from radix_transfer_plugins.schema import RawField

# (start, end) -> [num_tokens, end - start, top_k]: the half-open range [start, end) of
# a per-layer axis. A writer's blocks tile the axis; a reader may hold a subset.
LayerBlocks = dict[tuple[int, int], np.ndarray]


class TokenOutputStash(msgspec.Struct, frozen=True, kw_only=True):
    """Field names are the stored names, except that each block of a per-layer field is
    stored as "{name}.{start}-{end}". Dtypes follow what the writer produced and are
    recorded in the ref; today the first four are int32 and the logprobs float32."""

    # Blocks of layer ids
    routed_experts: Optional[LayerBlocks] = None
    # Blocks of indexer capturer slots, which are layer ids only on some models;
    # a split range that holds no slot has no block.
    indexer_topk: Optional[LayerBlocks] = None
    # [num_tokens]: how many allowed token ids each output token's sampling mask holds
    output_token_sampling_mask_lengths: Optional[np.ndarray] = None
    # [sum(lengths)]
    output_token_sampling_mask_token_ids: Optional[np.ndarray] = None
    # [sum(lengths)]: a logprob per allowed token; per-token sampled logprobs are not stored
    output_token_sampling_logprobs: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return all(value is None for value in msgspec.structs.astuple(self))

    def to_fields(self) -> dict[str, RawField]:
        fields = {}
        for name, value in msgspec.structs.asdict(self).items():
            if value is None:
                continue
            if name in _LAYER_BLOCK_FIELDS:
                for (start, end), array in value.items():
                    fields[f"{name}.{start}-{end}"] = _to_raw_field(array)
            else:
                fields[name] = _to_raw_field(value)
        return fields

    @classmethod
    def from_fields(cls, fields: Mapping[str, RawField]) -> TokenOutputStash:
        values, unknown = {}, []
        for stored, field in fields.items():
            block = _LAYER_BLOCK_NAME.fullmatch(stored)
            if block is not None:
                name, start, end = block.groups()
                values.setdefault(name, {})[(int(start), int(end))] = _from_raw_field(
                    field
                )
            elif stored in cls.__struct_fields__ and stored not in _LAYER_BLOCK_FIELDS:
                values[stored] = _from_raw_field(field)
            else:
                unknown.append(stored)
        if unknown:
            raise ValueError(f"unknown token output fields: {sorted(unknown)}")
        return cls(**values)


_LAYER_BLOCK_FIELDS = ("routed_experts", "indexer_topk")
_LAYER_BLOCK_NAME = re.compile(rf"({'|'.join(_LAYER_BLOCK_FIELDS)})\.(\d+)-(\d+)")


def _to_raw_field(array: np.ndarray) -> RawField:
    # Copies a block sliced out of a wider array; a whole contiguous array is not copied.
    array = np.ascontiguousarray(array)
    return RawField(
        dtype=array.dtype.name,
        shape=tuple(array.shape),
        data=memoryview(array.reshape(-1).view(np.uint8)),
    )


def _from_raw_field(field: RawField) -> np.ndarray:
    return np.frombuffer(field.data, dtype=np.dtype(field.dtype)).reshape(field.shape)
