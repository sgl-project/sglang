# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trie-conditioned candidate scoring for beam search."""

from __future__ import annotations

from typing import Optional, Sequence

import msgspec
import numpy as np
import torch
from safetensors import safe_open

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig

_MAX_DENSE_LEVEL2_MASK_BYTES = 512 * 1024**2
_GROUP_VALIDATION_CHUNK = 250_000


def _integer_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array


def _validate_offsets(offsets: np.ndarray, count: int, end: int, name: str) -> None:
    if (
        len(offsets) != count + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != end
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"Invalid {name}")


def _validate_grouped_order(
    values: np.ndarray,
    offsets: np.ndarray,
    name: str,
    *,
    unique: bool,
) -> int:
    if np.any(offsets[1:] == offsets[:-1]):
        raise ValueError(f"{name} contains an empty group")
    unique_count = 0
    for group_start in range(0, len(offsets) - 1, _GROUP_VALIDATION_CHUNK):
        group_end = min(
            group_start + _GROUP_VALIDATION_CHUNK,
            len(offsets) - 1,
        )
        value_start = int(offsets[group_start])
        value_end = int(offsets[group_end])
        chunk = values[value_start:value_end]
        invalid = chunk[1:] <= chunk[:-1] if unique else chunk[1:] < chunk[:-1]
        boundaries = offsets[group_start + 1 : group_end] - value_start - 1
        invalid[boundaries] = False
        if np.any(invalid):
            qualifier = "sorted and unique" if unique else "sorted"
            raise ValueError(f"{name} must be {qualifier} within each group")
        if not unique:
            changes = chunk[1:] != chunk[:-1]
            changes[boundaries] = False
            unique_count += group_end - group_start + int(np.count_nonzero(changes))
    return len(values) if unique else unique_count


def _compact_signed_array(values: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= np.iinfo(np.int16).max:
        dtype = np.int16
    elif maximum <= np.iinfo(np.int32).max:
        dtype = np.int32
    else:
        dtype = np.int64
    if values.dtype.itemsize == np.dtype(dtype).itemsize:
        return values.view(dtype)
    return values.astype(dtype)


class _SparseLevel2(msgspec.Struct, frozen=True):
    keys: torch.Tensor
    offsets: torch.Tensor
    values: torch.Tensor
    max_children: int


class BeamTrieConstraint:
    """Device-resident valid-child masks for a three-level SID trie."""

    def __init__(
        self,
        config: TrieOutputHeadConfig,
        root_mask: torch.Tensor,
        level1_masks: torch.Tensor,
        level2_masks: Optional[torch.Tensor],
        level2_sparse: Optional[_SparseLevel2],
        level_cardinalities: tuple[int, int, int],
    ):
        self.config = config
        self.root_mask = root_mask
        self.level1_masks = level1_masks
        self.level2_masks = level2_masks
        self.level2_sparse = level2_sparse
        self.level_cardinalities = level_cardinalities

    @classmethod
    def load(
        cls, config: TrieOutputHeadConfig, device: torch.device | str
    ) -> BeamTrieConstraint:
        with safe_open(config.tensor_path, framework="pt", device="cpu") as data:
            required = {
                "root_token_ids",
                "level1_token_ids",
                "level1_offsets",
                "level2_offsets",
                "level2_token_ids",
            }
            missing = required - set(data.keys())
            if missing:
                raise ValueError(
                    f"Trie output head is missing tensors: {sorted(missing)}"
                )
            a_keys = _integer_array(
                data.get_tensor("root_token_ids").numpy(), "root_token_ids"
            )
            b_keys = _integer_array(
                data.get_tensor("level1_token_ids").numpy(), "level1_token_ids"
            )
            a_offsets = _integer_array(
                data.get_tensor("level1_offsets").numpy(), "level1_offsets"
            )
            c_offsets = _integer_array(
                data.get_tensor("level2_offsets").numpy(), "level2_offsets"
            )
            c_values = _integer_array(
                data.get_tensor("level2_token_ids").numpy(), "level2_token_ids"
            )

        size = config.codebook_size
        _validate_offsets(a_offsets, len(a_keys), len(b_keys), "level1_offsets")
        _validate_offsets(c_offsets, len(b_keys), len(c_values), "level2_offsets")
        for values, name in (
            (a_keys, "root_token_ids"),
            (b_keys, "level1_token_ids"),
            (c_values, "level2_token_ids"),
        ):
            if len(values) and (int(values.min()) < 0 or int(values.max()) >= size):
                raise ValueError(f"{name} contains an out-of-range SID code")
        if np.any(a_keys[1:] <= a_keys[:-1]):
            raise ValueError("root_token_ids must be sorted and unique")

        root_mask = np.zeros(size, dtype=np.bool_)
        level1_masks = np.zeros((size, size), dtype=np.bool_)
        root_mask[a_keys] = True

        a_for_b = np.repeat(a_keys, np.diff(a_offsets))
        _validate_grouped_order(
            b_keys,
            a_offsets,
            "B children",
            unique=True,
        )
        level1_masks[a_for_b, b_keys] = True

        if not root_mask.any():
            raise ValueError("Beam trie has no valid root tokens")
        if np.any(level1_masks[a_keys].sum(axis=1) == 0):
            raise ValueError("Beam trie contains an A prefix with no B children")
        level2_cardinality = _validate_grouped_order(
            c_values,
            c_offsets,
            "C children",
            unique=False,
        )

        ab_dtype = np.int32 if size**2 - 1 <= np.iinfo(np.int32).max else np.int64
        ab_rows = a_for_b.astype(ab_dtype, copy=False) * size + b_keys.astype(
            ab_dtype, copy=False
        )
        level2_masks = None
        level2_sparse = None
        if size**3 <= _MAX_DENSE_LEVEL2_MASK_BYTES:
            level2_masks = np.zeros((size * size, size), dtype=np.bool_)
            for group_start in range(
                0,
                len(ab_rows),
                _GROUP_VALIDATION_CHUNK,
            ):
                group_end = min(
                    group_start + _GROUP_VALIDATION_CHUNK,
                    len(ab_rows),
                )
                value_start = int(c_offsets[group_start])
                value_end = int(c_offsets[group_end])
                rows = np.repeat(
                    ab_rows[group_start:group_end],
                    np.diff(c_offsets[group_start : group_end + 1]),
                )
                level2_masks[rows, c_values[value_start:value_end]] = True
        else:
            offset_dtype = (
                np.int32 if int(c_offsets[-1]) <= np.iinfo(np.int32).max else np.int64
            )
            level2_sparse = _SparseLevel2(
                keys=torch.from_numpy(_compact_signed_array(ab_rows, size**2 - 1)).to(
                    device
                ),
                offsets=torch.from_numpy(c_offsets.astype(offset_dtype, copy=False)).to(
                    device
                ),
                values=torch.from_numpy(_compact_signed_array(c_values, size - 1)).to(
                    device
                ),
                max_children=int(np.diff(c_offsets).max()),
            )

        return cls(
            config=config,
            root_mask=torch.from_numpy(root_mask).to(device),
            level1_masks=torch.from_numpy(level1_masks).to(device),
            level2_masks=(
                torch.from_numpy(level2_masks).to(device)
                if level2_masks is not None
                else None
            ),
            level2_sparse=level2_sparse,
            level_cardinalities=(
                int(root_mask.sum()),
                int(level1_masks.sum()),
                level2_cardinality,
            ),
        )

    def _sparse_level2_rows(self, prefix_codes: torch.Tensor) -> torch.Tensor:
        sparse = self.level2_sparse
        assert sparse is not None
        size = self.config.codebook_size
        ab_rows = (prefix_codes[:, 0] * size + prefix_codes[:, 1]).to(sparse.keys.dtype)
        pair_rows = torch.searchsorted(sparse.keys, ab_rows)
        safe_rows = pair_rows.clamp_max(sparse.keys.numel() - 1)
        valid = ~(
            (pair_rows == sparse.keys.numel()) | (sparse.keys[safe_rows] != ab_rows)
        )
        if pair_rows.is_cuda:
            torch._assert_async(
                torch.all(valid),
                "Trie prefix contains an unknown AB pair",
            )
        elif not torch.all(valid):
            raise ValueError("Trie prefix contains an unknown AB pair")
        return pair_rows

    def _sparse_level2_mask(self, prefix_codes: torch.Tensor) -> torch.Tensor:
        sparse = self.level2_sparse
        assert sparse is not None
        size = self.config.codebook_size
        pair_rows = self._sparse_level2_rows(prefix_codes)
        if pair_rows.is_cuda:
            from sglang.srt.beam_search.trie_kernels import (
                materialize_sparse_mask,
            )

            return materialize_sparse_mask(
                pair_rows,
                sparse.offsets,
                sparse.values,
                self.config.codebook_size,
                sparse.max_children,
            )
        starts = sparse.offsets[pair_rows].long()
        lengths = sparse.offsets[pair_rows + 1].long() - starts
        total_values = int(lengths.sum().item())
        mask = torch.zeros(
            (prefix_codes.shape[0], size),
            dtype=torch.bool,
            device=prefix_codes.device,
        )
        if total_values == 0:
            return mask
        output_rows = torch.repeat_interleave(
            torch.arange(prefix_codes.shape[0], device=prefix_codes.device),
            lengths,
            output_size=total_values,
        )
        range_bases = torch.cumsum(lengths, dim=0) - lengths
        value_indices = torch.arange(
            total_values,
            device=prefix_codes.device,
        ) + torch.repeat_interleave(
            starts - range_bases,
            lengths,
            output_size=total_values,
        )
        child_codes = sparse.values[value_indices].long()
        mask[output_rows, child_codes] = True
        return mask

    def _mask_sparse_level2_logits(
        self,
        logits: torch.Tensor,
        prefix_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse = self.level2_sparse
        assert sparse is not None and logits.is_cuda
        from sglang.srt.beam_search.trie_kernels import mask_sparse_logits

        pair_rows = self._sparse_level2_rows(prefix_codes)
        starts = sparse.offsets[pair_rows].long()
        return (
            mask_sparse_logits(
                logits,
                pair_rows,
                sparse.offsets,
                sparse.values,
                sparse.max_children,
            ),
            sparse.values[starts].long().unsqueeze(1),
        )

    def valid_child_mask(self, prefix_codes: torch.Tensor, depth: int) -> torch.Tensor:
        size = self.config.codebook_size
        if depth == 0:
            if prefix_codes.shape != (1, 0):
                raise ValueError(
                    f"Depth-zero prefix state must have shape (1, 0), got "
                    f"{tuple(prefix_codes.shape)}"
                )
            return self.root_mask.unsqueeze(0)
        if prefix_codes.ndim != 2 or prefix_codes.shape[1] != depth:
            raise ValueError(
                f"Expected prefix state [rows, {depth}], got "
                f"{tuple(prefix_codes.shape)}"
            )
        if depth == 1:
            return self.level1_masks[prefix_codes[:, 0]]
        if depth == 2:
            rows = prefix_codes[:, 0] * size + prefix_codes[:, 1]
            if self.level2_masks is not None:
                return self.level2_masks[rows]
            return self._sparse_level2_mask(prefix_codes)
        raise ValueError(f"Unsupported SID depth: {depth}")

    def topk_logprobs(
        self,
        pieces: Sequence[torch.Tensor],
        prefix_codes: torch.Tensor,
        depth: int,
        num_candidates: int,
        normalizers: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score valid children from dense or compact normalized logits.

        Dense logits preserve the full-vocabulary score exactly: normalize over the
        full vocabulary, then gather valid SID tokens. Compact logits use a
        streamed full-vocabulary normalizer supplied by the output head.
        """

        sparse_cuda = (
            depth == 2
            and self.level2_sparse is not None
            and pieces
            and pieces[0].is_cuda
            and all(piece.shape[1] == self.config.codebook_size for piece in pieces)
        )
        masks = None if sparse_cuda else self.valid_child_mask(prefix_codes, depth)
        token_start = self.config.token_start_for_depth(depth)
        size = self.config.codebook_size
        vals, toks = [], []
        row_start = 0
        if normalizers is not None and len(normalizers) != len(pieces):
            raise ValueError("Beam trie logits and normalizer pieces must align")
        for piece_index, piece in enumerate(pieces):
            rows = piece.shape[0]
            if rows == 0:
                continue
            row_masks = None if masks is None else masks[row_start : row_start + rows]
            row_start += rows
            x = piece.float()
            if x.shape[1] == size:
                if normalizers is None:
                    from sglang.srt.beam_search.trie_output_head import (
                        use_topk_logprob,
                    )

                    if not use_topk_logprob():
                        raise ValueError(
                            "Compact beam trie logits require full-vocabulary "
                            "normalizer metadata"
                        )
                else:
                    normalizer = normalizers[piece_index]
                    if normalizer.shape != (rows, 2):
                        raise ValueError(
                            "Beam trie normalizer rows must match compact logits"
                        )
                codebook_logits = x
            else:
                token_end = token_start + size
                if x.shape[1] < token_end:
                    raise ValueError(
                        f"Dense beam logits width {x.shape[1]} does not cover "
                        f"SID token range [{token_start}, {token_end})"
                    )
                codebook_logits = x[:, token_start:token_end]
            if sparse_cuda:
                candidates, fallback = self._mask_sparse_level2_logits(
                    codebook_logits,
                    prefix_codes[row_start - rows : row_start],
                )
            else:
                candidates = codebook_logits.masked_fill(
                    ~row_masks,
                    -torch.inf,
                )
                fallback = row_masks.to(torch.int64).argmax(
                    dim=-1,
                    keepdim=True,
                )
            k = min(num_candidates, size)
            v, t = torch.topk(candidates, k, dim=-1)
            # Wide beams can retain score-dead (-inf) candidates. Keep their
            # token IDs on-trie so a later decode never indexes an empty prefix.
            t = torch.where(torch.isneginf(v), fallback, t)
            if x.shape[1] == size:
                if normalizers is None:
                    vals.append(v - torch.logsumexp(v, dim=-1, keepdim=True))
                else:
                    vals.append(
                        (v - normalizer[:, :1].float()) - normalizer[:, 1:].float()
                    )
            else:
                vals.append(v - torch.logsumexp(x, dim=-1, keepdim=True))
            toks.append(t + token_start)
        if row_start != prefix_codes.shape[0]:
            raise ValueError(
                f"Beam logits rows ({row_start}) do not match trie prefix rows "
                f"({prefix_codes.shape[0]})"
            )
        values, tokens = torch.cat(vals), torch.cat(toks)
        if depth == 0 and values.shape[1] < num_candidates:
            padding = num_candidates - values.shape[1]
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (values.shape[0], padding),
                        -torch.inf,
                        dtype=values.dtype,
                        device=values.device,
                    ),
                ],
                dim=1,
            )
            tokens = torch.cat(
                [tokens, tokens[:, :1].expand(-1, padding)],
                dim=1,
            )
        return values, tokens

    def advance_prefixes(
        self,
        prefix_codes: torch.Tensor,
        parent_idx: torch.Tensor,
        next_tokens: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        token_start = self.config.token_start_for_depth(depth)
        codes = next_tokens - token_start
        return torch.cat([prefix_codes[parent_idx], codes.unsqueeze(1)], dim=1)
