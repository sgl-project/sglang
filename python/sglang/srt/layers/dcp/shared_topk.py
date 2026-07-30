# Copyright 2023-2026 SGLang Team
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

"""Exact global Top-K merge for an owner-sharded DCP Indexer.

Each DCP rank scores only the Indexer-K entries that it owns. A global Top-K
entry must be present in its owner's local Top-K, so exchanging K candidates
per owner is exact while avoiding replication of the full Indexer cache and
logit matrix.

The explicit candidate AllGather in this module is the control transport. The
VMM transport uses the same candidate membership semantics: descending score
with lower global token ID as the deterministic cutoff tie-break. CUDA
selectors may return that exact set in an unspecified array order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel

_UINT32_MASK = (1 << 32) - 1
_INT64_SIGN = torch.iinfo(torch.int64).min


@triton.jit
def _mask_owner_mandatory_tokens_kernel(
    logits,
    global_lengths,
    row_starts,
    logits_row_stride: tl.constexpr,
    logits_col_stride: tl.constexpr,
    num_cols: tl.constexpr,
    dcp_rank: tl.constexpr,
    dcp_size: tl.constexpr,
    num_init_tokens: tl.constexpr,
    num_local_tokens: tl.constexpr,
    has_row_starts: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, block_size)
    global_length = tl.load(global_lengths + row)

    is_init = slot < num_init_tokens
    is_local = (slot >= num_init_tokens) & (slot < num_init_tokens + num_local_tokens)
    global_position = tl.where(
        is_init,
        slot,
        global_length - 1 - (slot - num_init_tokens),
    )
    valid = (
        (is_init | is_local)
        & (global_position >= 0)
        & (global_position < global_length)
        & (global_position % dcp_size == dcp_rank)
    )
    local_position = global_position // dcp_size
    row_start = 0
    if has_row_starts:
        row_start = tl.load(row_starts + row)
    column = row_start + local_position
    valid &= (column >= 0) & (column < num_cols)
    tl.store(
        logits + row * logits_row_stride + column * logits_col_stride,
        float("inf"),
        mask=valid,
    )


def mask_owner_mandatory_tokens(
    logits: torch.Tensor,
    global_lengths: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_size: int,
    num_init_tokens: int,
    num_local_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Force globally mandatory tokens into this owner's local candidate set."""
    mandatory = num_init_tokens + num_local_tokens
    if mandatory == 0 or logits.shape[0] == 0:
        return logits
    if logits.device.type != "cuda":
        raise RuntimeError("owner-sharded DCP mandatory masking requires CUDA")
    if global_lengths.shape[0] != logits.shape[0]:
        raise ValueError(
            "global_lengths must have one entry per logit row: "
            f"{global_lengths.shape[0]} != {logits.shape[0]}"
        )
    row_starts_arg = row_starts if row_starts is not None else global_lengths
    _mask_owner_mandatory_tokens_kernel[(logits.shape[0],)](
        logits,
        global_lengths,
        row_starts_arg,
        logits.stride(0),
        logits.stride(1),
        logits.shape[1],
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        num_init_tokens=num_init_tokens,
        num_local_tokens=num_local_tokens,
        has_row_starts=row_starts is not None,
        block_size=triton.next_power_of_2(mandatory),
    )
    return logits


def pack_owner_candidates(
    logits: torch.Tensor,
    local_indices: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_size: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack local ``(score, global logical token id)`` candidates as fp32."""
    if logits.dtype != torch.float32 or local_indices.dtype != torch.int32:
        raise TypeError(
            "DCP Top-K candidates require fp32 logits and int32 indices, got "
            f"{logits.dtype} and {local_indices.dtype}"
        )
    if logits.shape[0] != local_indices.shape[0]:
        raise ValueError("logits and local_indices row counts must match")

    if row_starts is None:
        row_starts = torch.zeros(
            local_indices.shape[0],
            dtype=torch.int32,
            device=local_indices.device,
        )
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=local_indices.device)

    valid = local_indices >= 0
    score_columns = local_indices.clamp_min(0) + row_starts[:, None]
    if logits.shape[1] == 0:
        scores = logits.new_full(local_indices.shape, float("-inf"))
    else:
        scores = torch.gather(
            logits,
            1,
            score_columns.clamp_max(logits.shape[1] - 1).to(torch.int64),
        )
        scores = scores.masked_fill(~valid, float("-inf"))
    global_ids = local_indices.to(torch.int64) * dcp_size + dcp_rank
    global_ids = global_ids.masked_fill(~valid, -1).to(torch.int32)
    # Preserve the full int32 id range while retaining one homogeneous fp32
    # candidate tensor for a single collective.
    return torch.stack((scores, global_ids.view(torch.float32)), dim=-1)


def stable_topk_from_candidates(
    candidates: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select by descending score, then ascending global token ID.

    This PyTorch reference returns IDs in key order. CUDA selectors are only
    required to return the same exact set because sparse attention is
    permutation-invariant over its selected tokens.
    """
    if (
        candidates.dtype != torch.float32
        or candidates.ndim != 3
        or candidates.shape[-1] != 2
    ):
        raise TypeError("candidates must be fp32 with shape [rows, candidates, 2]")
    rows, count, _ = candidates.shape
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=candidates.device)
    if rows == 0 or count == 0 or topk == 0:
        return output

    scores = candidates[..., 0].contiguous()
    token_ids = candidates[..., 1].contiguous().view(torch.int32)
    score_bits = scores.view(torch.int32).to(torch.int64) & _UINT32_MASK
    score_keys = torch.where(
        (score_bits & 0x80000000) != 0,
        score_bits ^ _UINT32_MASK,
        score_bits ^ 0x80000000,
    )
    keys = (score_keys << 32) | ((~token_ids.to(torch.int64)) & _UINT32_MASK)
    keys = keys.masked_fill(token_ids < 0, 0)

    # torch.topk orders signed int64. Flipping the sign bit gives unsigned
    # ordering, which is the ordering of the packed stable keys above.
    valid_topk = min(topk, count)
    signed_keys = keys ^ _INT64_SIGN
    selected_signed = torch.topk(signed_keys, valid_topk, dim=1, sorted=True).values
    selected_keys = selected_signed ^ _INT64_SIGN
    selected_ids = ((~selected_keys) & _UINT32_MASK).to(torch.int32)
    selected_ids = selected_ids.masked_fill(selected_keys == 0, -1)
    output[:, :valid_topk] = selected_ids
    return output


def merge_owner_topk_allgather(
    logits: torch.Tensor,
    local_indices: torch.Tensor,
    topk: int,
    *,
    dcp_rank: int,
    dcp_size: int,
    row_starts: torch.Tensor | None = None,
    group: GroupCoordinator | None = None,
) -> torch.Tensor:
    """Exact candidate merge using an explicit DCP AllGather control."""
    if dcp_size <= 1:
        return local_indices
    candidates = pack_owner_candidates(
        logits,
        local_indices,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        row_starts=row_starts,
    )
    if group is None:
        group = get_parallel().dcp_group
    gathered = group.all_gather(candidates, dim=1)
    if gathered.device.type == "cuda" and gathered.shape[1] % 512 == 0:
        from sglang.kernels.ops.attention.dsa.dcp_indexer_cutedsl import (
            stable_topk_from_gathered_candidates_cutedsl,
        )

        return stable_topk_from_gathered_candidates_cutedsl(gathered, topk)
    return stable_topk_from_candidates(gathered, topk)
