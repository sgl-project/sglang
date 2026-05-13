# Copyright 2023-2024 SGLang Team
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

"""Donor KV injection for non-prefix-anchored fuzzy match.

When a request's fuzzy match has a ``match_block`` whose target position
isn't at the exact-prefix boundary, the block's KV cannot be expressed as
a contiguous prefix of the request's KV pool slots. Instead the scheduler
treats the request as two chunked-prefill iterations:

    chunk 1: normal forward_extend on the lead-in tokens
             ``[exact_matched_len, target_start)``
    inject:  per-layer pool memcpy of donor KV into recipient-owned slots
             at the block's positions, with RoPE corrected from the
             donor's absolute position to the recipient's
    chunk 2: normal forward_extend on the trailing tokens
             ``[target_start + block_len, seq_len)``; produces sampling
             logits

This module owns the "inject" step. It runs OUTSIDE the model executor's
forward path — it's a memory-pool operation, not a model-execution
operation. The scheduler invokes it between chunks (typically from
``init_next_round_input`` or the equivalent chunk-transition hook) so the
model_runner's forward path stays free of fuzzy-match-specific branching.

Pre-conditions:
    * ``realized_locs`` was allocated for the block at match-prefix time
      (in ``RadixCache._match_prefix_fuzzy_match_block``) and lives on
      ``req.fuzzy_realized_locs`` until consumed here.
    * The donor's TreeNode is ``inc_lock_ref``'d so its slots aren't
      LRU-evicted before this copy runs.

Post-conditions (when the function returns True):
    * Per-layer K, V buffers at ``realized_locs`` contain RoPE-corrected
      donor KV.
    * ``req_to_token[req_pool_idx, target_start:target_start + L]`` points
      at ``realized_locs``.
    * The caller is responsible for advancing ``req.prefix_indices`` by
      ``L`` so chunk 2's ``forward_extend`` sees the block as cached.

On failure (False), the caller must release ``realized_locs`` to the
allocator and let the request fall through to a standard cold prefill
of the unmatched suffix.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    as_long_tensor,
    copy_kv_with_rope_correction,
)

logger = logging.getLogger(__name__)


def inject_donor_match_block_kv(
    *,
    pool: Any,
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    rotary_emb: Any,
    block: Any,
    realized_locs: Any,
) -> bool:
    """Place donor KV into recipient slots at the block's positions, RoPE-fixed.

    Args:
        pool: KV pool (``token_to_kv_pool``) with ``k_buffer``, ``v_buffer``
            lists and ``layer_num``.
        req_to_token: The flat ``req_to_token_pool.req_to_token`` tensor.
            Indexed as ``[req_pool_idx, prompt_position]`` to KV slot.
        req_pool_idx: This request's row in ``req_to_token``.
        rotary_emb: The model's rotary embedding object, exposing
            ``cos_sin_cache``, ``is_neox_style``, ``rotary_dim``. None-check
            and resolution is the caller's responsibility (typically via
            ``model_runner._fuzzy_get_rotary_emb()``).
        block: A ``FuzzyMatchBlock``-shaped object with
            ``target_start_in_prompt``, ``donor_start``, ``length``, and
            ``donor_kv_indices`` fields.
        realized_locs: Recipient-owned destination slots, length ==
            ``block.length``. Accepts a torch.Tensor or any sequence the
            ``as_long_tensor`` helper can coerce.

    Returns:
        True on a successful placement; the caller must then advance
            ``req.prefix_indices`` by ``block.length``.
        False on a shape / arg failure; the caller must release
            ``realized_locs`` and fall back to cold prefill of the block
            region.
    """
    if rotary_emb is None:
        logger.warning(
            "[FUZZY] inject_donor_match_block_kv: rotary_emb is None; "
            "skipping injection (request will cold-prefill the block)"
        )
        return False
    if realized_locs is None:
        logger.warning(
            "[FUZZY] inject_donor_match_block_kv: realized_locs is None; "
            "skipping injection"
        )
        return False

    device = pool.k_buffer[0].device
    target_start = int(block.target_start_in_prompt)
    donor_start = int(block.donor_start)
    block_len = int(block.length)
    if block_len <= 0:
        logger.warning(
            "[FUZZY] inject_donor_match_block_kv: non-positive block_len=%d; "
            "skipping injection",
            block_len,
        )
        return False

    old_positions = torch.arange(
        donor_start, donor_start + block_len, device=device, dtype=torch.long
    )
    new_positions = torch.arange(
        target_start, target_start + block_len, device=device, dtype=torch.long
    )

    donor_kv_indices = as_long_tensor(block.donor_kv_indices, device)
    new_locs = as_long_tensor(realized_locs, device)
    if donor_kv_indices.numel() != block_len or new_locs.numel() != block_len:
        logger.warning(
            "[FUZZY] inject_donor_match_block_kv: shape mismatch "
            "(donor_kv=%d, realized_locs=%d, block_len=%d); skipping",
            int(donor_kv_indices.numel()),
            int(new_locs.numel()),
            block_len,
        )
        return False

    copy_kv_with_rope_correction(
        pool=pool,
        rotary_emb=rotary_emb,
        old_locs=donor_kv_indices,
        new_locs=new_locs,
        old_positions=old_positions,
        new_positions=new_positions,
    )

    # Repoint req_to_token at the recipient-owned slots so chunk 2's
    # attention reads donor KV from these positions.
    req_to_token[req_pool_idx, target_start : target_start + block_len] = (
        new_locs.to(req_to_token.dtype)
    )

    logger.info(
        "[FUZZY] match_block realized: %d tokens reused "
        "(donor_start=%d -> target_start=%d). Saved ~%d tokens of prefill vs cold.",
        block_len,
        donor_start,
        target_start,
        block_len,
    )
    return True
