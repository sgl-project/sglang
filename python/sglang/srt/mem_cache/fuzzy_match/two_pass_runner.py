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

"""Two-pass extend orchestration for non-prefix-anchored fuzzy match.

When ``match_prefix`` returns a ``FuzzyMatchBlock`` whose target position
isn't at the exact-prefix boundary, the unmatched suffix decomposes as::

    |exact_matched_len ... target_start ... target_start+L ... seq_len|
       ^                    ^                      ^                  ^
       prefix already       lead-in tokens          block tokens      trailing
       cached               (cold prefill)          (donor KV reuse)  tokens

We need a forward extend that prefills the lead-in and trailing regions
while placing donor KV (with RoPE corrected from the donor's positions
to the recipient's) into the block region. SGLang's stock
``forward_extend`` assumes a contiguous extend window, so we orchestrate
this here as:

    pass 1   standard extend on the lead-in tokens
    block    per-layer pool memcpy of donor KV with RoPE correction
    pass 2   standard extend on the trailing tokens (produces sampling logits)

Pulling this module out of ``model_runner.py`` keeps the model executor's
forward path free of fuzzy-match-specific branching: ``forward_extend``
gets a one-line delegating gate, and all of the orchestration lives in
this fuzzy-match module that maintainers can review or skip independently.

This is the v1 single-block path. Multi-region support (multiple disjoint
fuzzy regions) is documented separately and would extend this module to
``N+1`` passes interleaved with ``N`` block placements.
"""

from __future__ import annotations

import copy as _copy
import logging
from typing import Any, Tuple

import torch

from sglang.srt.mem_cache.fuzzy_match.donor_injection import (
    inject_donor_match_block_kv,
)

logger = logging.getLogger(__name__)


def run_two_pass_extend(
    model_runner: Any,
    forward_batch: Any,
    skip_attn_backend_init: bool,
    pp_proxy_tensors: Any,
) -> Tuple[Any, bool]:
    """Two-pass forward_extend for a non-prefix-anchored fuzzy match.

    See module docstring for the architectural overview.

    Args:
        model_runner: The ``ModelRunner`` instance dispatching this call.
            We hold a reference so we can invoke the model forward, build
            the attention metadata, and resolve the rotary embedding.
        forward_batch: The full unmatched-suffix ForwardBatch produced by
            the scheduler. Contains ``fuzzy_match_blocks`` (length-1 list
            in v1) and ``out_cache_loc`` covering the full extend window.
        skip_attn_backend_init: Forwarded from ``forward_extend``. Currently
            unused on this path (each sub-pass re-initializes the attention
            metadata for its own slice).
        pp_proxy_tensors: Forwarded from ``forward_extend``. Currently
            unused on this path (pipeline parallelism + fuzzy match is not
            exercised in v1).

    Returns:
        ``(logits_processor_output, can_run_graph)`` matching
        ``ModelRunner.forward_extend``'s contract. ``can_run_graph`` is
        always ``False`` here because the per-pass shapes are not graph-
        cacheable; this is consistent with the previous in-line
        implementation and a known limitation of the v1 design.
    """
    block = forward_batch.fuzzy_match_blocks[0]
    req = forward_batch.reqs[0]
    pool = forward_batch.token_to_kv_pool

    exact_matched_len = int(forward_batch.extend_prefix_lens_cpu[0])
    seq_len = (
        int(forward_batch.seq_lens_cpu[0].item())
        if hasattr(forward_batch.seq_lens_cpu, "item")
        else int(forward_batch.seq_lens_cpu[0])
    )
    target_start = int(block.target_start_in_prompt)
    block_len = int(block.length)
    lead_in_len = target_start - exact_matched_len
    main_extend_len = seq_len - target_start - block_len
    total_extend_len = int(forward_batch.extend_num_tokens or 0)

    # Geometry validation; the provider should have caught this earlier
    # but the defensive check here prevents silent pool corruption.
    if (
        lead_in_len < 0
        or block_len <= 0
        or main_extend_len < 1  # need >=1 trailing token for sampling logits
        or lead_in_len + block_len + main_extend_len != total_extend_len
    ):
        logger.warning(
            "[FUZZY] match_block geometry invalid: lead_in=%d, block=%d, "
            "main=%d, total_extend=%d; falling back to cold prefill",
            lead_in_len,
            block_len,
            main_extend_len,
            total_extend_len,
        )
        release_fuzzy_realized_locs(model_runner, req)
        forward_batch.fuzzy_match_blocks = None
        forward_batch.fuzzy_matched_len = 0
        return model_runner.forward_extend(
            forward_batch, skip_attn_backend_init, pp_proxy_tensors
        )

    # Slice the pre-allocated out_cache_loc into three regions matching
    # the conceptual layout of the unmatched suffix.
    lead_in_slots = forward_batch.out_cache_loc[:lead_in_len]
    middle_slots = forward_batch.out_cache_loc[
        lead_in_len : lead_in_len + block_len
    ]
    main_slots = forward_batch.out_cache_loc[lead_in_len + block_len :]

    # === PASS 1: lead-in extend (skip if lead-in is empty) ===
    if lead_in_len > 0:
        pass1_batch = _build_fuzzy_subforward_batch(
            forward_batch,
            start_in_extend=0,
            end_in_extend=lead_in_len,
            new_extend_prefix_len=exact_matched_len,
            new_seq_len=target_start,
            out_cache_loc=lead_in_slots,
        )
        model_runner.attn_backend.init_forward_metadata(pass1_batch)
        # Discard pass 1's LogitsProcessorOutput; only the K,V cache
        # writes performed inside attention layers matter for pass 2.
        model_runner.model.forward(
            pass1_batch.input_ids,
            pass1_batch.positions,
            pass1_batch,
        )

    # === Block placement: donor KV → realized_locs with RoPE correction ===
    realized_locs = getattr(req, "fuzzy_realized_locs", None)
    rotary_emb, _, _, _ = model_runner._fuzzy_get_rotary_emb()
    req_idx = int(forward_batch.req_pool_indices[0].item())
    placed = inject_donor_match_block_kv(
        pool=pool,
        req_to_token=forward_batch.req_to_token_pool.req_to_token,
        req_pool_idx=req_idx,
        rotary_emb=rotary_emb,
        block=block,
        realized_locs=realized_locs,
    )
    if not placed:
        # Injection failure: realized_locs survives (we keep its handle so
        # the scheduler can free it on req completion). Fall through to
        # pass 2 unchanged; pass 2's attention will read whatever K/V
        # happens to live at those slots, so quality will degrade for
        # this one request, but the pool stays consistent.
        logger.warning(
            "[FUZZY] match_block injection returned False; pass 2 will "
            "attend over uninitialized block slots. Quality may suffer "
            "for this request."
        )

    # Free the slots alloc_for_extend reserved for the block's position
    # range. realized_locs (from match_prefix) holds the donor KV instead;
    # those middle slots would otherwise leak.
    if middle_slots.numel() > 0:
        try:
            model_runner.token_to_kv_pool_allocator.free(middle_slots)
        except Exception as e:
            logger.warning("[FUZZY] match_block middle slot free failed: %s", e)

    # === PASS 2: main extend ===
    pass2_batch = _build_fuzzy_subforward_batch(
        forward_batch,
        start_in_extend=lead_in_len + block_len,
        end_in_extend=total_extend_len,
        new_extend_prefix_len=target_start + block_len,
        new_seq_len=seq_len,
        out_cache_loc=main_slots,
    )
    model_runner.attn_backend.init_forward_metadata(pass2_batch)
    output = model_runner.model.forward(
        pass2_batch.input_ids,
        pass2_batch.positions,
        pass2_batch,
    )

    # Cleanup: clear realized_locs ownership (its slots are now committed
    # via req_to_token) and zero out the fuzzy gate so chunked-prefill
    # re-entry, decode batches, or retraction-and-resume don't re-fire.
    req.fuzzy_realized_locs = None
    req.cache_fuzzy_matched_len = 0

    logger.info(
        "[FUZZY] match_block realized: lead_in=%d tokens prefilled, "
        "block=%d tokens reused (donor_start=%d -> target_start=%d), "
        "main=%d tokens prefilled. Saved ~%d tokens of prefill vs cold.",
        lead_in_len,
        block_len,
        block.donor_start,
        target_start,
        main_extend_len,
        block_len,
    )

    # CUDA graph isn't supported on this path in v1 (variable per-pass
    # shapes). Matches the previous inline implementation's behavior.
    return output, False


def release_fuzzy_realized_locs(model_runner: Any, req: Any) -> None:
    """Free ``req.fuzzy_realized_locs`` and clear the request's match_block state.

    Invoked from the geometry-invalid fallback path and (if exposed)
    by request-completion cleanup hooks. Idempotent.
    """
    rl = getattr(req, "fuzzy_realized_locs", None)
    if rl is not None:
        try:
            model_runner.token_to_kv_pool_allocator.free(rl)
        except Exception:
            pass
        req.fuzzy_realized_locs = None
    req.cache_fuzzy_matched_len = 0


def _build_fuzzy_subforward_batch(
    batch: Any,
    start_in_extend: int,
    end_in_extend: int,
    new_extend_prefix_len: int,
    new_seq_len: int,
    out_cache_loc: torch.Tensor,
) -> Any:
    """Build a partial ForwardBatch covering ``batch.input_ids[start:end]``.

    Returns a new ForwardBatch with sliced input_ids/positions and
    re-computed extend metadata. Pool references, model state, attention
    backend, and req objects are shared with the parent batch by reference
    (we are not actually creating a new request).

    Fuzzy-related fields are zeroed on the sub-batch so the legacy
    ``_correct_fuzzy_kv_rope`` paths no-op if anyone calls them; the
    two-pass entry point is gated above, so the subforward must not
    recurse into another two-pass dispatch.
    """
    sub = _copy.copy(batch)
    device = batch.input_ids.device

    sub.input_ids = batch.input_ids[start_in_extend:end_in_extend].contiguous()
    sub.positions = batch.positions[start_in_extend:end_in_extend].contiguous()
    sub.out_cache_loc = out_cache_loc.contiguous()

    new_extend_seq_len = end_in_extend - start_in_extend
    sub.extend_num_tokens = new_extend_seq_len
    sub.extend_seq_lens = torch.tensor(
        [new_extend_seq_len], dtype=torch.int64, device=device
    )
    sub.extend_seq_lens_cpu = [new_extend_seq_len]
    sub.extend_prefix_lens = torch.tensor(
        [new_extend_prefix_len], dtype=torch.int64, device=device
    )
    sub.extend_prefix_lens_cpu = [new_extend_prefix_len]
    sub.extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)

    sub.seq_lens = torch.tensor(
        [new_seq_len], dtype=batch.seq_lens.dtype, device=batch.seq_lens.device
    )
    if isinstance(batch.seq_lens_cpu, torch.Tensor):
        sub.seq_lens_cpu = torch.tensor([new_seq_len], dtype=torch.int64)
    else:
        sub.seq_lens_cpu = [new_seq_len]
    sub.seq_lens_sum = new_seq_len

    # Disarm any fuzzy handling inside the subforward; the two-pass
    # caller is responsible for orchestration.
    sub.fuzzy_match_blocks = None
    sub.fuzzy_matched_len = 0
    sub.fuzzy_segments = None
    sub.fuzzy_cached_start_pos = 0

    return sub
