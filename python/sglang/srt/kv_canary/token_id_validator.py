"""Real-model token-id validator for kv_canary.

When ``SGLANG_KV_CANARY_ENABLE_REQ_TOKEN_IDS_CHECK`` is on, every forward fills
the canary write kernel's ``expected_tokens`` placeholder from each req's own
``origin_input_ids + output_ids`` (the 1-req source-of-truth). The write kernel
then asserts that ``forward_batch.input_ids[i]`` matches the expected token at
the same slot for every token whose ``logical_pos`` is in-range; out-of-range
tokens (speculative drafts, verify-stage proposals, prefill rotation bonus
tail, etc.) get ``expected_tokens[i] = input_ids[i]`` so the write kernel's
mismatch check becomes tautological and skips them.

The validator also writes the same ``seq[:valid_len]`` into the per-req row of
``CanaryDeviceState.req_to_expected_token_ids_pool`` and the matching length
into ``req_to_expected_token_ids_valid_lens``; both tensors are allocated only
when this flag is on. They are not consumed by the current host-side gather
path, but exist so a future refactor can move the gather into the verify
kernel (see the plan doc) without re-introducing host overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def fill_expected_inputs_from_reqs(
    *,
    forward_batch: "ForwardBatch",
    expected_inputs_out: ExpectedInputs,
    pool: Optional[torch.Tensor],
    valid_lens: Optional[torch.Tensor],
) -> None:
    """Real-model fill of ``expected_inputs_out`` from per-req prompt+output history.

    Per-forward steps:
      1. Refresh the per-req pool row and ``valid_lens`` entry for every req in
         this forward's batch (full rewrite — the same req can re-enter the
         batch after preemption with a new history; we never trust stale rows).
      2. Build a flat ``expected_tokens[num_tokens]`` on host by walking each
         req's segment of the forward and looking up ``seq[logical_pos]`` for
         each token, where ``logical_pos = positions[i] + mode_offset``.
      3. Out-of-range tokens (``logical_pos < 0`` or ``>= valid_len``) get
         ``expected_tokens[i] = input_ids[i]`` so the write kernel's check is
         tautological and skips them — these are speculative draft proposals,
         verify-stage proposals, or prefill rotation bonus tail with no
         source-of-truth.
      4. Single H2D copy into ``expected_inputs_out.tokens[:num_tokens]``;
         ``expected_inputs_out.positions[:num_tokens]`` just mirrors
         ``forward_batch.positions`` (same as the mock-model path).

    Forward-mode → ``logical_pos`` offset:
      - ``is_extend()`` (target prefill) / ``is_decode_or_idle()``: 0
      - ``is_draft_extend(include_v2=True)``: +1 (EAGLE rotation: slot ``p``
        stores token at logical position ``p+1``)
      - ``is_target_verify()`` and any other mode: all draft proposals — they
        sit past the commit point so ``logical_pos >= valid_len`` for every
        token and the whole batch auto-skips via the out-of-range branch.

    Args:
        forward_batch: The forward batch. ``forward_batch.reqs`` must be
            populated; ``ForwardBatch.init_new`` does this when the env flag
            is on.
        expected_inputs_out: Pre-allocated capacity tensors; we overwrite the
            first ``num_tokens`` entries.
        pool: Optional device tensor shape
            ``[req_to_token_alloc_size, max_context_len]``. Written but not
            read by this path; kept for forward-compat with a future
            kernel-side gather.
        valid_lens: Optional device tensor shape ``[req_to_token_alloc_size]``.
            Same as ``pool``.
    """
    reqs = forward_batch.reqs
    if reqs is None:
        raise RuntimeError(
            "kv-canary: fill_expected_inputs_from_reqs called but "
            "forward_batch.reqs is None; set "
            "SGLANG_KV_CANARY_ENABLE_REQ_TOKEN_IDS_CHECK=1 so ForwardBatch.init_new "
            "populates it"
        )

    positions = forward_batch.positions
    input_ids = forward_batch.input_ids
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return

    mode_offset = _logical_pos_offset(forward_batch=forward_batch)
    per_req_token_counts = _per_req_token_counts(
        forward_batch=forward_batch, num_tokens=num_tokens, num_reqs=len(reqs)
    )

    req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()
    positions_cpu = positions.tolist()
    input_ids_cpu = input_ids.tolist()

    expected_tokens_host: List[int] = [0] * num_tokens
    if pool is not None and valid_lens is not None:
        valid_lens_cpu = valid_lens.cpu().tolist()
    else:
        valid_lens_cpu = None

    pool_row_updates: List[tuple[int, List[int]]] = []
    cursor = 0
    for req, req_pool_idx, token_count in zip(
        reqs, req_pool_indices_cpu, per_req_token_counts
    ):
        seq = _build_req_truth_sequence(req=req)
        valid_len = len(seq)
        if pool is not None and valid_lens_cpu is not None:
            pool_row_updates.append((req_pool_idx, seq))
            valid_lens_cpu[req_pool_idx] = valid_len

        for local_idx in range(token_count):
            flat_idx = cursor + local_idx
            logical_pos = int(positions_cpu[flat_idx]) + mode_offset
            if 0 <= logical_pos < valid_len:
                expected_tokens_host[flat_idx] = int(seq[logical_pos])
            else:
                expected_tokens_host[flat_idx] = int(input_ids_cpu[flat_idx])
        cursor += token_count

    expected_tokens_tensor = torch.tensor(
        expected_tokens_host, dtype=torch.int64, device=input_ids.device
    )
    expected_inputs_out.tokens[:num_tokens].copy_(expected_tokens_tensor)
    expected_inputs_out.positions[:num_tokens].copy_(positions.to(torch.int64))

    if pool is not None and valid_lens is not None and valid_lens_cpu is not None:
        _refresh_pool_rows(pool=pool, valid_lens=valid_lens, updates=pool_row_updates)
        valid_lens.copy_(
            torch.tensor(valid_lens_cpu, dtype=torch.int32, device=valid_lens.device)
        )


def _logical_pos_offset(*, forward_batch: "ForwardBatch") -> int:
    forward_mode = forward_batch.forward_mode
    if forward_mode.is_draft_extend(include_v2=True):
        return 1
    return 0


def _per_req_token_counts(
    *, forward_batch: "ForwardBatch", num_tokens: int, num_reqs: int
) -> List[int]:
    forward_mode = forward_batch.forward_mode
    if forward_mode.is_target_verify():
        per_req = int(forward_batch.spec_info.draft_token_num)
        counts = [per_req] * num_reqs
    elif forward_mode.is_draft_extend(include_v2=True):
        per_req = int(forward_batch.spec_info.num_tokens_per_req)
        counts = [per_req] * num_reqs
    elif forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "kv-canary: extend mode but forward_batch.extend_seq_lens is None"
            )
        counts = extend_seq_lens.tolist()
    else:
        counts = [1] * num_reqs

    total = sum(counts)
    if total != num_tokens:
        raise RuntimeError(
            f"kv-canary: fill_expected_inputs_from_reqs sum(counts)={total} "
            f"!= num_tokens={num_tokens} (forward_mode={forward_mode})"
        )
    return counts


def _build_req_truth_sequence(*, req: "Req") -> List[int]:
    # req.origin_input_ids / req.output_ids are typed arrays ("q", int64);
    # concatenating them yields the full source-of-truth token history.
    return list(req.origin_input_ids) + list(req.output_ids)


def _refresh_pool_rows(
    *,
    pool: torch.Tensor,
    valid_lens: torch.Tensor,
    updates: List[tuple[int, List[int]]],
) -> None:
    if not updates:
        return
    max_context_len = int(pool.shape[1])
    for req_pool_idx, seq in updates:
        seq_len = len(seq)
        if seq_len > max_context_len:
            raise RuntimeError(
                f"kv-canary: req sequence length {seq_len} exceeds canary pool "
                f"max_context_len {max_context_len}; raise --context-length or "
                f"audit ReqToTokenPool sizing"
            )
        if seq_len == 0:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.int32, device=pool.device)
        pool[req_pool_idx, :seq_len].copy_(seq_tensor)
