"""Real-model token-id validator for kv_canary: fills ``ExpectedInputs.tokens``
from each req's snapshotted ``origin_input_ids + output_ids`` so the canary
write kernel can flag any ``forward_batch.input_ids[i]`` mismatch against the
1-req source-of-truth."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def fill_expected_inputs_from_reqs(
    *,
    forward_batch: "ForwardBatch",
    out_expected_inputs: ExpectedInputs,
    pool: Optional[torch.Tensor],
    valid_lens: Optional[torch.Tensor],
) -> None:
    """Fill ``out_expected_inputs.tokens`` per forward from each req's
    snapshotted ``origin_input_ids + output_ids``; out-of-range positions
    (speculative drafts, bonus tail) get ``input_ids[i]`` for tautological
    skip. ``pool`` / ``valid_lens`` are written for future kernel-side gather
    (currently unused by the host-side path)."""
    positions = forward_batch.positions
    input_ids = forward_batch.input_ids
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return

    req_truth_seqs = forward_batch.req_truth_seqs
    if req_truth_seqs is None:
        out_expected_inputs.tokens[:num_tokens].copy_(input_ids.to(torch.int64))
        out_expected_inputs.positions[:num_tokens].copy_(positions.to(torch.int64))
        return

    mode_offset = _logical_pos_offset(forward_batch=forward_batch)
    per_req_token_counts = _per_req_token_counts(
        forward_batch=forward_batch,
        num_tokens=num_tokens,
        num_reqs=len(req_truth_seqs),
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
    for seq, req_pool_idx, token_count in zip(
        req_truth_seqs, req_pool_indices_cpu, per_req_token_counts
    ):
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
    out_expected_inputs.tokens[:num_tokens].copy_(expected_tokens_tensor)
    out_expected_inputs.positions[:num_tokens].copy_(positions.to(torch.int64))

    if pool is not None and valid_lens is not None and valid_lens_cpu is not None:
        _refresh_pool_rows(pool=pool, valid_lens=valid_lens, updates=pool_row_updates)
        valid_lens.copy_(
            torch.tensor(valid_lens_cpu, dtype=torch.int32, device=valid_lens.device)
        )


def _logical_pos_offset(*, forward_batch: "ForwardBatch") -> int:
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    forward_mode = forward_batch.forward_mode
    if forward_mode.is_draft_extend(include_v2=True):
        return 1
    # EAGLE draft prefill keeps forward_mode == EXTEND but rotates input_ids in place
    # so slot p stores K/V for token p+1 (see eagle_worker_v2._draft_extend_for_prefill).
    if forward_mode.is_extend() and isinstance(
        forward_batch.spec_info, EagleDraftInput
    ):
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
