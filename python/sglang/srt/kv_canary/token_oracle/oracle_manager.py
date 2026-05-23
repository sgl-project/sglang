from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.token_oracle.oracle import TokenOracle

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TokenOracleManager:
    def __init__(self, *, oracle: TokenOracle) -> None:
        self.oracle = oracle

    def fill_expected_inputs(
        self,
        *,
        forward_batch: "ForwardBatch",
        expected_inputs_out: ExpectedInputs,
    ) -> None:
        """Called by SingleForwardManager (phase 2) BEFORE its per-forward write
        launch when CanaryConfig.input_check_mode is ON. Layout of the output
        tensors mirrors forward_batch.input_ids / forward_batch.positions —
        flat, indexed by the same write_offsets canary uses. Capacity of the
        placeholders is sized at install time to max per-forward write
        capacity (cuda-graph safe).

        For positions outside [0, current_seq_len) the value is undefined — caller is expected to
        keep mock-model running such that every (generalized_req_id, position) hit by the canary
        write loop is a valid oracle query.
        """
        positions = forward_batch.positions
        input_ids = forward_batch.input_ids
        num_tokens = int(input_ids.shape[0])

        if num_tokens == 0:
            return

        generalized_req_ids_per_row = select_generalized_req_ids(
            vanilla_req_ids=forward_batch.rids_int,
            bootstrap_room_ids_int=forward_batch.bootstrap_room_ids_int,
        )
        if forward_batch.forward_mode.is_extend():
            # Extend / target_verify / draft_extend all use forward_batch.input_ids
            # as ground truth (the prefill / verify path already pins the
            # expected tokens into the batch). Capture is irrelevant here:
            # extend runs eager, and target_verify / draft_extend take a
            # zero-allocation shortcut that doesn't touch the oracle.
            expected_tokens = input_ids
        else:
            expected_tokens = _compute_expected_tokens_capture_safe(
                oracle=self.oracle,
                forward_batch=forward_batch,
                num_tokens=num_tokens,
                positions=positions,
                generalized_req_ids_per_row=generalized_req_ids_per_row,
            )
        expected_inputs_out.tokens[:num_tokens].copy_(expected_tokens.to(torch.int64))
        expected_inputs_out.positions[:num_tokens].copy_(positions.to(torch.int64))

    def sample_next_tokens(
        self, *, generalized_req_ids: torch.Tensor, logits_positions: torch.Tensor
    ) -> torch.Tensor:
        return self.oracle.expected_tokens(
            generalized_req_ids=generalized_req_ids,
            positions=logits_positions.to(torch.int64) + 1,
        )


def _compute_expected_tokens_capture_safe(
    *,
    oracle: TokenOracle,
    forward_batch: "ForwardBatch",
    num_tokens: int,
    positions: torch.Tensor,
    generalized_req_ids_per_row: torch.Tensor,
) -> torch.Tensor:
    """Compute ``oracle(generalized_req_ids, positions)`` for every token in
    the batch without materializing a flat ``[num_tokens]`` copy of the
    per-row req-ids inside cuda-graph capture.

    Reached only from non-extend forward modes (the extend / target_verify
    / draft_extend caller branches all short-circuit to
    ``expected_tokens = input_ids`` before this helper is called).
    Effectively the captured cases are DECODE and DRAFT_EXTEND_V2.

    For DRAFT_EXTEND_V2 the per-row count is a Python int from spec_info
    — a capture-time constant. The oracle is fed a 2D ``[bs, per_row]``
    view of positions (zero-alloc reshape because positions is contiguous
    after the dtype cast) and an ``[bs, 1]`` view of the per-row req-ids
    that broadcasts element-wise inside the oracle. The output is
    allocated as ``[bs, per_row]`` directly in the cuda-graph pool, and
    the final ``reshape(num_tokens)`` on that contiguous tensor is a free
    view — crucially, we never call ``reshape`` on a stride-0 broadcast
    view, which would force a host-visible contiguous copy.

    For DECODE the per-row count is 1, so we skip the expansion entirely.
    """
    forward_mode = forward_batch.forward_mode
    if forward_mode.is_draft_extend(include_v2=True):
        per_row = int(forward_batch.spec_info.num_tokens_per_req)
        return _oracle_per_row_uniform(
            oracle=oracle,
            num_tokens=num_tokens,
            per_row=per_row,
            positions=positions,
            generalized_req_ids_per_row=generalized_req_ids_per_row,
        )
    # DECODE (and any other non-extend mode): one token per row, so
    # ``generalized_req_ids_per_row`` already has shape ``[num_tokens]``.
    if int(generalized_req_ids_per_row.shape[0]) != num_tokens:
        raise RuntimeError(
            f"fill_expected_inputs: per_row_shape={int(generalized_req_ids_per_row.shape[0])} "
            f"!= num_tokens={num_tokens}"
        )
    return oracle.expected_tokens(
        generalized_req_ids=generalized_req_ids_per_row,
        positions=positions.to(torch.int64),
    )


def _oracle_per_row_uniform(
    *,
    oracle: TokenOracle,
    num_tokens: int,
    per_row: int,
    positions: torch.Tensor,
    generalized_req_ids_per_row: torch.Tensor,
) -> torch.Tensor:
    """Run the oracle with uniform ``per_row`` tokens per request.

    Capture-safe: no ``reshape`` on a stride-0 view, no eager allocation
    outside the cuda-graph pool. ``positions`` is reshaped via ``view``
    (alloc-free because contiguous after the dtype cast), and
    ``generalized_req_ids_per_row`` is broadcast against ``positions_2d``
    element-wise inside the oracle. The final output is then
    ``reshape``-flattened back to ``[num_tokens]`` — since the broadcast
    output is contiguous, that reshape is a free view.
    """
    bs = int(generalized_req_ids_per_row.shape[0])
    if bs * per_row != num_tokens:
        raise RuntimeError(
            f"fill_expected_inputs: bs*per_row={bs * per_row} != num_tokens={num_tokens}"
        )
    # ``.reshape(bs, per_row)`` returns a view when positions is contiguous
    # (the common case for forward_batch.positions) and only falls back to
    # a copy otherwise — but unlike reshape on a stride-0 broadcast, the
    # copy here is bounded by num_tokens (not amplified by per_row), and
    # only happens when positions is already non-contiguous.
    positions_2d = positions.to(torch.int64).reshape(bs, per_row)
    expected_tokens_2d = oracle.expected_tokens(
        generalized_req_ids=generalized_req_ids_per_row.unsqueeze(1),
        positions=positions_2d,
    )
    return expected_tokens_2d.reshape(num_tokens)


def select_generalized_req_ids(
    *,
    vanilla_req_ids: torch.Tensor,
    bootstrap_room_ids_int: torch.Tensor | None,
) -> torch.Tensor:
    if bootstrap_room_ids_int is None:
        return vanilla_req_ids

    bootstrap_room_ids_int = bootstrap_room_ids_int.to(
        device=vanilla_req_ids.device,
        dtype=torch.int64,
    )
    return torch.where(
        bootstrap_room_ids_int >= 0,
        bootstrap_room_ids_int,
        vanilla_req_ids.to(torch.int64),
    )
