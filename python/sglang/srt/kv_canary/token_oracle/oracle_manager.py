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

        generalized_req_ids = _build_generalized_req_id_per_token(
            forward_batch=forward_batch,
            num_tokens=num_tokens,
            generalized_req_ids_per_row=select_generalized_req_ids(
                vanilla_req_ids=forward_batch.rids_int,
                bootstrap_room_ids_int=forward_batch.bootstrap_room_ids_int,
            ),
        )
        if forward_batch.forward_mode.is_extend():
            expected_tokens = input_ids
        else:
            expected_tokens = self.oracle.expected_tokens(
                generalized_req_ids=generalized_req_ids,
                positions=positions.to(torch.int64),
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


def _build_generalized_req_id_per_token(
    *,
    forward_batch: "ForwardBatch",
    num_tokens: int,
    generalized_req_ids_per_row: torch.Tensor,
) -> torch.Tensor:
    # target_verify / draft_extend run a constant number of tokens per req
    # (Python int from spec_info). Capture-safe expansion: broadcast +
    # reshape; the output shape is a capture-time constant since both
    # ``bs`` and ``per_req`` are Python ints.
    forward_mode = forward_batch.forward_mode
    if forward_mode.is_target_verify():
        per_req = int(forward_batch.spec_info.draft_token_num)
        result = _expand_uniform(generalized_req_ids_per_row, per_req)
    elif forward_mode.is_draft_extend(include_v2=True):
        per_req = int(forward_batch.spec_info.num_tokens_per_req)
        result = _expand_uniform(generalized_req_ids_per_row, per_req)
    elif forward_mode.is_extend():
        # Extend (prefill) runs eager / never enters cuda-graph capture, so
        # ``torch.repeat_interleave`` with a runtime-valued repeats tensor is
        # acceptable here.
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "fill_expected_inputs: extend_seq_lens is None in extend mode"
            )
        lens = extend_seq_lens.to(torch.int64)
        result = torch.repeat_interleave(generalized_req_ids_per_row, lens)
    else:
        result = generalized_req_ids_per_row

    if int(result.shape[0]) != num_tokens:
        raise RuntimeError(
            f"fill_expected_inputs: sum(lens)={int(result.shape[0])} != num_tokens={num_tokens}"
        )
    return result


def _expand_uniform(values: torch.Tensor, per_row: int) -> torch.Tensor:
    """Capture-safe equivalent of ``torch.repeat_interleave(values, per_row)``
    when ``per_row`` is a Python int (uniform per row). Returns a contiguous
    flat tensor of shape ``[values.shape[0] * per_row]``."""
    bs = int(values.shape[0])
    return values.unsqueeze(1).expand(bs, per_row).reshape(bs * per_row)


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
