from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.token_oracle.oracle import TokenOracle

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TokenOracleManager:
    def __init__(self, *, oracle: TokenOracle) -> None:
        self.oracle = oracle
        self._rids_per_row: Optional[torch.Tensor] = None

    def fill_expected_inputs(
        self,
        *,
        forward_batch: "ForwardBatch",
        expected_inputs_out: ExpectedInputs,
    ) -> None:
        """Called by CanaryRunner BEFORE its per-forward write launch when
        CanaryConfig.input_check_mode is ON. Layout of the output tensors mirrors
        forward_batch.input_ids / forward_batch.positions — flat, indexed by the same
        write_offsets canary uses. Capacity of the placeholders is sized at CanaryRunner install
        time to max per-forward write capacity (cuda-graph safe).

        For positions outside [0, current_seq_len) the value is undefined — caller is expected to
        keep mock-model running such that every (req_id, position) hit by the canary write loop
        is a valid oracle query.
        """
        positions = forward_batch.positions
        input_ids = forward_batch.input_ids
        num_tokens = int(input_ids.shape[0])

        rids = forward_batch.rids
        if rids is None:
            raise RuntimeError(
                "fill_expected_inputs: forward_batch.rids is None; "
                "token oracle requires per-request rid strings"
            )

        self._rids_per_row = _hash_rids_to_i64_tensor(
            rids=rids,
            padded_bs=int(forward_batch.req_pool_indices.shape[0]),
            device=forward_batch.req_pool_indices.device,
        )

        if num_tokens == 0:
            return

        req_ids = _build_req_id_per_token(
            forward_batch=forward_batch,
            num_tokens=num_tokens,
            rids_per_row=self._rids_per_row,
        )
        expected_tokens = self.oracle.expected_tokens(
            req_ids=req_ids, positions=positions.to(torch.int64)
        )
        expected_inputs_out.tokens[:num_tokens].copy_(expected_tokens)
        expected_inputs_out.positions[:num_tokens].copy_(positions.to(torch.int32))

    def sample(self, *, logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        stash = self._rids_per_row
        if stash is None:
            raise RuntimeError(
                "TokenOracleManager.sample: rids not stashed; "
                "fill_expected_inputs must be called before sampling "
                "(input_check_mode is True required)"
            )

        if int(stash.shape[0]) != int(logits.shape[0]):
            raise RuntimeError(
                f"TokenOracleManager.sample: stashed rids length "
                f"{int(stash.shape[0])} != logits batch size {int(logits.shape[0])}"
            )

        return self.oracle.expected_tokens(
            req_ids=stash, positions=positions.to(torch.int64)
        )


def _build_req_id_per_token(
    *,
    forward_batch: "ForwardBatch",
    num_tokens: int,
    rids_per_row: torch.Tensor,
) -> torch.Tensor:
    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "fill_expected_inputs: extend_seq_lens is None in extend mode"
            )
        lens = extend_seq_lens.to(torch.int64)
        result = torch.repeat_interleave(rids_per_row, lens)
    else:
        result = rids_per_row

    if int(result.shape[0]) != num_tokens:
        raise RuntimeError(
            f"fill_expected_inputs: sum(lens)={int(result.shape[0])} != num_tokens={num_tokens}"
        )
    return result


def _hash_rids_to_i64_tensor(
    *, rids: List[str], padded_bs: int, device: torch.device
) -> torch.Tensor:
    values: List[int] = [_stable_hash_rid_i64(rid) for rid in rids]
    if padded_bs > len(values):
        # Cuda-graph padding rows have no real request; outputs at these rows are discarded.
        values.extend([0] * (padded_bs - len(values)))
    elif padded_bs < len(values):
        raise RuntimeError(
            f"_hash_rids_to_i64_tensor: padded_bs={padded_bs} < len(rids)={len(values)}"
        )
    return torch.tensor(values, dtype=torch.int64, device=device)


def _stable_hash_rid_i64(rid: str) -> int:
    digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)
