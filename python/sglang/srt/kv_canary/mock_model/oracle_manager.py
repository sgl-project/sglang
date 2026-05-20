from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.mock_model.oracle import TokenIdOracle

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TokenIdOracleManager:
    """One-per-server state for the oracle sampler integration. Owns the oracle and the per-step
    row -> req-id mapping. Both canary's input-check path (fill_expected_inputs) and sglang's
    sampler dispatch (_OracleSampler) operate on the same instance.

    Lifecycle: install_oracle_sampler constructs one and registers it as the "oracle" sampler
    factory. The returned hook is then attached to the CanaryRunner so its before_forward can
    populate the row -> req-id stash that _OracleSampler will read at sample time.
    """

    def __init__(self, *, oracle: TokenIdOracle) -> None:
        self.oracle = oracle
        self._req_pool_indices_per_row: Optional[torch.Tensor] = None

    def fill_expected_inputs(
        self,
        *,
        forward_batch: "ForwardBatch",
        expected_inputs_out: ExpectedInputs,
    ) -> None:
        """Per-token compute expected (token, position) from oracle and write into placeholders.

        Called by CanaryRunner BEFORE its per-forward write launch when
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

        self._req_pool_indices_per_row = forward_batch.req_pool_indices.to(torch.int64)

        if num_tokens == 0:
            return

        req_ids = _build_req_id_per_token(
            forward_batch=forward_batch,
            num_tokens=num_tokens,
            req_pool_indices_per_row=self._req_pool_indices_per_row,
        )
        expected_tokens = self.oracle.expected_tokens(
            req_ids=req_ids, positions=positions.to(torch.int64)
        )
        expected_inputs_out.tokens[:num_tokens].copy_(expected_tokens)
        expected_inputs_out.positions[:num_tokens].copy_(positions.to(torch.int32))

    def sample(self, *, logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Produce one token per row from the oracle, using the row -> req-id stash filled by
        fill_expected_inputs. Caller is _OracleSampler.forward.
        """
        stash = self._req_pool_indices_per_row
        if stash is None:
            raise RuntimeError(
                "TokenIdOracleManager.sample: req_pool_indices not stashed; "
                "fill_expected_inputs must be called before sampling "
                "(input_check_mode == ON required)"
            )

        assert int(stash.shape[0]) == int(logits.shape[0]), (
            f"TokenIdOracleManager.sample: stashed req_pool_indices length "
            f"{int(stash.shape[0])} != logits batch size {int(logits.shape[0])}"
        )

        return self.oracle.expected_tokens(
            req_ids=stash, positions=positions.to(torch.int64)
        )


def _build_req_id_per_token(
    *,
    forward_batch: "ForwardBatch",
    num_tokens: int,
    req_pool_indices_per_row: torch.Tensor,
) -> torch.Tensor:
    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "fill_expected_inputs: extend_seq_lens is None in extend mode"
            )
        lens = extend_seq_lens.to(torch.int64)
        result = torch.repeat_interleave(req_pool_indices_per_row, lens)
    else:
        result = req_pool_indices_per_row

    assert (
        int(result.shape[0]) == num_tokens
    ), f"fill_expected_inputs: sum(lens)={int(result.shape[0])} != num_tokens={num_tokens}"
    return result
