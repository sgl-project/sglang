from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_cache_canary.mock_model.oracle import Oracle
from sglang.srt.layers.sampler import Sampler, register_sampler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_REGISTERED_ORACLE: Optional[Oracle] = None
_ORACLE_BACKEND_NAME: str = "oracle"
_ORACLE_BACKEND_REGISTERED: bool = False


def install_oracle_sampler(*, oracle: Oracle) -> None:
    """Register an oracle-driven sampler backend with sglang's sampler registry.

    sglang's main Sampler has a single-line dispatch at the top of its sample() that, when the
    'oracle' backend is selected, delegates here. No monkey-patching - relies on the existing
    sampler-backend mechanism. Calling twice replaces the previous oracle.

    After this call, sampler.sample(batch) returns, per req:
        token_id = oracle.expected_token(req_id=req.rid, position=req.next_position)
    bypassing logits entirely.
    """
    global _REGISTERED_ORACLE, _ORACLE_BACKEND_REGISTERED
    _REGISTERED_ORACLE = oracle
    if not _ORACLE_BACKEND_REGISTERED:
        register_sampler_backend(_ORACLE_BACKEND_NAME, _OracleSampler)
        _ORACLE_BACKEND_REGISTERED = True


def fill_expected_inputs(
    *,
    forward_batch: "ForwardBatch",
    expected_input_tokens_out: torch.Tensor,
    expected_input_positions_out: torch.Tensor,
) -> None:
    """Per-token compute expected (token, position) from oracle and write into placeholders.

    Called by CanaryRunner BEFORE its per-forward write launch when CanaryConfig.input_check_mode
    is ON. Layout of the output tensors mirrors forward_batch.input_ids / forward_batch.positions -
    flat, indexed by the same write_offsets canary uses. Capacity of the placeholders is sized at
    CanaryRunner install time to max per-forward write capacity (cuda-graph safe).

    For positions outside [0, current_seq_len) the value is undefined - caller is expected to
    keep mock-model running such that every (req_id, position) hit by the canary write loop is
    a valid oracle query.
    """
    oracle = _REGISTERED_ORACLE
    if oracle is None:
        raise RuntimeError("fill_expected_inputs called before install_oracle_sampler")

    positions = forward_batch.positions
    input_ids = forward_batch.input_ids
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return

    req_id_per_token = _build_req_id_per_token(
        forward_batch=forward_batch, num_tokens=num_tokens
    )
    positions_cpu = positions.detach().to("cpu", dtype=torch.int64).tolist()

    expected_tokens: List[int] = [0] * num_tokens
    expected_positions: List[int] = [0] * num_tokens
    for i in range(num_tokens):
        req_id = req_id_per_token[i]
        position = positions_cpu[i]
        expected_tokens[i] = int(
            oracle.expected_token(req_id=req_id, position=position)
        )
        expected_positions[i] = position

    tokens_tensor = torch.tensor(
        expected_tokens, dtype=torch.int32, device=expected_input_tokens_out.device
    )
    positions_tensor = torch.tensor(
        expected_positions,
        dtype=torch.int32,
        device=expected_input_positions_out.device,
    )
    expected_input_tokens_out[:num_tokens].copy_(tokens_tensor)
    expected_input_positions_out[:num_tokens].copy_(positions_tensor)


def _build_req_id_per_token(
    *, forward_batch: "ForwardBatch", num_tokens: int
) -> List[int]:
    req_pool_indices = forward_batch.req_pool_indices.detach().to("cpu").tolist()
    bs = len(req_pool_indices)

    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and forward_mode.is_extend():
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "fill_expected_inputs: extend_seq_lens is None in extend mode"
            )
        lens = extend_seq_lens.detach().to("cpu", dtype=torch.int64).tolist()
    else:
        lens = [1] * bs

    out: List[int] = []
    for r in range(bs):
        out.extend([int(req_pool_indices[r])] * int(lens[r]))
    if len(out) < num_tokens:
        out.extend(
            [int(req_pool_indices[bs - 1] if bs > 0 else 0)] * (num_tokens - len(out))
        )
    elif len(out) > num_tokens:
        out = out[:num_tokens]
    return out


class _OracleSampler(Sampler):
    """Sampler subclass that bypasses logits and returns oracle-driven token ids per row.

    Uses the per-forward row index as req_id (matches fill_expected_inputs row -> req mapping)
    and forward positions tensor as the per-row position.
    """

    def forward(
        self,
        logits_output: "LogitsProcessorOutput",
        sampling_info: "SamplingBatchInfo",
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        oracle = _REGISTERED_ORACLE
        if oracle is None:
            raise RuntimeError(
                "_OracleSampler.forward: no oracle registered (install_oracle_sampler not called)"
            )

        logits = logits_output.next_token_logits
        device = logits.device
        bs = int(logits.shape[0])

        positions_cpu = positions.detach().to("cpu", dtype=torch.int64).tolist()
        token_ids: List[int] = [0] * bs
        for r in range(bs):
            token_ids[r] = int(
                oracle.expected_token(req_id=r, position=int(positions_cpu[r]))
            )

        batch_next_token_ids = torch.tensor(token_ids, dtype=torch.int32, device=device)

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids
