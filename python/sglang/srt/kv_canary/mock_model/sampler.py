from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_canary.mock_model.oracle import Oracle
from sglang.srt.layers.sampler import Sampler, register_sampler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_ORACLE_BACKEND_NAME: str = "oracle"


class OracleSamplerHook:
    """One-per-server state for the oracle sampler integration. Owns the oracle and the per-step
    row -> req-id mapping. Both canary's input-check path (fill_expected_inputs) and sglang's
    sampler dispatch (_OracleSampler) operate on the same instance.

    Lifecycle: install_oracle_sampler constructs one and registers it as the "oracle" sampler
    factory. The returned hook is then attached to the CanaryRunner so its before_forward can
    populate the row -> req-id stash that _OracleSampler will read at sample time.
    """

    def __init__(self, *, oracle: Oracle) -> None:
        self.oracle = oracle
        self._req_pool_indices_per_row: Optional[List[int]] = None

    def fill_expected_inputs(
        self,
        *,
        forward_batch: "ForwardBatch",
        expected_input_tokens_out: torch.Tensor,
        expected_input_positions_out: torch.Tensor,
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

        req_pool_indices_per_row = [
            int(v) for v in forward_batch.req_pool_indices.detach().to("cpu").tolist()
        ]
        self._req_pool_indices_per_row = req_pool_indices_per_row

        if num_tokens == 0:
            return

        req_id_per_token = _build_req_id_per_token(
            forward_batch=forward_batch,
            num_tokens=num_tokens,
            req_pool_indices_per_row=req_pool_indices_per_row,
        )
        positions_cpu = positions.detach().to("cpu", dtype=torch.int64).tolist()

        expected_tokens: List[int] = [0] * num_tokens
        expected_positions: List[int] = [0] * num_tokens
        for i in range(num_tokens):
            req_id = req_id_per_token[i]
            position = positions_cpu[i]
            expected_tokens[i] = int(
                self.oracle.expected_token(req_id=req_id, position=position)
            )
            expected_positions[i] = position

        tokens_tensor = torch.tensor(
            expected_tokens,
            dtype=torch.int32,
            device=expected_input_tokens_out.device,
        )
        positions_tensor = torch.tensor(
            expected_positions,
            dtype=torch.int32,
            device=expected_input_positions_out.device,
        )
        expected_input_tokens_out[:num_tokens].copy_(tokens_tensor)
        expected_input_positions_out[:num_tokens].copy_(positions_tensor)

    def sample(self, *, logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Produce one token per row from the oracle, using the row -> req-id stash filled by
        fill_expected_inputs. Caller is _OracleSampler.forward.
        """
        req_pool_indices_per_row = self._req_pool_indices_per_row
        if req_pool_indices_per_row is None:
            raise RuntimeError(
                "OracleSamplerHook.sample: req_pool_indices not stashed; "
                "fill_expected_inputs must be called before sampling "
                "(input_check_mode == ON required)"
            )

        device = logits.device
        bs = int(logits.shape[0])

        assert len(req_pool_indices_per_row) == bs, (
            f"OracleSamplerHook.sample: stashed req_pool_indices length "
            f"{len(req_pool_indices_per_row)} != logits batch size {bs}"
        )

        positions_cpu = positions.detach().to("cpu", dtype=torch.int64).tolist()
        token_ids: List[int] = [0] * bs
        for r in range(bs):
            token_ids[r] = int(
                self.oracle.expected_token(
                    req_id=int(req_pool_indices_per_row[r]),
                    position=int(positions_cpu[r]),
                )
            )

        return torch.tensor(token_ids, dtype=torch.int32, device=device)


def install_oracle_sampler(*, oracle: Oracle) -> OracleSamplerHook:
    """Register an oracle-driven sampler backend with sglang's sampler registry. Returns the
    OracleSamplerHook so the caller can attach it to a CanaryRunner for the input-check path.

    sglang's main Sampler has a single-line dispatch at the top of its sample() that, when the
    'oracle' backend is selected, delegates to _OracleSampler — which forwards to the hook
    instance bound at registration. No monkey-patching — relies on the existing
    sampler-backend mechanism. Calling twice replaces the previously registered hook.
    """
    hook = OracleSamplerHook(oracle=oracle)
    register_sampler_backend(
        _ORACLE_BACKEND_NAME, lambda: _OracleSampler(oracle_sampler_hook=hook)
    )
    return hook


def _build_req_id_per_token(
    *,
    forward_batch: "ForwardBatch",
    num_tokens: int,
    req_pool_indices_per_row: List[int],
) -> List[int]:
    bs = len(req_pool_indices_per_row)

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
        out.extend([int(req_pool_indices_per_row[r])] * int(lens[r]))
    assert (
        len(out) == num_tokens
    ), f"fill_expected_inputs: sum(lens)={len(out)} != num_tokens={num_tokens}"
    return out


class _OracleSampler(Sampler):
    """Sampler subclass that bypasses logits and returns oracle-driven token ids per row.

    Constructed by the factory closure register_sampler_backend installs in
    install_oracle_sampler; the closure captures a single OracleSamplerHook so every sampler
    instance dispatched for the "oracle" backend shares the same stash filled by canary's
    before_forward.
    """

    def __init__(self, *, oracle_sampler_hook: OracleSamplerHook) -> None:
        super().__init__()
        self._oracle_sampler_hook = oracle_sampler_hook

    def forward(
        self,
        logits_output: "LogitsProcessorOutput",
        sampling_info: "SamplingBatchInfo",
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_next_token_ids = self._oracle_sampler_hook.sample(
            logits=logits_output.next_token_logits,
            positions=positions,
        )
        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids
