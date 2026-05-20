from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.srt.kv_canary.token_oracle.oracle import TokenIdOracle
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenIdOracleManager
from sglang.srt.layers.sampler import Sampler, register_sampler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_ORACLE_BACKEND_NAME: str = "oracle"


def install_oracle_sampler(*, oracle: TokenIdOracle) -> TokenIdOracleManager:
    """Register an oracle-driven sampler backend with sglang's sampler registry. Returns the
    TokenIdOracleManager so the caller can attach it to a CanaryRunner for the input-check path.

    sglang's main Sampler has a single-line dispatch at the top of its sample() that, when the
    'oracle' backend is selected, delegates to _OracleSampler — which forwards to the manager
    instance bound at registration. No monkey-patching — relies on the existing
    sampler-backend mechanism. Calling twice replaces the previously registered manager.
    """
    manager = TokenIdOracleManager(oracle=oracle)
    register_sampler_backend(
        _ORACLE_BACKEND_NAME,
        lambda: _OracleSampler(token_id_oracle_manager=manager),
    )
    return manager


class _OracleSampler(Sampler):
    """Sampler subclass that bypasses logits and returns oracle-driven token ids per row.

    Constructed by the factory closure register_sampler_backend installs in
    install_oracle_sampler; the closure captures a single TokenIdOracleManager so every sampler
    instance dispatched for the "oracle" backend shares the same stash filled by canary's
    before_forward.
    """

    def __init__(self, *, token_id_oracle_manager: TokenIdOracleManager) -> None:
        super().__init__()
        self._token_id_oracle_manager = token_id_oracle_manager

    def forward(
        self,
        logits_output: "LogitsProcessorOutput",
        sampling_info: "SamplingBatchInfo",
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_next_token_ids = self._token_id_oracle_manager.sample(
            logits=logits_output.next_token_logits,
            positions=positions,
        )
        return batch_next_token_ids
