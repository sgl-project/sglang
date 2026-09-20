from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.srt.kv_canary.perturb.next_token_swap import maybe_perturb_swap_next_tokens
from sglang.srt.kv_canary.token_oracle.oracle import TokenOracle
from sglang.srt.kv_canary.token_oracle.oracle_manager import (
    TokenOracleManager,
    select_generalized_req_ids,
)
from sglang.srt.layers.sampler import Sampler, register_sampler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def install_oracle_sampler(*, oracle: TokenOracle) -> TokenOracleManager:
    manager = TokenOracleManager(oracle=oracle)
    register_sampler_backend(
        "token_oracle",
        lambda: _OracleSampler(token_oracle_manager=manager),
    )
    return manager


class _OracleSampler(Sampler):
    def __init__(self, *, token_oracle_manager: TokenOracleManager) -> None:
        super().__init__()
        self._token_oracle_manager = token_oracle_manager

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        vanilla_req_ids = sampling_info.rids_int
        if vanilla_req_ids is None:
            raise RuntimeError(
                "_OracleSampler.forward: generalized_req_id source tensor is None; "
                "token oracle requires a per-forward generalized_req_id source tensor "
                "(set in ForwardBatch.init_new when SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1)"
            )
        batch_next_token_ids = self._token_oracle_manager.sample_next_tokens(
            generalized_req_ids=select_generalized_req_ids(
                vanilla_req_ids=vanilla_req_ids,
                bootstrap_room_ids_int=sampling_info.bootstrap_room_ids_int,
            ),
            logits_positions=positions,
        )

        batch_next_token_ids = maybe_perturb_swap_next_tokens(batch_next_token_ids)
        return batch_next_token_ids
