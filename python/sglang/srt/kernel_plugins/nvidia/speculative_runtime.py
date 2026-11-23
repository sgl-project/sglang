# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral defaults for speculative runtime hooks.

The runtime owns verification and all cross-rank exchange. Providers may replace
only the pure local-key policy. Missing providers preserve the exact default by
identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.plugin_kernels.nvidia.spec import (
    LocalKeysFn,
    PostVerifyIO,
    PostVerifyOutput,
    PostVerifyStrategy,
    ShardedLogitsView,
    SpeculativeRuntimeHooks,
    VerifySamplingView,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_mode import ForwardMode


class StockVerifyBatch(msgspec.Struct, frozen=True):
    """The ScheduleBatch fields eagle_sample and VerifyStep.adjust read, and no
    others."""

    device: torch.device
    forward_mode: ForwardMode
    seq_lens: torch.Tensor
    sampling_info: VerifySamplingView
    req_pool_indices: torch.Tensor
    return_logprob: bool


def default_local_keys(view: ShardedLogitsView) -> torch.Tensor | None:
    """Build local keys, or decline before any fallback RNG is consumed."""
    from sglang.srt.speculative.sharded_verify import shard_keys

    # Temperature sampling is an explicit provider numeric policy.  The stock
    # default never selects the distribution-equivalent Gumbel path, even if a
    # caller happens to supply seed material.
    if view.temperatures is not None:
        return None
    return shard_keys(
        view.values,
        view.shard,
        temperatures=view.temperatures,
        seeds=view.seeds,
        splits=view.splits,
    )


def default_post_verify(
    io: PostVerifyIO,
    local_keys: LocalKeysFn,
) -> PostVerifyOutput:
    """Run the existing verifier and preserve its outputs and mutation order."""
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.sampling.sampling_pipeline import VerifyRoute, VerifyStep
    from sglang.srt.speculative import eagle_utils

    batch = StockVerifyBatch(
        device=io.device,
        forward_mode=ForwardMode.IDLE if io.is_idle else ForwardMode.TARGET_VERIFY,
        seq_lens=io.seq_lens,
        sampling_info=io.sampling_info,
        req_pool_indices=io.req_pool_indices,
        return_logprob=io.return_logprob,
    )
    verify = VerifyStep(
        batch=batch,
        route=VerifyRoute.derive(
            batch.sampling_info,
            has_grammar=io.grammar_mask is not None,
            has_watermark=io.watermark is not None,
        ),
        watermark=io.watermark,
        watermark_full_mask=io.watermark_full_mask,
        grammar_mask=io.grammar_mask,
    )
    token_ids, accepted_lengths, accepted_indices = eagle_utils.eagle_sample(
        io.verify_tree,
        verify,
        io.logits,
        uno_target_max_top_k=io.sparse_target_max_top_k,
        simulated_acc_len=io.simulated_accept_length,
        simulated_acc_len_rng=io.simulated_accept_length_rng,
        sparse_bonus_argmax=io.sparse_bonus_argmax,
        local_keys=local_keys,
    )
    return PostVerifyOutput(
        token_ids=token_ids,
        accepted_lengths=accepted_lengths,
        accepted_indices=accepted_indices,
    )


def default_speculative_runtime_hooks() -> SpeculativeRuntimeHooks:
    """Build the stock verification strategy."""
    return SpeculativeRuntimeHooks(
        post_verify=PostVerifyStrategy(
            run=default_post_verify,
            local_keys=default_local_keys,
        ),
    )
