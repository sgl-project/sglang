# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral defaults for speculative runtime hooks.

The runtime owns verification and all cross-rank exchange. Providers may replace
only the pure local-key policy. Missing providers preserve the exact default by
identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import (
    LocalKeysFn,
    PostVerifyIO,
    PostVerifyOutput,
    PostVerifyStrategy,
    ShardedLogitsView,
    SpeculativeRuntimeHooks,
)


def default_local_keys(view: ShardedLogitsView) -> torch.Tensor | None:
    """Build local keys, or decline before any fallback RNG is consumed."""
    try:
        from sglang.srt.speculative.sharded_verify import shard_keys
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.speculative.sharded_verify":
            raise
        # The sharded verifier is a separate runtime optimization. Binding can
        # land first and must preserve the stock gathered path until it exists.
        return None

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
    from sglang.srt.speculative import eagle_utils

    batch = SimpleNamespace(
        device=io.device,
        forward_mode=SimpleNamespace(is_idle=lambda: io.is_idle),
        seq_lens=io.seq_lens,
        sampling_info=io.sampling_info,
        req_pool_indices=io.req_pool_indices,
        return_logprob=io.return_logprob,
    )
    kwargs = dict(
        uno_target_max_top_k=io.sparse_target_max_top_k,
        watermark_state=io.watermark,
        watermark_full_mask=io.watermark_full_mask,
        simulated_acc_len=io.simulated_accept_length,
        simulated_acc_len_rng=io.simulated_accept_length_rng,
        sparse_bonus_argmax=io.sparse_bonus_argmax,
    )
    if getattr(eagle_utils, "LOCAL_KEYS_SUPPORTED", False):
        kwargs["local_keys"] = local_keys
    token_ids, accepted_lengths, accepted_indices = eagle_utils.eagle_sample(
        io.verify_tree,
        batch,
        io.logits,
        io.grammar_mask,
        **kwargs,
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
