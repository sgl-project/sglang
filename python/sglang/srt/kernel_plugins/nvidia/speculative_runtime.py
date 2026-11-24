# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral defaults for speculative runtime hooks.

The runtime owns verification and all cross-rank exchange. Providers may replace
only the pure local-key policy. Missing providers preserve the exact default by
identity.
"""

from __future__ import annotations

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import (
    ShardedLogitsView,
    SpeculativeRuntimeHooks,
)


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


def default_speculative_runtime_hooks() -> SpeculativeRuntimeHooks:
    """SGLang's stock hooks: the default local key builder."""
    return SpeculativeRuntimeHooks(local_keys=default_local_keys)
