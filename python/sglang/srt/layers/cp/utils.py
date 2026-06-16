# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Public import facade and runtime helpers for context parallel strategies."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)


def maybe_prepare_cp_forward(forward_batch) -> None:
    """Build CP-v2 metadata for a prefill batch when the strategy can apply."""
    strategy = get_cp_strategy()
    if strategy is None:
        return

    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return

    num_tokens = len(input_ids)
    if not strategy.can_apply(num_tokens, forward_batch):
        return

    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    forward_batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens=num_tokens,
        seqs_len=seq_lens_cpu,
        extend_seqs_len=extend_lens_cpu,
    )
    forward_batch.cp_v2_active = True


def maybe_cp_split_before_forward(
    model: Any,
    forward_batch,
    kwargs: Dict[str, Any],
) -> Optional[Any]:
    """Shard embeddings and positions for CP-v2 model-runner forwarding."""
    strategy = get_cp_strategy()
    if (
        strategy is None
        or getattr(forward_batch, "attn_cp_metadata", None) is None
        or not getattr(forward_batch, "cp_v2_active", False)
    ):
        return None

    input_embeds = kwargs.get("input_embeds")
    if input_embeds is None:
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(forward_batch.input_ids)
    kwargs["input_embeds"] = strategy.shard_hidden_states(input_embeds, forward_batch)
    return strategy.shard_position_ids(forward_batch.positions, forward_batch)


@contextmanager
def disable_legacy_cp_during_cp_v2(forward_batch):
    """Prevent model-local legacy CP hooks from double-sharding CP-v2 inputs."""
    if not getattr(forward_batch, "cp_v2_active", False):
        yield
        return

    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    old_enable_prefill_context_parallel = server_args.enable_prefill_context_parallel
    server_args.enable_prefill_context_parallel = False
    try:
        yield
    finally:
        server_args.enable_prefill_context_parallel = (
            old_enable_prefill_context_parallel
        )


def _to_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "ContextParallelMetadata",
    "ContextParallelStrategy",
    "ContextParallelStrategyKind",
    "InterleaveCPStrategy",
    "InterleaveContextParallelMetadata",
    "ZigzagCPStrategy",
    "ZigzagContextParallelMetadata",
    "disable_legacy_cp_during_cp_v2",
    "get_cp_strategy",
    "maybe_cp_split_before_forward",
    "maybe_prepare_cp_forward",
]
