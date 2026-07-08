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
from typing import Any, Optional, Tuple

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

CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
    {
        "DeepseekV32ForCausalLM",
        "Qwen3MoeForCausalLM",
    }
)


def enable_cp_v2() -> bool:
    """Return whether the CP-v2 path is enabled for this process."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_CP_V2.get())


def is_cp_v2_active(forward_batch) -> bool:
    """Return whether the current forward batch is running through CP-v2."""
    if not enable_cp_v2():
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False

    strategy = get_cp_strategy()
    if strategy is None:
        return False

    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False

    return strategy.can_apply(len(input_ids), forward_batch)


def prepare_cp_forward(forward_batch) -> None:
    """Build CP-v2 metadata for an active context-parallel prefill batch."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    num_tokens = len(forward_batch.input_ids)

    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    forward_batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens=num_tokens,
        seqs_len=seq_lens_cpu,
        extend_seqs_len=extend_lens_cpu,
    )


def cp_split_before_forward(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Shard embeddings and positions for CP-v2 model-runner forwarding."""
    assert is_cp_v2_active(forward_batch)
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return (
        cp_shard_hidden_states(complete_hidden_states, forward_batch),
        cp_shard_position_ids(complete_position_ids, forward_batch),
    )


def cp_shard_hidden_states(complete_hidden_states: Any, forward_batch):
    """Shard a CP-v2 token-major hidden-state tensor without changing positions."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_hidden_states(complete_hidden_states, forward_batch)


def cp_shard_position_ids(complete_position_ids: Any, forward_batch):
    """Shard CP-v2 position ids without changing hidden states."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_position_ids is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_position_ids(complete_position_ids, forward_batch)


def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
    """Gather CP-v2 hidden states at the model boundary when this batch is active."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    if isinstance(x, tuple):
        hidden_states, *rest = x
        hidden_states = strategy.gather_hidden_states(
            hidden_states, forward_batch, stream
        )
        return (hidden_states, *rest)

    return strategy.gather_hidden_states(x, forward_batch, stream)


@contextmanager
def cp_shard_model_inputs(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
):
    """Shard all model inputs for CP-v2 at the runner boundary.

    Yields ``(sharded_hidden_states, sharded_positions)``. ``spec_info.hidden_states``
    is sharded in-place (the model reads it via ``forward_batch``, not as an
    argument) and restored on exit, so the model stays CP-agnostic. This mirrors
    the backup/restore pattern already used by the EAGLE cuda-graph runners.
    """
    assert is_cp_v2_active(forward_batch)
    sharded_hidden_states = cp_shard_hidden_states(
        complete_hidden_states, forward_batch
    )
    sharded_positions = cp_shard_position_ids(complete_position_ids, forward_batch)

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_hidden_states = getattr(spec_info, "hidden_states", None)
    spec_hidden_states_backup = None
    if (
        spec_hidden_states is not None
        and spec_hidden_states.shape[0] == complete_hidden_states.shape[0]
    ):
        spec_hidden_states_backup = spec_hidden_states
        spec_info.hidden_states = cp_shard_hidden_states(
            spec_hidden_states, forward_batch
        )

    try:
        yield sharded_hidden_states, sharded_positions
    finally:
        if spec_hidden_states_backup is not None:
            spec_info.hidden_states = spec_hidden_states_backup


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
    "CP_V2_DEFAULT_MODEL_CLASSES",
    "enable_cp_v2",
    "get_cp_strategy",
    "is_cp_v2_active",
    "cp_gather_after_forward",
    "cp_shard_hidden_states",
    "cp_shard_model_inputs",
    "cp_shard_position_ids",
    "cp_split_before_forward",
    "prepare_cp_forward",
]
