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
from typing import TYPE_CHECKING, Any, Optional, Tuple

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
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
    {
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "MiMoV2FlashForCausalLM",
        "MiMoV2ForCausalLM",
        "Qwen3MoeForCausalLM",
        "DeepseekV3ForCausalLM",
    }
)


def is_glm_dsa_cache_layer_split_enabled(model_runner: "ModelRunner") -> bool:
    """Whether DSA GPU KV/indexer cache layers are sharded across CP ranks.

    Layer split is a prefill-CP-only optimization for DSA (DeepSeek Sparse
    Attention) MLA models (e.g. GLM-5.2). Draft workers keep the full cache.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )


def get_glm_dsa_cp_layer_shard_info(
    model_runner: "ModelRunner",
) -> Tuple[Optional[int], int]:
    """Return ``(layer_shard_rank, layer_shard_size)`` for the DSA KV pool.

    ``(None, 1)`` disables sharding (feature off or only one CP rank).
    """
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return None, 1
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return None, 1
    return get_parallel().attn_cp_rank, shard_size


def get_glm_dsa_layer_split_effective_num_layers(
    model_runner: "ModelRunner", num_layers: int
) -> int:
    """Per-rank owned layer count used when sizing the DSA KV cell.

    Under layer split each CP rank only stores ``ceil(num_layers / shard_size)``
    layers, plus one extra layer for the remote scratch buffer used when reading
    a layer owned by another CP rank.
    """
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return num_layers
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return num_layers
    owned_layers_upper_bound = (num_layers + shard_size - 1) // shard_size
    return max(1, owned_layers_upper_bound + 1)


def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> Tuple[int, int]:
    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.

    Layers are split as evenly as possible; the first ``total_layers %
    shard_size`` ranks own one extra layer.
    """
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
    """CP rank that owns ``local_layer_idx`` under the contiguous split."""
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
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

    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    num_tokens = (
        sum(extend_lens_cpu)
        if extend_lens_cpu is not None
        else len(forward_batch.input_ids)
    )
    if forward_batch.attn_cp_metadata is None:
        forward_batch.attn_cp_metadata = strategy.build_metadata(
            num_tokens=num_tokens,
            seqs_len=seq_lens_cpu,
            extend_seqs_len=extend_lens_cpu,
        )
        pad_logical_token_to_physical(forward_batch.attn_cp_metadata)

    if getattr(forward_batch, "global_num_tokens_cpu", None) is not None:
        from sglang.srt.layers.dp_attention import set_local_dp_buffer_len

        # TODO: Check why this is needed
        set_local_dp_buffer_len(
            sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
        )

    if getattr(forward_batch, "out_cache_loc", None) is not None:
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]


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
        # MiMo's text-only body returns (hidden_states, None); logits expects a tensor.
        if len(rest) == 1 and rest[0] is None:
            return hidden_states
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
    "is_glm_dsa_cache_layer_split_enabled",
    "get_glm_dsa_cp_layer_shard_info",
    "get_glm_dsa_layer_split_effective_num_layers",
    "get_layer_shard_range",
    "get_layer_owner",
]
