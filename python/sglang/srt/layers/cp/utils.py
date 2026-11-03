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

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from itertools import accumulate
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple

import msgspec
import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
    is_cp_enabled,
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
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel, get_schedule, uses_mla_backend
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.nvtx_pytorch_hooks import profile_method

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def get_cp_sharded_padded(value):
    """Convert a global KV count to local storage for sharded CP storage."""
    strategy = get_cp_strategy()
    return value // strategy.cp_size


def from_cp_sharded(value):
    """Convert a local KV count back to global for sharded CP storage."""
    strategy = get_cp_strategy()
    return value * strategy.cp_size


def get_cp_sharded_padded_cache_indices(value):
    """Recover complete reserved local pages from actual cache-group rows."""
    strategy = get_cp_strategy()
    c = strategy.cp_size
    p = get_schedule().page_size
    # The interval starts at a group boundary; its last group may be partial.
    representatives = value[:: c * p]
    pages = (
        get_cp_sharded_padded_cache_transfer_indices(representatives.to(torch.int64))
        // p
    )
    local = pages[:, None] * p + torch.arange(p, device=value.device)
    return torch.where(
        representatives[:, None] > 0,
        local.to(value.dtype),
        representatives[:, None],
    ).reshape(-1)


def from_cp_sharded_expanded_indices(value):
    """Expand complete local pages into every CP column in cache-row order."""
    strategy = get_cp_strategy()
    c = strategy.cp_size
    p = get_schedule().page_size
    pages = value[::p] // p
    columns = torch.arange(c, device=value.device)
    offsets = torch.arange(p, device=value.device)
    return (
        ((pages[:, None, None] - 1) * c + 1 + columns[None, :, None]) * p
        + offsets[None, None, :]
    ).reshape(-1)


def _ceil_group_count(value, group_size):
    """Count complete groups needed to cover a token endpoint."""
    return (value + group_size - 1) // group_size


def get_cp_sharded_padded_allocation_tokens(value, *, prefix_tokens):
    """Convert raw request tokens to fresh KV capacity for sharded CP storage."""
    strategy = get_cp_strategy()
    page_size = get_schedule().page_size
    group_size = strategy.cp_size * page_size
    return (
        _ceil_group_count(prefix_tokens + value, group_size) * group_size
        - _ceil_group_count(prefix_tokens, group_size) * group_size
    )


def get_cp_sharded_padded_cache_transfer_indices(value):
    """Project cache rows to local rows, preserving shape, dtype and sentinels."""
    strategy = get_cp_strategy()
    c = strategy.cp_size
    p = get_schedule().page_size
    shifted = value.to(torch.int64) + (c - 1) * p
    local = (shifted // (c * p)) * p + shifted % p
    return torch.where(value > 0, local.to(value.dtype), value)


def get_cp_sharded_padded_allocation_lengths(value):
    """Convert request endpoints to reserved local capacity for allocation."""
    strategy = get_cp_strategy()
    page_size = get_schedule().page_size
    group_size = strategy.cp_size * page_size
    return _ceil_group_count(value, group_size) * page_size


def get_cp_sharded_padded_allocation_size(value, *, prefix_lens, extend_lens):
    """Count fresh local rows from the existing CPU request endpoints."""
    strategy = get_cp_strategy()
    page_size = get_schedule().page_size
    seq_lens = prefix_lens + extend_lens
    group_size = strategy.cp_size * page_size
    pages = _ceil_group_count(seq_lens, group_size) - _ceil_group_count(
        prefix_lens, group_size
    )
    return int(pages.sum()) * page_size


def get_cp_sharded_padded_chunk_tokens(value, *, prefix_tokens):
    """Align a chunk to complete fresh groups, retaining its reserved prefix tail."""
    strategy = get_cp_strategy()
    page_size = get_schedule().page_size
    group_size = strategy.cp_size * page_size
    tail = _ceil_group_count(prefix_tokens, group_size) * group_size - prefix_tokens
    groups = max(0, int((value - tail) // group_size))
    return min(max(value, 0), tail + groups * group_size)


def get_cp_sharded_padded_available_tokens(value, *, reserved_tokens=0):
    """Adapt free capacity before the existing chunk headroom subtraction.

    The resource group is C times the strategy's physical token page size.
    Without a prefix, allow at most one partial group's tail. A request-aware
    geometry entry bounds that allowance by the actual reserved tail.
    """
    strategy = get_cp_strategy()
    if value < 0:
        return value
    group_size = strategy.cp_size * get_schedule().page_size
    return (
        reserved_tokens
        + max(0, (value - reserved_tokens) // group_size) * group_size
        + group_size
        - 1
    )


def is_glm_dsa_cache_layer_split_enabled(model_runner: ModelRunner) -> bool:
    """Whether DSA GPU KV/indexer cache layers are sharded across CP ranks.

    Layer split is a prefill-CP-only optimization for DSA (DeepSeek Sparse
    Attention) MLA models (e.g. GLM-5.2). Draft workers keep the full cache.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    return (
        not model_runner.is_draft_worker
        and get_parallel().enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )


def get_glm_dsa_cp_layer_shard_info(
    model_runner: ModelRunner,
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
    model_runner: ModelRunner, num_layers: int
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


def is_cp_active(forward_batch) -> bool:
    """Return whether the current forward batch is running through CP."""
    if is_sharded_cp_enabled():
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


def is_mla_cp_enabled() -> bool:
    """Return whether prefill CP is configured for an MLA attention backend."""
    return is_cp_enabled() and uses_mla_backend()


def is_mla_cp_active(forward_batch) -> bool:
    """Return whether this MLA forward batch is using prefill CP."""
    return is_mla_cp_enabled() and is_cp_active(forward_batch)


def prepare_cp_forward(forward_batch) -> None:
    """Build CP metadata for an active context-parallel prefill batch."""
    assert is_cp_active(forward_batch)
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

    # Under CP-TP group sharing the model owns the CP row layout and gathers
    # full rows for its MLPs, so their buffers keep the full-batch length.
    if (
        not get_parallel().enable_cp_tp_group_sharing
        and getattr(forward_batch, "global_num_tokens_cpu", None) is not None
    ):
        from sglang.srt.layers.dp_attention import set_local_dp_buffer_len

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
    """Shard embeddings and positions for CP model-runner forwarding."""
    assert is_cp_active(forward_batch)
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return (
        cp_shard_hidden_states(complete_hidden_states, forward_batch),
        cp_shard_position_ids(complete_position_ids, forward_batch),
    )


def cp_shard_hidden_states(complete_hidden_states: Any, forward_batch):
    assert is_cp_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_hidden_states(complete_hidden_states, forward_batch)


def cp_gather_full_sequence_states(sharded_states, forward_batch, stream=None):
    """Gather CP-sharded states into the full logical sequence order."""
    metadata = getattr(forward_batch, "prefill_cp_metadata", None)
    if metadata is not None:
        output = metadata.token_collectives().gather(sharded_states)
        forward_batch.num_token_non_padded = metadata.original_non_padded
        return output
    assert is_cp_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert sharded_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.gather_hidden_states(sharded_states, forward_batch, stream)


def cp_shard_position_ids(complete_position_ids: Any, forward_batch):
    metadata = getattr(forward_batch, "prefill_cp_metadata", None)
    if metadata is not None:
        return complete_position_ids.index_select(-1, metadata.query_indices)
    assert is_cp_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_position_ids is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_position_ids(complete_position_ids, forward_batch)


def cp_local_rope_positions(forward_batch, positions: Any, local_k: Any) -> Any:
    """Return the CP-local positions matching the local K shard, row for row.

    ``forward_batch.positions`` stays global in every mode (eager CP and
    breakable prefill graph capture/replay); the shard is taken here from the
    CP metadata, which BCG rewrites to the captured per-rank geometry, so the
    result is padded to the same physical rows as the local K shard.
    """
    assert is_cp_active(forward_batch)
    positions = cp_shard_position_ids(positions, forward_batch)
    assert positions.shape[0] == local_k.shape[0], (
        "CP prefill: rope positions do not match the local K shard "
        f"(positions={positions.shape[0]}, k={local_k.shape[0]})"
    )
    return positions


def cp_live_store_rows(forward_batch, gathered_rows: int, what: str) -> int:
    """Rows to write to the KV cache after a CP all-gather: the live tokens.

    The gathered tensor may carry padded bucket rows at its tail and BCG replay
    hands over a bucket-sized static out_cache_loc; both must cover the live
    rows and only those rows are written.
    """
    assert is_cp_active(forward_batch)
    n_store = int(forward_batch.attn_cp_metadata.total_seq_lens)
    n_loc = int(forward_batch.out_cache_loc.shape[0])
    assert n_store <= gathered_rows, (
        f"CP prefill ({what}): gathered rows {gathered_rows} < live tokens {n_store}"
    )
    assert n_store <= n_loc, (
        f"CP prefill ({what}): out_cache_loc has {n_loc} rows < live tokens "
        f"{n_store}; padding must sit at the tail of out_cache_loc"
    )
    return n_store


def cp_interleave_input_ids(input_ids: Any, forward_batch):
    assert is_cp_active(forward_batch)
    if not get_moe_a2a_backend().is_none():
        return cp_shard_hidden_states(input_ids, forward_batch)

    physical_tokens = sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
    padded_input_ids = input_ids.new_zeros(physical_tokens)
    padded_input_ids[: input_ids.shape[0]] = input_ids
    return padded_input_ids.view(-1, get_parallel().attn_cp_size).T.flatten()


def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
    """Gather CP hidden states at the model boundary when this batch is active."""
    assert is_cp_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    if isinstance(x, tuple):
        gathered = tuple(
            (
                strategy.gather_hidden_states(item, forward_batch, stream)
                if item is not None
                else None
            )
            for item in x
        )
        # MiMo's text-only body returns (hidden_states, None); logits expects a tensor.
        if len(gathered) == 2 and gathered[1] is None:
            return gathered[0]
        return gathered

    return strategy.gather_hidden_states(x, forward_batch, stream)


def cp_materialize_global_token_order(
    x: Any, forward_batch, stream: Optional[Any] = None
):
    """Materialize a CP tensor in the global logical token order."""
    assert is_cp_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    return strategy.gather_kv_cache(x, forward_batch, stream)


@contextmanager
def cp_shard_model_inputs(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
    complete_input_ids: Optional[Any] = None,
):
    """Restore the shared batch so logits processing keeps full-batch metadata."""
    assert is_cp_active(forward_batch)
    sharded_hidden_states = (
        cp_shard_hidden_states(complete_hidden_states, forward_batch)
        if complete_hidden_states is not None
        else None
    )
    sharded_positions = cp_shard_position_ids(complete_position_ids, forward_batch)
    model_input_ids = (
        cp_shard_hidden_states(complete_input_ids, forward_batch)
        if complete_input_ids is not None
        else None
    )

    had_input_ids_global = hasattr(forward_batch, "input_ids_global")
    input_ids_global_backup = getattr(forward_batch, "input_ids_global", None)
    if complete_input_ids is not None:
        forward_batch.input_ids_global = cp_interleave_input_ids(
            complete_input_ids, forward_batch
        )

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_hidden_states = getattr(spec_info, "hidden_states", None)
    spec_hidden_states_backup = None
    if (
        spec_hidden_states is not None
        and complete_hidden_states is not None
        and spec_hidden_states.shape[0] == complete_hidden_states.shape[0]
    ):
        spec_hidden_states_backup = spec_hidden_states
        spec_info.hidden_states = cp_shard_hidden_states(
            spec_hidden_states, forward_batch
        )

    try:
        yield sharded_hidden_states, sharded_positions, model_input_ids
    finally:
        if spec_hidden_states_backup is not None:
            spec_info.hidden_states = spec_hidden_states_backup
        if had_input_ids_global:
            forward_batch.input_ids_global = input_ids_global_backup
        elif hasattr(forward_batch, "input_ids_global"):
            delattr(forward_batch, "input_ids_global")


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
    "get_cp_strategy",
    "is_cp_active",
    "is_mla_cp_enabled",
    "is_mla_cp_active",
    "cp_gather_after_forward",
    "cp_materialize_global_token_order",
    "cp_interleave_input_ids",
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


logger = logging.getLogger(__name__)


def _cp_metadata_tensor(data, *, device, dtype):
    # NumPy constructs integer index arrays without PyTorch walking boxed Python
    # integers. Keep planning CPU-only; upload() packs the readonly indices later.
    if dtype in (torch.int32, torch.int64):
        array = np.asarray(data, dtype=np.int32 if dtype == torch.int32 else np.int64)
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.as_tensor(data, dtype=dtype)
    if torch.device(device).type == "cpu":
        return tensor
    if is_pin_memory_available(device):
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=True)


@dataclass
class ShardedContextParallelMetadata(BaseContextParallelMetadata):
    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None

    # metadata for attention
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None

    total_seq_lens: torch.Tensor = None

    # Token ownership and reusable plans for hybrid, sharded-KV prefill.
    group: object = None
    rank: int = 0
    world: int = 1
    moe_tp: bool = False
    device: torch.device | None = None
    lengths: tuple = ()
    prefixes: tuple = ()
    contiguous: tuple = ()
    attention: tuple = ()
    layout: CPAttentionLayout | None = None
    token_layouts: dict = field(default_factory=dict)
    layout_names: dict = field(default_factory=dict)
    routes: dict = field(default_factory=dict)
    gather_scatter_plans: dict = field(default_factory=dict)
    local_token_counts: dict = field(default_factory=dict)
    moe_output_plans: dict = field(default_factory=dict)
    current: str | None = None
    residual_exchange: TokenExchangeStream | None = None
    queued_residual: tuple | None = None
    pending_residual: tuple | None = None
    mlp_destination: str | None = None
    original_non_padded: torch.Tensor | None = None
    total_non_padded: torch.Tensor | None = None
    query_indices: torch.Tensor | None = None
    kv_write_indices: torch.Tensor | None = None
    attention_segments: tuple = ()
    gdn_conv_metadata: tuple = ()
    kv_read_indices: torch.Tensor | None = None
    kv_read_slots: torch.Tensor | None = None
    kv_write_slots: torch.Tensor | None = None
    device_buffers: tuple = ()

    @classmethod
    def from_prefill(
        cls,
        batch,
        device,
        *,
        cp_mode="in-seq-split",
        residual_exchange=None,
        previous=None,
    ):
        """Build a frame for direct backend callers without a scheduling phase."""
        metadata = cls.plan_prefill(
            batch.extend_seq_lens_cpu,
            batch.extend_prefix_lens_cpu,
            device,
            cp_mode=cp_mode,
            residual_exchange=residual_exchange,
            previous=previous,
        )
        metadata.original_non_padded = batch.num_token_non_padded
        metadata.upload()
        return metadata

    @classmethod
    @profile_method("sglang.cp.plan_metadata_cpu")
    def plan_prefill(
        cls,
        lengths,
        prefixes,
        device,
        *,
        cp_mode="in-seq-split",
        residual_exchange=None,
        previous=None,
        request_indices=None,
        request_stride=None,
        packed_kv=False,
        needs_kv_descale=True,
    ):
        from sglang.srt.distributed import (
            get_attn_cp_group,
            get_moe_tensor_parallel_world_size,
        )

        metadata = cls()
        if cp_mode not in ("in-seq-split", "round-robin-split"):
            raise ValueError(f"Unsupported prefill CP mode: {cp_mode}")
        # Full attention follows the existing CP split option. GDN's recurrence
        # requires the contiguous layout independently of the full-attention mode.
        query_layout = "zigzag" if cp_mode == "in-seq-split" else "interleaved"
        cp_group = get_attn_cp_group()
        metadata.group = cp_group.device_group
        metadata.rank, metadata.world = cp_group.rank_in_group, cp_group.world_size
        metadata.moe_tp = get_moe_tensor_parallel_world_size() > 1
        metadata.device = device
        metadata.lengths = tuple(map(int, lengths))
        metadata.prefixes = tuple(map(int, prefixes))
        metadata.contiguous = contiguous_chunk_indices(metadata.lengths, metadata.world)
        metadata.layout = make_cp_attention_layout(
            seq_lens=tuple(p + n for p, n in zip(metadata.prefixes, metadata.lengths)),
            query_lens=metadata.lengths,
            world_size=metadata.world,
            query_layout=query_layout,
        )
        metadata.attention = metadata.layout.query_indices
        metadata.token_layouts = {
            "contiguous": metadata.contiguous,
            "attention": metadata.attention,
        }
        plan_layouts = dict(metadata.token_layouts, kv=metadata.layout.new_kv_indices)
        # Resolve equivalent layouts once. Per-layer plan lookups must not hash
        # every token index in the chunk again.
        canonical_layouts = {}
        for name, indices in plan_layouts.items():
            metadata.layout_names[name] = canonical_layouts.setdefault(indices, name)
        metadata.residual_exchange = residual_exchange
        # Plans contain only token ownership and read-only device indices. Keep
        # one geometry across chunks; request slots, causal lengths and in-flight
        # residuals always belong to this new frame. Do not retain old frames or
        # accumulate a cache for arbitrary request shapes.
        if (
            previous is not None
            and previous.group is metadata.group
            and previous.device == device
            and (previous.rank, previous.world, previous.moe_tp)
            == (metadata.rank, metadata.world, metadata.moe_tp)
            and previous.residual_exchange is residual_exchange
            and previous.contiguous == metadata.contiguous
            and previous.attention == metadata.attention
            and previous.layout.new_kv_indices == metadata.layout.new_kv_indices
        ):
            metadata.routes = previous.routes.copy()
            metadata.gather_scatter_plans = previous.gather_scatter_plans.copy()
            metadata.local_token_counts = previous.local_token_counts.copy()
            metadata.moe_output_plans = previous.moe_output_plans.copy()
            metadata.total_non_padded = previous.total_non_padded
            metadata.query_indices = previous.query_indices
            metadata.kv_write_indices = previous.kv_write_indices
        else:
            metadata.total_non_padded = _cp_metadata_tensor(
                sum(metadata.lengths), device="cpu", dtype=torch.int32
            )
            metadata.query_indices = _cp_metadata_tensor(
                metadata.attention[metadata.rank], device="cpu", dtype=torch.long
            )
            metadata.kv_write_indices = _cp_metadata_tensor(
                metadata.layout.new_kv_indices[metadata.rank],
                device="cpu",
                dtype=torch.long,
            )
        # Both attention layouts are known before any layer runs. Prepare the
        # final output gather as well as intermediate MoE collectives here;
        # a new/ragged chunk must not upload indices at the end of forward.
        for name, indices in metadata.token_layouts.items():
            name = metadata.layout_names[name]
            if name not in metadata.gather_scatter_plans:
                metadata.gather_scatter_plans[name] = make_token_gather_scatter(
                    indices=indices, group=metadata.group, device="cpu"
                )
            count = len(indices[metadata.rank])
            if count not in metadata.local_token_counts:
                metadata.local_token_counts[count] = _cp_metadata_tensor(
                    count, device="cpu", dtype=torch.int32
                )
        for source, target in (
            ("contiguous", "attention"),
            ("attention", "contiguous"),
            ("attention", "kv"),
        ):
            key = (metadata.layout_names[source], metadata.layout_names[target], False)
            if key not in metadata.routes:
                metadata.routes[key] = make_token_exchange(
                    source=plan_layouts[source],
                    target=plan_layouts[target],
                    group=metadata.group,
                    device="cpu",
                )
        if metadata.moe_tp and residual_exchange is not None:
            for source, target in (
                ("contiguous", "attention"),
                ("attention", "contiguous"),
            ):
                key = (metadata.layout_names[source], metadata.layout_names[target])
                if key[0] != key[1] and key not in metadata.moe_output_plans:
                    metadata.moe_output_plans[key] = make_token_reduce_scatter(
                        source=plan_layouts[source],
                        target=plan_layouts[target],
                        group=metadata.group,
                        device="cpu",
                    )

        metadata.attention_segments = make_cp_attention_segments(
            layout=metadata.layout,
            rank=metadata.rank,
            device="cpu",
            packed_kv=packed_kv,
            needs_kv_descale=needs_kv_descale,
        )
        if request_indices is not None:
            if len(request_indices) != len(metadata.lengths) or request_stride is None:
                raise ValueError("CP request slots must match sequence metadata")
            offsets = [
                np.arange(metadata.rank, length, metadata.world, dtype=np.int64)
                + slot * request_stride
                for slot, length in zip(request_indices, metadata.layout.seq_lens)
            ]
            metadata.kv_read_indices = torch.from_numpy(np.concatenate(offsets))
        return metadata

    @profile_method("sglang.cp.upload_metadata")
    def upload(self):
        """Pack aligned readonly metadata into one pinned H2D allocation.

        Reused device plans remain immutable. CPU-only lengths needed by packed
        attention stay on the host; request activations and state are not copied.
        """
        tensors = {}

        def collect(value):
            if isinstance(value, torch.Tensor):
                if value.device.type == "cpu":
                    tensors.setdefault(id(value), value)
            elif isinstance(value, msgspec.Struct):
                for name in value.__struct_fields__:
                    if not name.endswith("_cpu"):
                        collect(getattr(value, name))
            elif isinstance(value, (tuple, dict)):
                for item in value.values() if isinstance(value, dict) else value:
                    collect(item)

        names = (
            "total_non_padded",
            "query_indices",
            "kv_write_indices",
            "kv_read_indices",
            "routes",
            "gather_scatter_plans",
            "local_token_counts",
            "moe_output_plans",
            "attention_segments",
            "gdn_conv_metadata",
        )
        for name in names:
            collect(getattr(self, name))
        if torch.device(self.device).type == "cpu":
            return
        # Match the 16-byte pointer alignment of separately allocated warmup
        # tensors. A weaker alignment creates new Triton specializations during
        # serving, whose module loading can block behind queued CP collectives.
        positions, size = {}, 0
        for key, value in tensors.items():
            size = (size + 15) // 16 * 16
            positions[key] = size
            size += value.numel() * value.element_size()
        host = torch.empty(
            size, dtype=torch.uint8, pin_memory=is_pin_memory_available(self.device)
        )
        for key, value in tensors.items():
            start = positions[key]
            end = start + value.numel() * value.element_size()
            host[start:end].view(value.dtype).copy_(value.reshape(-1))
        device = host.to(self.device, non_blocking=True)
        copies = {}
        for key, value in tensors.items():
            start = positions[key]
            end = start + value.numel() * value.element_size()
            copies[key] = device[start:end].view(value.dtype).view(value.shape)

        def replace(value):
            if isinstance(value, torch.Tensor):
                return copies.get(id(value), value)
            if isinstance(value, msgspec.Struct):
                return msgspec.structs.replace(
                    value,
                    **{
                        name: replace(getattr(value, name))
                        for name in value.__struct_fields__
                        if not name.endswith("_cpu")
                    },
                )
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            return value

        for name in names:
            setattr(self, name, replace(getattr(self, name)))
        # Retain the full allocations and record their forward-stream use below.
        self.device_buffers = (device,)

    @profile_method("sglang.cp.resolve_kv_slots")
    def resolve_kv_slots(self, req_table, out_cache_loc):
        """Read the authoritative GPU slot table; never copy it back to the host."""
        self.kv_read_slots = (
            req_table.reshape(-1).index_select(0, self.kv_read_indices).long()
            // self.world
        )
        self.kv_write_slots = (
            out_cache_loc.index_select(0, self.kv_write_indices).long() // self.world
        )

    def record_stream(self, stream):
        # Frame ownership survives through the queued forward; cached plans may
        # outlive the frame, but their allocations still need cross-stream tracking.
        def record(value):
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                value.record_stream(stream)
            elif isinstance(value, msgspec.Struct):
                for name in value.__struct_fields__:
                    if not name.endswith("_cpu"):
                        record(getattr(value, name))
            elif isinstance(value, (tuple, dict)):
                for item in value.values() if isinstance(value, dict) else value:
                    record(item)

        for value in (
            self.device_buffers,
            self.kv_read_slots,
            self.kv_write_slots,
            self.query_indices,
            self.kv_write_indices,
            self.total_non_padded,
            self.routes,
            self.gather_scatter_plans,
            self.local_token_counts,
            self.moe_output_plans,
            self.attention_segments,
            self.gdn_conv_metadata,
        ):
            record(value)

    def exchange_plan(self, source, destination, *, allow_subset=False):
        key = (self.layout_names[source], self.layout_names[destination], allow_subset)
        return self.routes[key]

    def route(self, source, destination, *, allow_subset=False):
        return self.exchange_plan(source, destination, allow_subset=allow_subset).apply

    def local_token_count(self, count):
        return self.local_token_counts[count]

    def token_collectives(self):
        """Reuse device ownership indices across layers with the same layout."""
        if self.current is None:
            raise RuntimeError("Ring CP tokens have not been partitioned")
        return self.gather_scatter_plans[self.current]


def is_sharded_cp_enabled(server_args=None):
    if server_args is not None:
        from sglang.srt.arg_groups.overrides import resolved_view

        args = resolved_view(server_args)
        return (
            args.attn_cp_size > 1
            and args.linear_attn_prefill_backend == "flashqla_cp"
            and args.disaggregation_mode == "prefill"
            and not args.encoder_only
        )
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    return (
        get_parallel().attn_cp_size > 1
        and get_exec().mamba.linear_attn_prefill_backend == "flashqla_cp"
        and get_disagg().disaggregation_mode == "prefill"
        and not get_disagg().encoder_only
    )


class ZigzagLayout(msgspec.Struct, frozen=True):
    """Blocks are sequence-major; rank rows contain all early, then late blocks."""

    block_sizes: tuple[tuple[int, ...], ...]
    split_lens: tuple[int, ...]
    offsets: tuple[int, ...]
    rank_blocks: tuple[tuple[int, ...], ...]
    rank_lens: tuple[int, ...]
    gather_lens: tuple[int, ...]
    restore_blocks: tuple[int, ...]


def make_zigzag_layout(*, lengths: tuple[int, ...], world_size: int) -> ZigzagLayout:
    if world_size < 1 or any(length < 0 for length in lengths):
        raise ValueError("Invalid zigzag world size or sequence length")
    segments = 2 * world_size
    block_sizes = tuple(
        tuple(
            length // segments + (block < length % segments)
            for block in range(segments)
        )
        for length in lengths
    )
    split_lens = tuple(size for sizes in block_sizes for size in sizes)
    rank_blocks = tuple(
        tuple(
            sequence * segments + block
            for block in (rank, segments - 1 - rank)
            for sequence in range(len(lengths))
        )
        for rank in range(world_size)
    )
    gathered = tuple(block for blocks in rank_blocks for block in blocks)
    restore = [0] * len(gathered)
    for index, block in enumerate(gathered):
        restore[block] = index
    return ZigzagLayout(
        block_sizes=block_sizes,
        split_lens=split_lens,
        offsets=(0, *accumulate(split_lens)),
        rank_blocks=rank_blocks,
        rank_lens=tuple(sum(split_lens[b] for b in blocks) for blocks in rank_blocks),
        gather_lens=tuple(split_lens[b] for b in gathered),
        restore_blocks=tuple(restore),
    )


class QueryBlock(msgspec.Struct, frozen=True):
    sequence: int
    local_start: int
    position: int
    length: int
    stride: int = 1


class CPAttentionLayout(msgspec.Struct, frozen=True):
    seq_lens: tuple[int, ...]
    query_lens: tuple[int, ...]
    query_indices: tuple[tuple[int, ...], ...]
    new_kv_indices: tuple[tuple[int, ...], ...]
    blocks: tuple[tuple[QueryBlock, ...], ...]
    kv_lens: tuple[tuple[int, ...], ...]

    @property
    def world_size(self) -> int:
        return len(self.query_indices)

    @property
    def capacity(self) -> int:
        return max(map(sum, self.kv_lens), default=0)


def make_cp_attention_layout(
    *,
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    world_size: int,
    query_layout: str = "zigzag",
) -> CPAttentionLayout:
    if world_size < 1 or len(seq_lens) != len(query_lens):
        raise ValueError("Invalid ring size or batch lengths")
    if any(q < 0 or q > s for s, q in zip(seq_lens, query_lens)):
        raise ValueError("Each query length must lie in [0, sequence length]")
    if query_layout not in ("zigzag", "interleaved"):
        raise ValueError("Ring query layout must be zigzag or interleaved")
    geometry = make_zigzag_layout(lengths=query_lens, world_size=world_size)
    query_offsets = (0, *accumulate(query_lens))
    query_indices, new_kv_indices, blocks, kv_lens = [], [], [], []
    for rank in range(world_size):
        rank_q, rank_kv, rank_blocks, rank_lens = [], [], [], []
        for block in geometry.rank_blocks[rank]:
            sequence = block // (2 * world_size)
            start, end = geometry.offsets[block : block + 2]
            if start != end:
                position = (
                    seq_lens[sequence]
                    - query_lens[sequence]
                    + start
                    - query_offsets[sequence]
                )
                rank_blocks.append(
                    QueryBlock(sequence, len(rank_q), position, end - start)
                )
                rank_q.extend(range(start, end))
        packed_start = 0
        for seq_len, query_len in zip(seq_lens, query_lens):
            prefix = seq_len - query_len
            first = (rank - prefix) % world_size
            rank_kv.extend(
                range(packed_start + first, packed_start + query_len, world_size)
            )
            rank_lens.append(max(0, (seq_len + world_size - 1 - rank) // world_size))
            packed_start += query_len
        if query_layout == "interleaved":
            # Match the persistent KV ownership. Strided queries advance their
            # causal cutoff by exactly one local KV row, avoiding residue groups.
            rank_q = list(rank_kv)
            rank_blocks = []
            local_start = 0
            for sequence, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
                prefix = seq_len - query_len
                first = (rank - prefix) % world_size
                count = max(0, (query_len - first + world_size - 1) // world_size)
                if count:
                    rank_blocks.append(
                        QueryBlock(
                            sequence, local_start, prefix + first, count, world_size
                        )
                    )
                    local_start += count
        query_indices.append(tuple(rank_q))
        new_kv_indices.append(tuple(rank_kv))
        blocks.append(tuple(rank_blocks))
        kv_lens.append(tuple(rank_lens))
    return CPAttentionLayout(
        seq_lens,
        query_lens,
        tuple(query_indices),
        tuple(new_kv_indices),
        tuple(blocks),
        tuple(kv_lens),
    )


class CPQueryBatch(msgspec.Struct, frozen=True):
    """Residue-grouped queries sharing one owner's packed KV.

    All owners use the same query permutation. Each virtual sequence has its
    own KV start and causal end; cache_batch_idx aliases the single KV buffer.
    Packed attention calls materialize those virtual sequences. Empty causal
    prefixes are retained so the permutation stays owner invariant.
    """

    query_indices: torch.Tensor
    restore_indices: torch.Tensor
    cu_q: torch.Tensor
    max_q: int
    cache_batch_idx: torch.Tensor
    cache_leftpad: torch.Tensor
    cache_seqlens: torch.Tensor
    packed_kv_indices: torch.Tensor | None = None
    cu_k: torch.Tensor | None = None
    max_k: int = 0
    packed_kv_sequences: torch.Tensor | None = None
    kv_slice: tuple[int, int] | None = None
    q_lens_cpu: torch.Tensor | None = None
    kv_lens_cpu: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None


@profile_method("sglang.cp.prepare_attention_segments")
def make_cp_attention_segments(
    *,
    layout: CPAttentionLayout,
    rank: int,
    device: torch.device,
    needs_kv_descale: bool = True,
    packed_kv: bool = False,
) -> tuple[CPQueryBatch, ...]:
    device = torch.device(device)
    steps = []
    world = layout.world_size
    groups = [
        (
            block,
            residue,
            (block.length - residue + world // block.stride - 1)
            // (world // block.stride),
        )
        for block in layout.blocks[rank]
        for residue in range(min(world // block.stride, block.length))
    ]
    rows = (
        np.concatenate(
            [
                block.local_start
                + residue
                + np.arange(count, dtype=np.int64) * (world // block.stride)
                for block, residue, count in groups
            ]
        )
        if groups
        else np.empty(0, dtype=np.int64)
    )
    inverse = np.empty(len(rows), dtype=np.int64)
    inverse[rows] = np.arange(len(rows), dtype=np.int64)
    indices = _cp_metadata_tensor(rows, device=device, dtype=torch.long)
    restore = _cp_metadata_tensor(inverse, device=device, dtype=torch.long)
    cu_q = _cp_metadata_tensor(
        [0, *accumulate(count for _, _, count in groups)],
        device=device,
        dtype=torch.int32,
    )
    batch_idx = _cp_metadata_tensor(
        np.zeros(len(groups), dtype=np.int32), device=device, dtype=torch.int32
    )
    for owner in range(world):
        offsets = [0, *accumulate(layout.kv_lens[owner])]
        starts, ends = [], []
        for block, residue, count in groups:
            end = block.position + residue * block.stride + (count - 1) * world
            k_count = max(0, (end - owner) // world + 1)
            # Queries separated by CP size advance the causal KV cutoff by one
            # row. Bottom-right masking therefore preserves global causality.
            start = offsets[block.sequence]
            starts.append(start)
            ends.append(start + k_count)
        lengths = [end - start for start, end in zip(starts, ends)]
        kv_slice = (
            (starts[0], ends[-1])
            if starts and all(a == b for a, b in zip(ends, starts[1:]))
            else None
        )
        steps.append(
            CPQueryBatch(
                indices,
                restore,
                cu_q,
                max((count for _, _, count in groups), default=0),
                batch_idx,
                _cp_metadata_tensor(starts, device=device, dtype=torch.int32),
                _cp_metadata_tensor(ends, device=device, dtype=torch.int32),
                # Blackwell packed interfaces cannot alias cache batches.
                # Hopper keeps the tuned shared-cache FA3 path without packing
                # KV or replacing its automatic split heuristic.
                (
                    _cp_metadata_tensor(
                        [
                            i
                            for start, end in zip(starts, ends)
                            for i in range(start, end)
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                    if packed_kv and kv_slice is None
                    else None
                ),
                (
                    _cp_metadata_tensor(
                        [0, *accumulate(lengths)], device=device, dtype=torch.int32
                    )
                    if packed_kv
                    else None
                ),
                max(lengths, default=0),
                (
                    _cp_metadata_tensor(
                        [j for j, n in enumerate(lengths) for _ in range(n)],
                        device=device,
                        dtype=torch.long,
                    )
                    if packed_kv and needs_kv_descale
                    else None
                ),
                kv_slice,
                (
                    torch.tensor([count for _, _, count in groups], dtype=torch.int32)
                    if packed_kv
                    else None
                ),
                torch.tensor(lengths, dtype=torch.int32) if packed_kv else None,
                (
                    _cp_metadata_tensor(lengths, device=device, dtype=torch.int32)
                    if packed_kv
                    else None
                ),
            )
        )
    return tuple(steps)


def contiguous_token_indices(
    *, tokens: int, world_size: int
) -> tuple[tuple[int, ...], ...]:
    if tokens < 0 or world_size < 1:
        raise ValueError("Invalid token count or world size")
    base, extra = divmod(tokens, world_size)
    cuts = [0, *accumulate(base + (r < extra) for r in range(world_size))]
    return tuple(tuple(range(cuts[r], cuts[r + 1])) for r in range(world_size))


def contiguous_chunk_indices(lengths, world_size, chunk_size=64):
    """Per-request contiguous shards, with complete kernel chunks on each rank.

    Returned indices contain real rows only. Kernel padding is private to the
    attention implementation and never enters expert dispatch or the KV pool.
    """
    offsets = (0, *accumulate(lengths))
    ranks = []
    for rank in range(world_size):
        rows = []
        for start, length in zip(offsets, lengths):
            width = (length + world_size * chunk_size - 1) // (world_size * chunk_size)
            width *= chunk_size
            rows.extend(
                range(
                    start + min(rank * width, length),
                    start + min((rank + 1) * width, length),
                )
            )
        ranks.append(tuple(rows))
    return tuple(ranks)


class TokenExchange(msgspec.Struct, frozen=True):
    """All-to-all plan from source ownership to target ownership and row order."""

    send_indices: torch.Tensor
    restore_indices: torch.Tensor
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    group: object
    source_rows: int = -1
    send_identity: bool = False
    restore_identity: bool = False

    def apply(
        self, value: torch.Tensor, *, group=None, restore_order: bool = True
    ) -> torch.Tensor:
        """Exchange rows, optionally letting the consumer restore their order.

        With restore_order=False, the consumer must read recv[restore_indices].
        This permits a following norm or projection to fuse the row gather.
        """
        source_rows = (
            self.source_rows if self.source_rows >= 0 else sum(self.send_counts)
        )
        if value.shape[0] != source_rows:
            raise ValueError("Token count differs from the exchange plan")
        send = (
            value.contiguous()
            if self.send_identity
            else value.index_select(0, self.send_indices)
        )
        recv = value.new_empty((sum(self.recv_counts), *value.shape[1:]))
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=list(self.recv_counts),
            input_split_sizes=list(self.send_counts),
            group=self.group if group is None else group,
        )
        return (
            recv
            if self.restore_identity or not restore_order
            else recv.index_select(0, self.restore_indices)
        )


@profile_method("sglang.cp.prepare_token_exchange")
def make_token_exchange(
    *,
    source: tuple[tuple[int, ...], ...],
    target: tuple[tuple[int, ...], ...],
    group: dist.ProcessGroup,
    device: torch.device,
    allow_subset: bool = False,
) -> TokenExchange:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if len(source) != world or len(target) != world:
        raise ValueError("Token layout and process group sizes differ")
    source_flat = np.concatenate([np.asarray(row, dtype=np.int64) for row in source])
    target_flat = np.concatenate([np.asarray(row, dtype=np.int64) for row in target])
    source_sorted = np.sort(source_flat)
    target_order = np.argsort(target_flat)
    target_sorted = target_flat[target_order]
    valid_target = (
        np.isin(target_sorted, source_sorted, assume_unique=True).all()
        if allow_subset
        else np.array_equal(source_sorted, target_sorted)
    )
    if (
        np.any(source_sorted[1:] == source_sorted[:-1])
        or np.any(target_sorted[1:] == target_sorted[:-1])
        or not valid_target
    ):
        raise ValueError(
            "Exchange layouts must be permutations of the same unique tokens"
        )
    # Resolve ownership in bulk instead of doing a Python dict lookup for every
    # token on every peer. Unselected source tokens retain owner -1 for subsets.
    positions = np.searchsorted(target_sorted, source_flat)
    included = positions < len(target_sorted)
    included[included] &= target_sorted[positions[included]] == source_flat[included]
    target_owner = np.repeat(np.arange(world), [len(row) for row in target])
    owners = np.full(len(source_flat), -1, dtype=np.int64)
    owners[included] = target_owner[target_order[positions[included]]]
    offsets = np.cumsum([0, *map(len, source)])
    local_owners = owners[offsets[rank] : offsets[rank + 1]]
    send_indices, send_counts, recv_counts, received = [], [], [], []
    for peer in range(world):
        selected = np.flatnonzero(local_owners == peer)
        send_indices.append(selected)
        send_counts.append(len(selected))
        peer_slice = slice(offsets[peer], offsets[peer + 1])
        peer_tokens = source_flat[peer_slice][owners[peer_slice] == rank]
        received.append(peer_tokens)
        recv_counts.append(len(peer_tokens))
    send_indices = np.concatenate(send_indices)
    received = np.concatenate(received)
    received_order = np.argsort(received)
    restore_indices = received_order[
        np.searchsorted(received[received_order], target[rank])
    ]
    return TokenExchange(
        send_indices=_cp_metadata_tensor(send_indices, dtype=torch.long, device=device),
        restore_indices=_cp_metadata_tensor(
            restore_indices, dtype=torch.long, device=device
        ),
        send_counts=tuple(send_counts),
        recv_counts=tuple(recv_counts),
        group=group,
        source_rows=len(source[rank]),
        send_identity=np.array_equal(send_indices, np.arange(len(source[rank]))),
        restore_identity=np.array_equal(restore_indices, np.arange(len(received))),
    )


class PendingTokenExchange:
    """A token transfer whose result is consumed on another CUDA stream."""

    def __init__(self, value, ready, restore_indices=None):
        self.value, self.ready = value, ready
        self.restore_indices = restore_indices

    def wait(self, *, restore_order=True):
        stream = torch.cuda.current_stream(self.value.device)
        stream.wait_event(self.ready)
        self.value.record_stream(stream)
        if restore_order and self.restore_indices is not None:
            self.restore_indices.record_stream(stream)
            return self.value.index_select(0, self.restore_indices)
        return self.value


class TokenExchangeStream:
    """Overlap residual layout work with expert computation.

    Submit after the MoE input gather. Reuse its NCCL group so collectives on
    that group's communication stream run in order: gather, residual exchange,
    output reduce-scatter. Pure EP has no input gather; its next hidden exchange
    follows the residual exchange on the same group. A separate CUDA stream keeps
    residual packing independent of expert compute. The group is borrowed;
    its lifetime remains owned by the caller.
    """

    def __init__(self, *, group, device):
        self.group = group
        self.stream = torch.cuda.Stream(device=device)

    @profile_method("sglang.cp.residual_exchange")
    def submit(self, plan, value, *, ready=None, restore_order=True):
        if ready is None:
            self.stream.wait_stream(torch.cuda.current_stream(value.device))
        else:
            self.stream.wait_event(ready)
        value.record_stream(self.stream)
        plan.send_indices.record_stream(self.stream)
        plan.restore_indices.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            output = plan.apply(value, group=self.group, restore_order=restore_order)
            ready = self.stream.record_event()
        return PendingTokenExchange(
            output,
            ready,
            (
                plan.restore_indices
                if not restore_order and not plan.restore_identity
                else None
            ),
        )


class TokenReduceScatter(msgspec.Struct, frozen=True):
    """Write TP partial outputs directly into the next token ownership layout."""

    input_indices: torch.Tensor
    input_rows: int
    local_rows: int
    padded_rows: int
    group: object
    identity_order: bool = False

    def apply(self, value):
        if value.shape[0] != self.input_rows:
            raise ValueError("Token count differs from the reduce-scatter plan")
        if not self.input_rows:
            return value
        packed = (
            value.contiguous()
            if self.identity_order
            else copy_token_rows(value, self.input_indices)
        )
        output = value.new_empty((self.padded_rows, *value.shape[1:]))
        dist.reduce_scatter_tensor(output, packed, group=self.group)
        return output[: self.local_rows]


def make_token_reduce_scatter(*, source, target, group, device):
    """Match TokenGatherScatter.gather(rank_ordered=True), including padding.

    Balanced gathers expose rank order to the token-local MoE. Uneven gathers
    strip padding and expose packed request order.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if len(source) != world or len(target) != world:
        raise ValueError("Layouts must partition the same packed token rows")
    flat = np.concatenate([np.asarray(rows, dtype=np.int64) for rows in source])
    target_flat = np.concatenate([np.asarray(rows, dtype=np.int64) for rows in target])
    expected = np.arange(len(flat), dtype=np.int64)
    if not (
        np.array_equal(np.sort(flat), expected)
        and np.array_equal(np.sort(target_flat), expected)
    ):
        raise ValueError("Layouts must partition the same packed token rows")
    source_width = max(map(len, source), default=0)
    order = flat if len(flat) == world * source_width else expected
    positions = np.empty(len(flat), dtype=np.int64)
    positions[order] = expected
    width = max(map(len, target), default=0)
    indices = np.full((world, width), -1, dtype=np.int64)
    for peer, rows in enumerate(target):
        indices[peer, : len(rows)] = positions[np.asarray(rows, dtype=np.int64)]
    indices = indices.reshape(-1)
    return TokenReduceScatter(
        input_indices=_cp_metadata_tensor(indices, dtype=torch.long, device=device),
        input_rows=len(flat),
        local_rows=len(target[rank]),
        padded_rows=width,
        group=group,
        identity_order=np.array_equal(indices, expected),
    )


class TokenGatherScatter(msgspec.Struct, frozen=True):
    """All-gather / sum reduce-scatter for explicit token ownership.

    Only communication buffers are padded. By default results follow packed
    request order, including when some CP ranks own no rows. Token-local MoE
    can keep balanced batches in rank order between the paired collectives.
    The reduction consumes TP partial outputs; it must not follow an all-reduce.
    """

    packed_positions: torch.Tensor
    reduce_indices: torch.Tensor
    local_rows: int
    padded_rows: int
    world_size: int
    group: object
    identity_order: bool = False

    def gather(
        self, value: torch.Tensor, *, rank_ordered: bool = False
    ) -> torch.Tensor:
        if value.shape[0] != self.local_rows:
            raise ValueError("Token count differs from the gather plan")
        if self.padded_rows == 0:
            return value
        # Token-local MoE may keep rank order between gather and reduce-scatter.
        # Only balanced layouts can do this without exposing padding as tokens.
        direct = self.packed_positions.numel() == self.world_size * self.padded_rows
        direct = direct and (rank_ordered or self.identity_order)
        if direct:
            gathered = value.new_empty(
                (self.world_size * self.padded_rows, *value.shape[1:])
            )
            dist.all_gather_into_tensor(gathered, value.contiguous(), group=self.group)
            return gathered
        padded = (
            value.contiguous()
            if self.local_rows == self.padded_rows
            else copy_token_rows(value, rows=self.padded_rows)
        )
        gathered = value.new_empty(
            (self.world_size * self.padded_rows, *value.shape[1:])
        )
        dist.all_gather_into_tensor(gathered, padded, group=self.group)
        # Identity positions make the packed rows a prefix, without internal padding.
        if self.identity_order:
            return gathered[: self.packed_positions.numel()]
        return gathered.index_select(0, self.packed_positions)

    def reduce_scatter(
        self, value: torch.Tensor, *, rank_ordered: bool = False
    ) -> torch.Tensor:
        if value.shape[0] != self.packed_positions.numel():
            raise ValueError("Token count differs from the reduce-scatter plan")
        if self.padded_rows == 0:
            return value
        direct = self.packed_positions.numel() == self.world_size * self.padded_rows
        direct = direct and (rank_ordered or self.identity_order)
        if direct:
            packed = value.contiguous()
        else:
            packed = copy_token_rows(value, self.reduce_indices)
        output = value.new_empty((self.padded_rows, *value.shape[1:]))
        dist.reduce_scatter_tensor(output, packed, group=self.group)
        return output[: self.local_rows]


def make_token_gather_scatter(*, indices, group, device) -> TokenGatherScatter:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if len(indices) != world:
        raise ValueError("Token layout must partition the packed request rows")
    flat = np.concatenate([np.asarray(rows, dtype=np.int64) for rows in indices])
    if not np.array_equal(np.sort(flat), np.arange(len(flat))):
        raise ValueError("Token layout must partition the packed request rows")
    maximum = max(map(len, indices), default=0)
    packed = np.concatenate(
        [
            np.arange(len(rows), dtype=np.int64) + peer * maximum
            for peer, rows in enumerate(indices)
        ]
    )
    positions = np.empty(len(flat), dtype=np.int64)
    positions[flat] = packed
    reduce_indices = np.full(world * maximum, -1, dtype=np.int64)
    reduce_indices[packed] = flat
    return TokenGatherScatter(
        packed_positions=_cp_metadata_tensor(
            positions, device=device, dtype=torch.long
        ),
        reduce_indices=_cp_metadata_tensor(
            reduce_indices, device=device, dtype=torch.long
        ),
        local_rows=len(indices[rank]),
        padded_rows=maximum,
        world_size=world,
        group=group,
        identity_order=np.array_equal(positions, np.arange(len(positions))),
    )


def gather_token_rows(x, indices, *, group):
    """Restore packed request order from a possibly uneven CP layout."""
    return make_token_gather_scatter(
        indices=indices, group=group, device=x.device
    ).gather(x)


@cache
def _get_copy_token_rows_kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["input_rows"])
    def _copy_token_rows_kernel(
        value,
        indices,
        output,
        input_rows,
        row_stride: tl.constexpr,
        features: tl.constexpr,
        indexed: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        source = tl.load(indices + row) if indexed else row
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = (source >= 0) & (source < input_rows)
        data = tl.load(
            value + source.to(tl.int64) * row_stride + offsets,
            mask=valid & (offsets < features),
            other=0.0,
        )
        tl.store(
            output + row.to(tl.int64) * features + offsets,
            data,
            mask=offsets < features,
        )

    return _copy_token_rows_kernel


@cache
def warmup_copy_token_rows(*, device, dtype, features, row_strides):
    """Compile and load layout variants without launching or touching buffers."""
    kernel = _get_copy_token_rows_kernel()
    with torch.cuda.device(device):
        for row_stride in sorted(set(row_strides)):
            for index_dtype in (None, torch.int32, torch.int64):
                compiled = kernel.warmup(
                    dtype,
                    index_dtype,
                    dtype,
                    1,
                    row_stride,
                    features,
                    index_dtype is not None,
                    BLOCK=1024,
                    num_warps=4,
                    grid=(1, (features + 1023) // 1024),
                )
                # Triton warmup compiles, but defers module and launcher loading
                # until this property is read. Do not call the returned launcher.
                _ = compiled.run


def copy_token_rows(
    value: torch.Tensor,
    indices: torch.Tensor | None = None,
    *,
    rows: int | None = None,
) -> torch.Tensor:
    """Gather rows and zero padding in one pass, preserving the input dtype.

    ``indices`` contains valid input row indices or -1 for padding. Without
    indices, copy a prefix and zero any extra output rows. Trailing dimensions
    are preserved; a larger row stride does not require materialization.
    """
    if value.ndim < 1:
        raise ValueError("Token values must have a row dimension")
    if indices is not None:
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token indices must be a one-dimensional integer tensor")
        if indices.device != value.device:
            raise ValueError("Token indices and values must be on the same device")
        indices = indices.contiguous()
        if rows is not None and rows != indices.numel():
            raise ValueError("Output rows differ from the index count")
        rows = indices.numel()
    elif rows is None:
        rows = value.shape[0]
    if rows < 0:
        raise ValueError("Output rows must be nonnegative")
    shape = (rows, *value.shape[1:])
    if not value.is_cuda:
        if indices is None:
            result = value.new_zeros(shape)
            count = min(rows, value.shape[0])
            result[:count] = value[:count]
            return result
        if not value.shape[0]:
            return value.new_zeros(shape)
        result = value.index_select(0, indices.clamp_min(0))
        return result.masked_fill((indices < 0).view(-1, *([1] * (value.ndim - 1))), 0)
    output = value.new_empty(shape)
    features = math.prod(value.shape[1:])
    if not rows or not features:
        return output
    # The kernel flattens features, so only feature strides require materialization.
    expected_stride = 1
    for size, stride in zip(reversed(value.shape[1:]), reversed(value.stride()[1:])):
        if size > 1 and stride != expected_stride:
            value = value.contiguous()
            break
        expected_stride *= size
    _get_copy_token_rows_kernel()[(rows, (features + 1023) // 1024)](
        value,
        indices,
        output,
        value.shape[0],
        value.stride(0),
        features,
        indices is not None,
        BLOCK=1024,
        num_warps=4,
    )
    return output


class RingKVExchange:
    """One workspace, serialized on its construction stream; fully consume shards()."""

    @classmethod
    def reserve(
        cls,
        *,
        group,
        context_len: int,
        max_requests: int,
        cp_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        device,
        dtype: torch.dtype,
        stream,
        transport="nccl",
    ):
        if max_requests is None or min(context_len, max_requests, cp_size) <= 0:
            raise ValueError("Ring workspace bounds must be positive")
        # Prefixes count once per request even when radix shares physical KV slots.
        capacity = max_requests * ((context_len + cp_size - 1) // cp_size)
        if dist.get_world_size(group) != cp_size:
            raise ValueError("Ring workspace CP size does not match its process group")
        signature = (capacity, q_heads, kv_heads, head_dim, str(dtype), transport)
        signatures = [None] * cp_size
        dist.all_gather_object(signatures, signature, group=group)
        if any(peer != signature for peer in signatures):
            raise ValueError("Ring workspace geometry must agree on every CP rank")
        with torch.cuda.stream(stream):
            exchange = cls(
                group=group,
                capacity=capacity,
                kv_heads=kv_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
                overlap=True,
                transport=transport,
            )
            # A scheduler with overlap disabled creates its run-batch stream after
            # backend initialization. Bind there once, with event ordering only.
            # Initialize padding for kernels that may read masked V lanes.
            exchange.buffers.zero_()
            exchange.defer_stream_binding()
        logger.info(
            "Ring CP workspace reserved before KV sizing: capacity=%d "
            "max_requests=%d context_len=%d cp_size=%d dtype=%s payload_bytes=%d",
            capacity,
            max_requests,
            context_len,
            cp_size,
            dtype,
            exchange.buffers.numel() * exchange.buffers.element_size(),
        )
        return exchange

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        capacity: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        overlap: bool,
        transport: str | None = None,
    ):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.stream = torch.cuda.current_stream(device)
        self.comm_stream = torch.cuda.Stream(device=device) if overlap else self.stream
        self.overlap = overlap
        self.initialization_ready = None
        shape = (2, 2, max(capacity, 1), kv_heads, head_dim)
        self.transport = transport or "nccl"
        self.rma = None
        if self.transport == "symm_rma":
            from sglang.srt.distributed.device_communicators.torch_symm_mem import (
                TorchSymmMemCommunicator,
            )

            self.symm_mem_comm = TorchSymmMemCommunicator.get(group, device)
            self.rma = self.symm_mem_comm.allocate_ring_workspace(shape, dtype)
            self.buffers = self.rma.buffers
        elif self.transport == "nccl":
            self.buffers = torch.empty(shape, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown ring CP transport: {self.transport}")
        self.capacity = shape[2]
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.next_rank = dist.get_global_rank(group, (self.rank + 1) % self.world)
        self.prev_rank = dist.get_global_rank(group, (self.rank - 1) % self.world)

    def defer_stream_binding(self):
        """Publish startup writes before the first use on the run-batch stream."""
        self.initialization_ready = self.stream.record_event()

    def _rotate(
        self,
        *,
        current: int,
        ready: torch.cuda.Event,
        reusable: torch.cuda.Event | None,
        active_tokens: int,
    ):
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ready)
            if reusable is not None:
                self.comm_stream.wait_event(reusable)
            if self.rma is not None:
                # K/V planes retain capacity-sized strides even for short frames.
                token_bytes = (
                    self.kv_heads * self.head_dim * self.buffers.element_size()
                )
                plane_bytes = active_tokens * token_bytes
                self.symm_mem_comm.exchange_ring(
                    self.rma,
                    send_slot=current,
                    recv_slot=1 - current,
                    signal_slot=1 - current,
                    byte_ranges=(
                        (0, plane_bytes),
                        (self.capacity * token_bytes, plane_bytes),
                    ),
                )
            else:
                self._rotate_nccl(current, active_tokens)
            received = self.comm_stream.record_event()
        return received

    def _rotate_nccl(self, current, active_tokens):
        # P2P only copies bytes. NCCL/PyTorch versions differ in FP8 datatype
        # support; a byte view preserves quantized payloads without conversion.
        buffers = self.buffers.view(torch.uint8)
        if not active_tokens:
            return
        # Each plane prefix is contiguous; the K/V pair has a capacity-sized
        # gap and cannot be sent as one shortened contiguous tensor.
        operations = []
        for plane in range(2):
            operations.extend(
                [
                    dist.P2POp(
                        dist.isend,
                        buffers[current, plane, :active_tokens],
                        self.next_rank,
                        self.group,
                    ),
                    dist.P2POp(
                        dist.irecv,
                        buffers[1 - current, plane, :active_tokens],
                        self.prev_rank,
                        self.group,
                    ),
                ]
            )
        works = dist.batch_isend_irecv(operations)
        # Work.wait orders this CUDA stream; it does not synchronize the host.
        for work in works:
            work.wait()

    def shards(
        self, local_kv: torch.Tensor, *, active_tokens: int | None = None
    ) -> Iterator[tuple[int, torch.Tensor]]:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Ring exchange does not support CUDA graph capture")
        if self.initialization_ready is not None:
            self.stream = torch.cuda.current_stream(local_kv.device)
            self.stream.wait_event(self.initialization_ready)
            if not self.overlap:
                self.comm_stream = self.stream
            self.initialization_ready = None
        if torch.cuda.current_stream(local_kv.device) != self.stream:
            raise ValueError("Ring workspace must be used on its construction stream")
        active_tokens = self.capacity if active_tokens is None else active_tokens
        if (
            not 0 <= local_kv.shape[1] <= active_tokens <= self.capacity
            or local_kv.shape != (2, local_kv.shape[1], self.kv_heads, self.head_dim)
            or local_kv.dtype != self.buffers.dtype
            or local_kv.device != self.buffers.device
        ):
            raise ValueError(
                "KV payload exceeds the active ring frame or its workspace"
            )
        self.buffers[0][:, : local_kv.shape[1]].copy_(local_kv)
        ready = self.stream.record_event()
        reusable = [None, None]
        for step in range(self.world):
            current, owner = step % 2, (self.rank - step) % self.world
            self.stream.wait_event(ready)
            received = None
            if step + 1 < self.world:
                received = self._rotate(
                    current=current,
                    ready=ready,
                    reusable=reusable[1 - current],
                    active_tokens=active_tokens,
                )
            yield owner, self.buffers[current, :, :active_tokens]
            reusable[current] = self.stream.record_event()
            if received is not None:
                ready = received


class PendingRingTensor:
    """Received tails remain valid until the workspace's next prepare."""

    def __init__(self, exchange, ready):
        self.exchange = exchange
        self.ready = ready
        self.received = None
        self.consumed = False

    def start(self):
        self.exchange.check_stream()
        if self.received is not None:
            raise RuntimeError("Ring tensor transfer was already started")
        self.received = self.exchange.start(self.ready)

    def wait(self):
        exchange = self.exchange
        exchange.check_stream()
        if self.received is None or self.consumed:
            raise RuntimeError("Ring tensor must be started and consumed exactly once")
        exchange.stream.wait_event(self.received)
        exchange.pending = False
        self.consumed = True
        return exchange.tails


class RingTensorExchange:
    """Persistent small-tensor workspace using the KV ring's RMA protocol.

    The communication stream forwards all peer tails without waiting for model
    compute. Gathering every tail also preserves replicated terminal conv cache,
    including empty ranks and shards shorter than the convolution history.
    Prepare, consume and reuse must run on the construction stream.
    """

    def __init__(self, *, group, shape, device, dtype, transport=None):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.stream = torch.cuda.current_stream(device)
        self.comm_stream = torch.cuda.Stream(device=device)
        self.transport = transport or "nccl"
        self.rma = None
        if self.transport == "symm_rma":
            from sglang.srt.distributed.device_communicators.torch_symm_mem import (
                TorchSymmMemCommunicator,
            )

            # Each source owns one persistent output slot. Forward directly
            # from received slots; do not copy through a second gather buffer.
            self.symm_mem_comm = TorchSymmMemCommunicator.get(group, device)
            self.rma = self.symm_mem_comm.allocate_ring_workspace(
                (self.world, *shape), dtype
            )
            self.buffers = self.rma.buffers
            self.tails = self.buffers
        elif self.transport == "nccl":
            with torch.inference_mode(False), torch.no_grad():
                self.buffers = torch.empty((2, *shape), device=device, dtype=dtype)
                self.tails = torch.empty(
                    (self.world, *shape), device=device, dtype=dtype
                )
        else:
            raise ValueError(f"Unknown ring CP transport: {self.transport}")
        self.next_rank = dist.get_global_rank(group, (self.rank + 1) % self.world)
        self.prev_rank = dist.get_global_rank(group, (self.rank - 1) % self.world)
        self.pending = False

    def check_stream(self):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Ring tensor exchange does not support CUDA graph capture")
        if torch.cuda.current_stream(self.buffers.device) != self.stream:
            raise ValueError("Ring tensor workspace must use its construction stream")

    def prepare(self, local_tail):
        self.check_stream()
        if self.pending:
            raise RuntimeError("Previous ring tensor must be consumed before reuse")
        if (
            local_tail.shape != self.buffers.shape[1:]
            or local_tail.dtype != self.buffers.dtype
            or local_tail.device != self.buffers.device
        ):
            raise ValueError("Tensor does not match the ring workspace")
        self.buffers[self.rank if self.rma is not None else 0].copy_(local_tail)
        ready = self.stream.record_event()
        self.pending = True
        return PendingRingTensor(self, ready)

    def start(self, ready):
        # The caller enqueues interior compute before starting communication,
        # but this event precedes that compute. Waiting on the current stream
        # here would serialize communication behind the work meant to hide it.
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ready)
            if self.rma is None:
                self.tails[self.rank].copy_(self.buffers[0])
            for step in range(1, self.world):
                current = (step - 1) % 2
                if self.rma is not None:
                    owner = (self.rank - step + 1) % self.world
                    self.symm_mem_comm.exchange_ring(
                        self.rma,
                        send_slot=owner,
                        recv_slot=owner,
                        signal_slot=1 - current,
                    )
                else:
                    works = dist.batch_isend_irecv(
                        [
                            dist.P2POp(
                                dist.isend,
                                self.buffers[current],
                                self.next_rank,
                                self.group,
                            ),
                            dist.P2POp(
                                dist.irecv,
                                self.buffers[1 - current],
                                self.prev_rank,
                                self.group,
                            ),
                        ]
                    )
                    for work in works:
                        work.wait()
                    owner = (self.rank - step) % self.world
                    self.tails[owner].copy_(self.buffers[1 - current])
            received = self.comm_stream.record_event()
        return received


def map_src_to_dst_token_slots(
    src_pages,
    dst_pages,
    *,
    page_size,
    cp_size=1,
    cp_rank=0,
    dcp_size=1,
    dcp_rank=0,
    src_page_offset=0,
    decode_prefix_len=0,
    num_tokens=None,
    replicated=False,
):
    """Return paired source and destination physical token slots for KV copying.

    The result is (src_token_slots, dst_token_slots): element i of the first
    array is copied to element i of the second. This computes a mapping only.
    CP/DCP ownership and replicated NEXTN storage are resolved from page metadata.

    Source pages describe this chunk. Destination pages describe the entire
    uncached suffix. Main DCP pages hold page_size local rows; draft pages hold
    page_size * dcp_size rows. Both use the same page IDs.
    """
    src = np.asarray(src_pages, dtype=np.int64)
    dst = np.asarray(dst_pages, dtype=np.int64)
    if (
        src.ndim != 1
        or dst.ndim != 1
        or np.any(src < 0)
        or np.any(dst < 0)
        or page_size <= 0
        or cp_size <= 0
        or page_size % cp_size
        or not 0 <= cp_rank < cp_size
        or dcp_size <= 0
        or not 0 <= dcp_rank < dcp_size
        or src_page_offset < 0
        or decode_prefix_len < 0
    ):
        raise ValueError("Invalid CP/DCP token geometry")
    virtual_page = page_size * dcp_size
    if decode_prefix_len % virtual_page:
        raise ValueError("Decode prefix must align to the virtual DCP page")
    if num_tokens is None:
        num_tokens = src.size * page_size
    suffix_start = src_page_offset * page_size
    if (
        num_tokens < 0
        or num_tokens > src.size * page_size
        or suffix_start + num_tokens > dst.size * virtual_page
    ):
        raise ValueError("KV token range exceeds source or destination pages")
    # CP ownership is determined by the physical page row, not the zigzag
    # computation order. A page-aligned chunk preserves that ownership.
    offsets = np.arange(cp_rank, num_tokens, cp_size, dtype=np.int64)
    absolute = decode_prefix_len + suffix_start + offsets
    if not replicated:
        offsets = offsets[absolute % dcp_size == dcp_rank]
    local_page = page_size // cp_size
    src_token_slots = (
        src[offsets // page_size] * local_page + offsets % page_size // cp_size
    )
    local = suffix_start + offsets
    dst_page_size = virtual_page if replicated else page_size
    if not replicated:
        local = local // dcp_size
    dst_token_slots = (
        dst[local // dst_page_size] * dst_page_size + local % dst_page_size
    )
    return src_token_slots, dst_token_slots


def compute_kv_head_overlap(src_tp, dst_tp, src_rank, dst_rank, heads):
    """Return (src_head_start, num_heads, dst_head_start) for connected TP ranks."""
    if (
        min(src_tp, dst_tp, heads) < 1
        or src_rank < 0
        or dst_rank < 0
        or max(src_tp, dst_tp) % min(src_tp, dst_tp)
        or any(heads % tp and tp % heads for tp in (src_tp, dst_tp))
    ):
        raise ValueError("Invalid attention TP/head geometry")
    src_rank %= src_tp
    dst_rank %= dst_tp
    src_heads, dst_heads = max(1, heads // src_tp), max(1, heads // dst_tp)
    src_begin = src_rank // max(1, src_tp // heads) * src_heads
    dst_begin = dst_rank // max(1, dst_tp // heads) * dst_heads
    begin, end = (
        max(src_begin, dst_begin),
        min(src_begin + src_heads, dst_begin + dst_heads),
    )
    if begin >= end:
        raise ValueError("Connected TP ranks have no KV heads in common")
    return begin - src_begin, end - begin, begin - dst_begin


class KVRearrangeMetadata(NamedTuple):
    """Paired byte ranges describing a source-to-destination KV rearrangement.

    Offsets are relative to each allocation, so the same mapping can be reused
    for every K/V layer with the same geometry. Construct this metadata before
    issuing writes; execution only binds the source and destination pointers.
    """

    src_byte_offsets: np.ndarray
    dst_byte_offsets: np.ndarray
    byte_counts: np.ndarray

    @classmethod
    def from_byte_ranges(cls, src_byte_offsets, dst_byte_offsets, byte_counts):
        src, dst, sizes = (
            np.asarray(x, dtype=np.int64)
            for x in (src_byte_offsets, dst_byte_offsets, byte_counts)
        )
        if src.ndim != 1 or src.shape != dst.shape or src.shape != sizes.shape:
            raise ValueError(
                "KV rearrangement requires paired source/destination byte ranges"
            )
        if np.any(src < 0) or np.any(dst < 0) or np.any(sizes < 0):
            raise ValueError("KV rearrangement byte ranges must be nonnegative")
        active = sizes > 0
        src, dst, sizes = src[active], dst[active], sizes[active]
        if not src.size:
            return cls(src, dst, sizes)
        # Merge only when both source and destination ranges are contiguous.
        cuts = (
            np.flatnonzero((np.diff(src) != sizes[:-1]) | (np.diff(dst) != sizes[:-1]))
            + 1
        )
        starts = np.r_[0, cuts]
        return cls(src[starts], dst[starts], np.add.reduceat(sizes, starts))

    @classmethod
    def from_token_slots(
        cls,
        src_token_slots,
        dst_token_slots,
        *,
        src_token_bytes,
        dst_token_bytes,
        src_head_byte_offset=0,
        dst_head_byte_offset=0,
        copy_bytes=None,
    ):
        """Resolve paired token slots and a head slice into byte-range metadata."""
        if copy_bytes is None:
            copy_bytes = src_token_bytes
        if (
            src_token_bytes <= 0
            or dst_token_bytes <= 0
            or copy_bytes <= 0
            or min(src_head_byte_offset, dst_head_byte_offset) < 0
            or src_head_byte_offset + copy_bytes > src_token_bytes
            or dst_head_byte_offset + copy_bytes > dst_token_bytes
        ):
            raise ValueError("KV head slice exceeds a token slot")
        src_token_slots, dst_token_slots = (
            np.asarray(src_token_slots),
            np.asarray(dst_token_slots),
        )
        if src_token_slots.shape != dst_token_slots.shape:
            raise ValueError("Source and destination token slot counts differ")
        return cls.from_byte_ranges(
            src_token_slots * src_token_bytes + src_head_byte_offset,
            dst_token_slots * dst_token_bytes + dst_head_byte_offset,
            np.full(src_token_slots.shape, copy_bytes, dtype=np.int64),
        )

    @classmethod
    def from_kv_entries(cls, entry_pairs, src_entry_offsets, dst_entry_offsets):
        """Rearrange packed staging entries while leaving receiver-only NEXTN KV untouched."""
        src, dst, sizes = [], [], []
        for src_entry, dst_entry in entry_pairs:
            size = src_entry_offsets[src_entry + 1] - src_entry_offsets[src_entry]
            if size != dst_entry_offsets[dst_entry + 1] - dst_entry_offsets[dst_entry]:
                raise ValueError("Source and destination KV entry byte counts differ")
            src.append(src_entry_offsets[src_entry])
            dst.append(dst_entry_offsets[dst_entry])
            sizes.append(size)
        return cls.from_byte_ranges(src, dst, sizes)

    def build_transfer_blocks(self, src_ptr, dst_ptr):
        """Bind allocation pointers to validated (src address, dst address, bytes)."""
        return list(
            zip(
                (src_ptr + self.src_byte_offsets).tolist(),
                (dst_ptr + self.dst_byte_offsets).tolist(),
                self.byte_counts.tolist(),
            )
        )


def build_cp_kv_rearrange_metadata(
    *,
    src_pages,
    dst_pages,
    page_size,
    cp_size,
    cp_rank,
    src_page_bytes,
    dst_page_bytes,
    total_kv_heads=None,
    dst_tp_size=1,
    dst_tp_rank=0,
    dcp_size=1,
    dcp_rank=0,
    src_page_offset=0,
    decode_prefix_len=0,
    num_kv_tokens=None,
    replicated=False,
):
    """Build reusable rearrangement metadata for CP -> TP/DCP KV transfer."""
    src_token_slots, dst_token_slots = map_src_to_dst_token_slots(
        src_pages,
        dst_pages,
        page_size=page_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        src_page_offset=src_page_offset,
        decode_prefix_len=decode_prefix_len,
        num_tokens=num_kv_tokens,
        replicated=replicated,
    )
    local_page = page_size // cp_size
    if src_page_bytes <= 0 or src_page_bytes % local_page:
        raise ValueError("Invalid CP page byte stride")
    src_token_bytes = src_page_bytes // local_page
    offset, copied = 0, src_token_bytes
    if total_kv_heads is not None:
        start, heads, _ = compute_kv_head_overlap(
            1, dst_tp_size, 0, dst_tp_rank, total_kv_heads
        )
        if src_token_bytes % total_kv_heads:
            raise ValueError("Invalid KV head byte stride")
        head_bytes = src_token_bytes // total_kv_heads
        offset, copied = start * head_bytes, heads * head_bytes
    dst_tokens_per_page = page_size * (dcp_size if replicated else 1)
    if dst_page_bytes != dst_tokens_per_page * copied:
        raise ValueError("CP peers have incompatible per-entry KV geometry")
    return KVRearrangeMetadata.from_token_slots(
        src_token_slots,
        dst_token_slots,
        src_token_bytes=src_token_bytes,
        dst_token_bytes=copied,
        src_head_byte_offset=offset,
        copy_bytes=copied,
    )


@torch.inference_mode(False)
@torch.no_grad()
def select_cp_tactic(candidates, run, group):
    """Time identical iteration counts on every rank and minimize the slowest rank."""
    timings = []
    for tactic in candidates:
        for _ in range(3):
            run(tactic)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run(tactic)
        end.record()
        end.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end) / 20, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        timings.append(elapsed.item())
    best = min(range(len(timings)), key=timings.__getitem__)
    logger.info(
        "CP operator autotune: candidates=%s ms=%s winner=%s",
        candidates,
        timings,
        candidates[best],
    )
    return candidates[best]


def _heads_to_context(x: torch.Tensor, *, group) -> torch.Tensor:
    """[T, H, D] per TP rank -> [T / P, P * H, D] per CP rank."""
    world = dist.get_world_size(group)
    received = torch.empty_like(x)
    dist.all_to_all_single(received, x.contiguous(), group=group)
    return (
        received.view(world, x.shape[0] // world, *x.shape[1:])
        .transpose(0, 1)
        .flatten(1, 2)
        .contiguous()
    )


def _context_to_heads(x: torch.Tensor, *, group) -> torch.Tensor:
    """Inverse of _heads_to_context, including the original token ordering."""
    world = dist.get_world_size(group)
    send = (
        x.view(x.shape[0], world, x.shape[1] // world, x.shape[2])
        .transpose(0, 1)
        .contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return received.flatten(0, 1)


class ContextParallelOutput(msgspec.Struct, frozen=True):
    """Contiguous token shard with all heads (or projected hidden features)."""

    hidden_states: torch.Tensor


def _cp_to_token_shard(chunks, *, lengths, total_tokens, group):
    """Repartition per-request CP chunks into the communicator's packed token shard.

    CPU sequence lengths define all peer counts and receive offsets identically
    on every rank. Physical padding stays zero and never becomes a routed token.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    shard_size = total_tokens // world
    if (
        len(lengths) == 1
        and lengths[0] == total_tokens
        and total_tokens % (64 * world) == 0
    ):
        return chunks[0]

    def segments(source, destination):
        start = 0
        for index, length in enumerate(lengths):
            cp_size = (length + 64 * world - 1) // (64 * world) * 64
            cp_start = start + source * cp_size
            lo = max(cp_start, destination * shard_size)
            hi = min(start + length, cp_start + cp_size, (destination + 1) * shard_size)
            if hi > lo:
                yield index, lo - cp_start, lo - destination * shard_size, hi - lo
            start += length

    send_parts, send_sizes, recv_sizes, destinations = [], [], [], []
    for peer in range(world):
        outgoing = list(segments(rank, peer))
        incoming = list(segments(peer, rank))
        send_sizes.append(sum(n for _, _, _, n in outgoing))
        recv_sizes.append(sum(n for _, _, _, n in incoming))
        send_parts.extend(
            chunks[i][offset : offset + n] for i, offset, _, n in outgoing
        )
        destinations.extend(range(offset, offset + n) for _, _, offset, n in incoming)
    template = chunks[0]
    send = torch.cat(send_parts) if send_parts else template[:0]
    received = template.new_empty((sum(recv_sizes), *template.shape[1:]))
    dist.all_to_all_single(received, send, recv_sizes, send_sizes, group=group)
    result = template.new_zeros((shard_size, *template.shape[1:]))
    indices = torch.tensor(
        [i for part in destinations for i in part], device=result.device
    )
    result.index_copy_(0, indices.long(), received)
    return result
