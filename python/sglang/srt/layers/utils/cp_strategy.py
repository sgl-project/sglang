# Copyright 2023-2024 SGLang Team
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

"""Context Parallel Strategy abstraction.

Two concrete strategies are provided:

* ``ZigzagCPStrategy`` (formerly ``--prefill-cp-mode in-seq-split`` /
  ``--dsa-prefill-cp-mode in-seq-split``). Splits the prefill sequence into
  ``2 * cp_size`` blocks and assigns block-pair ``(r, 2*cp_size - 1 - r)`` to
  rank ``r``. Causal-balanced; single batch only.

* ``InterleaveCPStrategy`` (formerly ``--*-prefill-cp-mode round-robin-split``).
  Assigns token ``i`` to rank ``i % cp_size``. Multi-batch capable.

The strategy is a process-wide singleton selected at startup from
``ServerArgs.enable_prefill_cp`` + ``ServerArgs.cp_strategy``. Consumers obtain
it through ``get_cp_strategy()`` and gate their CP-specific code on
``cp_active(forward_batch)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


# ---------------------------------------------------------------------------
# Public payload type
# ---------------------------------------------------------------------------


@dataclass
class ContextParallelMetadata:
    """Per-forward CP payload attached to ``forward_batch.attn_cp_metadata``.

    Field names are kept stable so that hardware-backend mirrors (NPU's
    ``ascend_backend.py``, MUSA's ``flashattention_backend.py``) which read
    these directly do not need to change.

    For interleave strategy, only the descriptor fields (cp_size/cp_rank,
    bookkeeping for shard size) are needed; the rich split/zigzag fields stay
    at their dataclass defaults.
    """

    # ---- common ----
    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None

    # ---- attention (zigzag two-half) ----
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None

    total_seq_lens: torch.Tensor = None


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class ContextParallelStrategy(ABC):
    """Owns all per-mode policy: when CP applies, how tokens are sliced and
    reassembled, how attention is dispatched, and how K/V is materialised.

    A single instance is constructed at server start (``init_cp_strategy``).
    Stateless w.r.t. requests — all per-forward state goes on
    ``forward_batch.attn_cp_metadata`` via :py:meth:`build_metadata`.
    """

    #: Strategy short name, used in logs and as the CLI value.
    name: str

    #: True when the strategy expects each layer to run with the model body in
    #: SCATTERED layout (i.e. per-layer attn_cp allgather/scatter), as opposed
    #: to TP_ATTN_FULL with CP communication folded into the MoE-DP group.
    per_layer_attn_cp_comm: bool

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        # cp_rank is resolved lazily on first use — the cp group is built only
        # after distributed init, which happens after server_args validation.

    # ---- group / rank accessors (resolved lazily) ----

    @property
    def cp_rank(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_cp_rank

        return get_attention_cp_rank()

    # ---- preconditions ----

    @abstractmethod
    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        """Return True if this strategy can shard ``num_tokens`` on the given
        forward batch. Used by model entry points before
        ``build_metadata`` to decide whether to attach metadata at all."""

    # ---- metadata construction ----

    @abstractmethod
    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
    ) -> ContextParallelMetadata:
        """Produce the ``ContextParallelMetadata`` payload for one forward."""

    # ---- token sharding ----

    @abstractmethod
    def shard_tokens(
        self,
        x: Union[torch.Tensor, List, Tuple],
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Slice ``[L, ...]`` (along dim 0) down to this rank's per-rank shape.
        Used for hidden_states, positions, and input_ids."""

    @abstractmethod
    def shard_positions(
        self, positions: torch.Tensor, forward_batch: "ForwardBatch"
    ) -> torch.Tensor:
        """Slice positions along the last dim (positions can be 1-D or 2-D)."""

    # ---- token gather (inverse of shard) ----

    @abstractmethod
    def gather_tokens(
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Allgather ``[L/S, ...]`` per-rank tensors into ``[L, ...]`` in
        original token order."""

    @abstractmethod
    def gather_kv_cache(
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Gather variant for multi-dim KV cache tensors (e.g.
        ``[seq_len, num_heads, head_dim]``)."""

    # ---- per-request sequence-length sharding (interleave only) ----

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: torch.Tensor,
    ) -> Tuple[List[int], torch.Tensor, List[int], torch.Tensor]:
        """For interleave: split per-request lengths across CP ranks.
        Default: not supported."""
        raise NotImplementedError(
            f"{self.name} strategy does not support per-request sharding"
        )

    # ---- attention dispatch ----

    @abstractmethod
    def iter_attn_slices(
        self,
        q: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> List["CPAttnSlice"]:
        """Slice ``q`` along the token dim into the per-rank chunks the
        attention kernel should process. For zigzag, returns 2 chunks
        ``(q_prev, q_next)``. For interleave, returns a single chunk."""

    def run_attention(
        self,
        q: torch.Tensor,
        forward_batch: "ForwardBatch",
        device: torch.device,
        attn_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor
        ],
    ) -> torch.Tensor:
        """Default implementation: call ``attn_fn`` once per slice and concat.

        ``attn_fn`` signature: ``(q_chunk, cu_seqlens_q, cache_seqlens,
        max_seqlen_q) -> result``.
        """
        results = []
        for slc in self.iter_attn_slices(q, forward_batch):
            cu_seqlens_q = torch.tensor(
                [0, slc.actual_seq_q], device=device, dtype=torch.int32
            )
            results.append(
                attn_fn(
                    slc.q,
                    cu_seqlens_q,
                    slc.cache_seqlens_tensor,
                    slc.actual_seq_q,
                )
            )
        if len(results) == 1:
            return results[0]
        return torch.cat(results, dim=0)

    # ---- KV cache materialisation ----

    @abstractmethod
    def materialize_full_kv(
        self,
        forward_batch: "ForwardBatch",
        layer,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write the *full sequence* K/V into every rank's pool. Zigzag does
        an allgather+reorder then ``set_kv_buffer``; interleave's variant is
        currently unused (the dsv4 backend stores into the cache after gather
        in attention prep)."""

    # ---- attention backend metadata reindex ----

    def reindex_attn_metadata(self, core_attn_metadata) -> None:
        """Optional. Strided modes (interleave) re-slice page-tables /
        FlashMLA tensors so the backend sees per-rank metadata. Zigzag is a
        no-op."""
        return None

    # ---- layer-communicator integration ----

    def maybe_gather_for_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Called inside the layer communicator before MLP/MoE. By default no
        extra communication is performed (the MoE-DP group already handles it
        via ``MOE_FULL``). DSA-style strategies override to allgather across
        the attn_cp group."""
        return hidden_states

    def maybe_scatter_after_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Called inside the layer communicator after MLP/MoE. Inverse of
        :py:meth:`maybe_gather_for_mlp`."""
        return hidden_states


@dataclass
class CPAttnSlice:
    """One Q-chunk produced by :py:meth:`ContextParallelStrategy.iter_attn_slices`.

    All fields are populated for both strategies (``cache_seqlens_tensor`` is
    just ``forward_batch.seq_lens`` in the interleave case, with the strided
    page-table reindex carrying the actual CP shaping)."""

    q: torch.Tensor
    actual_seq_q: int
    cache_seqlens_tensor: torch.Tensor
    actual_seq_q_tensor: torch.Tensor


# ---------------------------------------------------------------------------
# Concrete: zigzag (in-seq-split)
# ---------------------------------------------------------------------------


class ZigzagCPStrategy(ContextParallelStrategy):
    """In-seq-split. Sequence is cut into ``2 * cp_size`` blocks; rank ``r``
    gets blocks ``r`` and ``2 * cp_size - 1 - r``. Each rank runs attention
    twice — once for each half it owns — to keep causal cost balanced."""

    name = "zigzag"
    per_layer_attn_cp_comm = False  # set by subclass below when DSA

    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        cur_cp_seq_len = num_tokens // (self.cp_size * 2)
        return (
            cur_cp_seq_len != 0
            and self.cp_size > 1
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.seq_lens_cpu is not None
            and forward_batch.seq_lens_cpu.shape[0] == 1
        )

    def build_metadata(
        self, num_tokens: int, seqs_len: Optional[List[int]]
    ) -> ContextParallelMetadata:
        return _build_zigzag_metadata(
            kv_len=num_tokens,
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
            seqs_len=seqs_len,
            bake_prefix=not _is_dsa_active(),
        )

    def shard_tokens(self, x, forward_batch):
        cp_meta = forward_batch.attn_cp_metadata
        x_list = list(torch.split(x, cp_meta.split_list, dim=0))
        return torch.cat([x_list[i] for i in cp_meta.zigzag_index], dim=0).view(
            -1, x.shape[-1]
        )

    def shard_positions(self, positions, forward_batch):
        cp_meta = forward_batch.attn_cp_metadata
        pos_list = list(torch.split(positions, cp_meta.split_list, dim=-1))
        return torch.cat([pos_list[i] for i in cp_meta.zigzag_index], dim=-1)

    def gather_tokens(self, x, forward_batch, stream=None):
        from sglang.srt.layers.utils.cp_collectives import (
            cp_all_gather_reorganized_into_tensor,
        )

        bs_seq_len, hidden_size = x.shape
        s = stream if stream is not None else torch.cuda.current_stream()
        full = cp_all_gather_reorganized_into_tensor(
            x,
            forward_batch.attn_cp_metadata.total_seq_lens,
            self.cp_size,
            forward_batch,
            s,
        )
        return self._undo_zigzag(full, forward_batch, hidden_size)

    def gather_kv_cache(self, x, forward_batch, stream=None):
        from sglang.srt.layers.utils.cp_collectives import (
            cp_all_gather_reorganized_into_tensor_kv_cache,
        )

        s = stream if stream is not None else torch.cuda.current_stream()
        full = cp_all_gather_reorganized_into_tensor_kv_cache(
            x,
            forward_batch.attn_cp_metadata.total_seq_lens,
            self.cp_size,
            forward_batch,
            s,
        )
        return self._undo_zigzag(full, forward_batch, hidden_size=None)

    def _undo_zigzag(self, full, forward_batch, hidden_size):
        cp_meta = forward_batch.attn_cp_metadata
        pieces = list(torch.split(full, cp_meta.reverse_split_len, dim=0))
        out = torch.cat([pieces[i] for i in cp_meta.cp_reverse_index], dim=0)
        if hidden_size is not None:
            out = out.view(-1, hidden_size)
        return out

    def iter_attn_slices(self, q, forward_batch):
        cp_meta = forward_batch.attn_cp_metadata
        q_prev, q_next = torch.chunk(q, 2, dim=0)
        return [
            CPAttnSlice(
                q=q_prev,
                actual_seq_q=cp_meta.actual_seq_q_prev,
                cache_seqlens_tensor=cp_meta.kv_len_prev_tensor,
                actual_seq_q_tensor=cp_meta.actual_seq_q_prev_tensor,
            ),
            CPAttnSlice(
                q=q_next,
                actual_seq_q=cp_meta.actual_seq_q_next,
                cache_seqlens_tensor=cp_meta.kv_len_next_tensor,
                actual_seq_q_tensor=cp_meta.actual_seq_q_next_tensor,
            ),
        ]

    def materialize_full_kv(self, forward_batch, layer, k, v):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        k = k.contiguous()
        v = v.contiguous()
        key_cache_full = self.gather_kv_cache(k, forward_batch)
        value_cache_full = self.gather_kv_cache(v, forward_batch)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            cache_loc,
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )


# ---------------------------------------------------------------------------
# Concrete: interleave (round-robin-split)
# ---------------------------------------------------------------------------


class InterleaveCPStrategy(ContextParallelStrategy):
    """Round-robin-split. Token ``i`` goes to rank ``i % cp_size``. Each rank
    runs attention once on its strided slice. Supports multi-batch prefill,
    fused MoE, and FP8 KV cache."""

    name = "interleave"
    per_layer_attn_cp_comm = True  # DSA-only today; layout is SCATTERED

    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        cur = num_tokens // self.cp_size
        if num_tokens % self.cp_size != 0:
            # The DSA backend requires this divisibility (see can_dsa_cp_split).
            return False
        return (
            cur != 0
            and self.cp_size > 1
            and forward_batch.forward_mode.is_context_parallel_extend()
            and sum(forward_batch.extend_seq_lens_cpu) >= self.cp_size
        )

    def build_metadata(
        self, num_tokens: int, seqs_len: Optional[List[int]]
    ) -> ContextParallelMetadata:
        # Strided sharding is a pure index map — no payload required.
        return ContextParallelMetadata()

    def shard_tokens(self, x, forward_batch):
        return _strided_take(x, self.cp_rank, self.cp_size)

    def shard_positions(self, positions, forward_batch):
        return _strided_take(positions, self.cp_rank, self.cp_size)

    def gather_tokens(self, x, forward_batch, stream=None):
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import (
            attn_cp_all_gather_into_tensor,
            get_attention_cp_group,
            is_allocation_symmetric,
        )

        with use_symmetric_memory(
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            out = x.new_empty((x.shape[0] * self.cp_size, *x.shape[1:]))
        attn_cp_all_gather_into_tensor(out, x)
        out_shape = out.shape
        out = (
            out.view(self.cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return out

    def gather_kv_cache(self, x, forward_batch, stream=None):
        return self.gather_tokens(x, forward_batch, stream)

    def shard_per_request(self, extend_seqs_cpu, extend_seqs):
        return _interleave_split_q_seqs(
            extend_seqs_cpu, extend_seqs, self.cp_size, self.cp_rank
        )

    def iter_attn_slices(self, q, forward_batch):
        # Interleave runs a single attention pass on the strided slice; the
        # backend rebuilds page-tables and cu_seqlens to match.
        seq_lens = forward_batch.seq_lens
        actual_seq_q = q.shape[0]
        actual_seq_q_tensor = torch.tensor(
            [actual_seq_q], device=q.device, dtype=torch.int32
        )
        return [
            CPAttnSlice(
                q=q,
                actual_seq_q=actual_seq_q,
                cache_seqlens_tensor=seq_lens,
                actual_seq_q_tensor=actual_seq_q_tensor,
            )
        ]

    def materialize_full_kv(self, forward_batch, layer, k, v):
        # Interleave with FA backend follows the same "allgather then write"
        # contract as zigzag. The dsv4 backend uses a different fast path
        # (gather before store_cache) and won't call this method.
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        k = k.contiguous()
        v = v.contiguous()
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            cache_loc,
            self.gather_kv_cache(k, forward_batch),
            self.gather_kv_cache(v, forward_batch),
            layer.k_scale,
            layer.v_scale,
        )

    def reindex_attn_metadata(self, core_attn_metadata) -> None:
        if hasattr(core_attn_metadata, "apply_cp_reindex"):
            core_attn_metadata.apply_cp_reindex()
        if hasattr(core_attn_metadata, "init_flashmla_related"):
            core_attn_metadata.init_flashmla_related()

    # ---- DSA-style per-layer communication ----

    def maybe_gather_for_mlp(self, hidden_states, forward_batch):
        if not _cp_active_for_per_layer(forward_batch):
            return hidden_states
        from sglang.srt.layers.dp_attention import (
            attn_cp_all_gather_into_tensor,
            get_attention_cp_group,
            get_local_dp_buffer,
        )

        full, local = (
            get_local_dp_buffer(get_attention_cp_group()),
            hidden_states,
        )
        attn_cp_all_gather_into_tensor(full, local)
        return full

    def maybe_scatter_after_mlp(self, hidden_states, forward_batch):
        if not _cp_active_for_per_layer(forward_batch):
            return hidden_states
        from sglang.srt.layers.dp_attention import (
            attn_cp_reduce_scatter_tensor,
        )

        local = hidden_states.tensor_split(self.cp_size)[self.cp_rank]
        attn_cp_reduce_scatter_tensor(local, hidden_states)
        return local


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_STRATEGY: Optional[ContextParallelStrategy] = None


def init_cp_strategy(server_args: "ServerArgs") -> None:
    """Bind the active strategy for this process. Called from
    ``ServerArgs.__post_init__`` after CP validation has fixed the values."""
    global _STRATEGY
    if not getattr(server_args, "enable_prefill_cp", False):
        _STRATEGY = None
        return
    if server_args.attn_cp_size <= 1:
        _STRATEGY = None
        return
    if server_args.cp_strategy == "zigzag":
        _STRATEGY = ZigzagCPStrategy(cp_size=server_args.attn_cp_size)
    elif server_args.cp_strategy == "interleave":
        _STRATEGY = InterleaveCPStrategy(cp_size=server_args.attn_cp_size)
    else:
        raise ValueError(
            f"Unknown cp_strategy={server_args.cp_strategy!r}; "
            "expected one of {'zigzag', 'interleave'}"
        )


def get_cp_strategy() -> Optional[ContextParallelStrategy]:
    return _STRATEGY


def is_cp_enabled() -> bool:
    return _STRATEGY is not None


def is_zigzag() -> bool:
    return isinstance(_STRATEGY, ZigzagCPStrategy)


def is_interleave() -> bool:
    return isinstance(_STRATEGY, InterleaveCPStrategy)


def cp_active(forward_batch: "ForwardBatch") -> bool:
    """True when the active strategy is engaged for this forward pass —
    metadata is attached and the mode is an extend variant. Most consumers
    should gate on this instead of checking the strategy directly."""
    return (
        _STRATEGY is not None
        and forward_batch.attn_cp_metadata is not None
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def _cp_active_for_per_layer(forward_batch: "ForwardBatch") -> bool:
    s = _STRATEGY
    return s is not None and s.per_layer_attn_cp_comm and cp_active(forward_batch)


# ---------------------------------------------------------------------------
# Helpers (formerly free functions in cp_utils.py + dsa/utils.py)
# ---------------------------------------------------------------------------


def _is_dsa_active() -> bool:
    """Zigzag KV-len computation in DSA mode defers prefix-baking to the
    indexer. The non-DSA generic path bakes ``prefix_len`` into
    ``kv_len_{prev,next}`` directly."""
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return bool(
        getattr(sa, "enable_prefill_cp", False)
        and getattr(sa, "_is_dsa_model_arch", False)
    )


def _strided_take(x: Union[torch.Tensor, List, Tuple], rank: int, world: int):
    if isinstance(x, (tuple, list)):
        indices = range(rank, len(x), world)
        return x[indices]

    n = len(x)
    if n % world != 0:
        cur_len = n // world + (n % world > rank)
        if cur_len == 0:
            return x.new_empty(0, *x.shape[1:])
        indices = torch.arange(rank, n, world, device=x.device)
        return x[indices]
    return x.view(-1, world, *x.shape[1:])[:, rank].contiguous()


def _build_zigzag_metadata(
    kv_len: int,
    cp_rank: int,
    cp_size: int,
    seqs_len: Optional[List[int]],
    *,
    bake_prefix: bool,
) -> ContextParallelMetadata:
    """Compute the full zigzag payload — block lengths, the zigzag
    permutation, per-half kv lengths, and their inverse tables.

    See the diagram in the docstring of the original
    ``prepare_context_parallel_metadata``."""

    kv_len_t = torch.tensor(kv_len)
    kv_len_origin = kv_len_t
    bs_per_cp_group = 1

    prefix_len = 0
    try:
        if seqs_len is not None and len(seqs_len) == 1:
            prefix_len = int(seqs_len[0]) - int(kv_len_origin.item())
            if prefix_len < 0:
                prefix_len = 0
    except Exception:  # noqa: BLE001 — match original error tolerance
        prefix_len = 0

    cp_segment_num = cp_size * 2
    seq_per_batch = kv_len_t // cp_segment_num
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
    remainder = kv_len_t % cp_segment_num
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

    seq_max_rank_len = (kv_len_t + cp_size - 1) // cp_size
    max_rank_len = seq_max_rank_len.repeat_interleave(cp_size).int().tolist()
    zigzag_index = list(
        range(cp_rank, cp_rank + bs_per_cp_group * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs_per_cp_group * cp_segment_num,
            cp_segment_num,
        )
    )

    per_rank_actual_token = list(
        split_list[i] + split_list[cp_size * 2 - i - 1] for i in range(cp_size)
    )
    reverse_split_len = [
        element
        for i in range(cp_size)
        for element in (split_list[i], split_list[cp_size * 2 - i - 1])
    ]
    cp_reverse_index: List[int] = []
    for batch_id in range(bs_per_cp_group):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs_per_cp_group, 2 * bs_per_cp_group))
            + list(
                range(
                    (cp_segment_num - 1) * bs_per_cp_group + batch_id,
                    0,
                    -2 * bs_per_cp_group,
                )
            )
        )
    prefix_sum_list = list(accumulate(split_list))

    if bake_prefix:
        kv_len_prev = prefix_len + prefix_sum_list[cp_rank]
        kv_len_next = prefix_len + prefix_sum_list[cp_size * 2 - cp_rank - 1]
    else:
        kv_len_prev = prefix_sum_list[cp_rank]
        kv_len_next = prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]
    kv_len_prev_tensor = torch.tensor([kv_len_prev], device="cuda", dtype=torch.int32)
    kv_len_next_tensor = torch.tensor([kv_len_next], device="cuda", dtype=torch.int32)
    actual_seq_q_prev_tensor = torch.tensor(
        [actual_seq_q_prev], device="cuda", dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        [actual_seq_q_next], device="cuda", dtype=torch.int32
    )

    return ContextParallelMetadata(
        split_list=split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        kv_len_prev=kv_len_prev,
        kv_len_next=kv_len_next,
        actual_seq_q_prev=actual_seq_q_prev,
        actual_seq_q_next=actual_seq_q_next,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        total_seq_lens=kv_len_origin,
    )


def _interleave_split_q_seqs(extend_seqs_cpu, extend_seqs, cp_size: int, cp_rank: int):
    """Compute per-rank ``q_seqs`` and the ``bs_idx`` mask. CPU pass mirrored
    by a Triton kernel for GPU tensors."""
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_q_seqs_kernel,
    )

    extra_seq = 0
    q_seqs = []
    for cur_len in extend_seqs_cpu:
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx_cpu = [i for i, x in enumerate(q_seqs) if x > 0]
    ret_q_lens_cpu = [q for q in q_seqs if q > 0]

    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx
