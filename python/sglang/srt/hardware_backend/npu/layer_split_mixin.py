# SPDX-License-Identifier: Apache-2.0
"""Shared owner-broadcast layer-split machinery for NPU KV pools."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


@dataclass
class _SplitKindState:
    """Pending-broadcast bookkeeping for one logical buffer kind.

    ``remote_layer_id`` is the layer whose payload currently sits in the
    scratch (valid cache); ``pending_*`` tracks the one-layer-ahead kick that
    has been issued but not yet waited on.
    """

    scratch: Optional[torch.Tensor] = None
    remote_layer_id: Optional[int] = None
    pending_remote_layer_id: Optional[int] = None
    pending_remote_broadcast: bool = False
    pending_index: object = None
    pending_dests: List = field(default_factory=list)
    works: List = field(default_factory=list)


class NPULayerSplitMixin:
    """Mixin for NPU KV pools that shard layers across CP prefill ranks.

    Subclasses provide the pool-structure hooks below; everything else
    (shard math, pairing discipline, state machine) lives here so the
    semantics stay identical across pool families.
    """

    def _init_layer_split(self, layer_shard_rank: int, layer_shard_size: int) -> None:
        assert layer_shard_rank is not None and layer_shard_size > 1, (
            f"{type(self).__name__} requires layer_shard_size > 1"
        )
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        self._ls_states: Dict[str, _SplitKindState] = {}

    def _register_split_kind(self, kind: str, scratch: Optional[torch.Tensor]) -> None:
        """Register one broadcast kind with its remote scratch (sized to
        hold one full layer of that family)."""
        assert kind not in self._ls_states, f"duplicate split kind {kind}"
        self._ls_states[kind] = _SplitKindState(scratch=scratch)

    def _split_state(self, kind: str) -> _SplitKindState:
        return self._ls_states[kind]

    def _owned_local_layer_range(self) -> Tuple[int, int]:
        return get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _is_local_layer_owned(self, local_layer_id: int) -> bool:
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_layer_id < owned_end

    def _is_layer_owned(self, layer_id: int) -> bool:
        return self._is_local_layer_owned(layer_id - self.start_layer)

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            layer_id - self.start_layer, self.layer_shard_size, self.layer_num
        )

    def _log_layer_shard_plan(self, label: str = "NPU") -> None:
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)
            partitions.append(f"r{rank}:[{st},{ed})")
        my_start, my_end = self._owned_local_layer_range()
        logger.info(
            f"Layer shard plan (continuous, {label}): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, local=[{my_start},{my_end}), "
            f"global=[{self.start_layer + my_start},{self.start_layer + my_end}), "
            f"partitions={'; '.join(partitions)}"
        )

    def _split_broadcast_units(
        self, kind: str, layer_id: int, index
    ) -> List[Tuple[Optional[torch.Tensor], torch.Tensor]]:
        """Tensors carried by one kind broadcast: ``(owner_src, non_owner_dest)``.

        ``owner_src`` is used as the broadcast root tensor on the owning rank;
        ``non_owner_dest`` receives the payload on every other rank. Shapes
        and dtypes must match across ranks for the same ``(kind, layer_id,
        index)``.
        """
        raise NotImplementedError

    def _split_unpack_after_broadcast(
        self, kind: str, layer_id: int, index, packed: List[torch.Tensor]
    ) -> None:
        """Post-broadcast fixup on non-owner ranks.

        Receives the tensors the broadcast wrote (``non_owner_dest`` in
        :meth:`_split_broadcast_units` order). Default: no-op — row-space
        families broadcast straight into the scratch views (GLM); page-space
        families scatter the packed payload into their scratch here.
        """
        return

    def _broadcast_kind(
        self, kind: str, layer_id: int, index=None, wait: bool = False
    ) -> List:
        """Issue the owner-broadcast of ``layer_id``'s ``kind`` payload.

        The owner sources from its own buffer, other ranks receive into
        their scratch. Returns the async work handles.
        """
        owner_rank = self._get_layer_owner_rank(layer_id)
        cp_group = get_parallel().attn_cp_group
        # torch.distributed expects the *global* rank of the source.
        src = cp_group.ranks[owner_rank]
        is_owner = self.layer_shard_rank == owner_rank
        works: List = []
        dests: List[torch.Tensor] = []
        for owner_src, dest in self._split_broadcast_units(kind, layer_id, index):
            tensor = owner_src if is_owner else dest
            assert tensor is not None
            dests.append(tensor)
            works.append(
                torch.distributed.broadcast(
                    tensor,
                    src=src,
                    group=cp_group.device_group,
                    async_op=True,
                )
            )
        st = self._ls_states[kind]
        st.pending_dests = dests
        st.pending_index = index
        if wait:
            for w in works:
                w.wait()
            if not is_owner:
                self._split_unpack_after_broadcast(kind, layer_id, index, dests)
        return works

    def _finalize_pending_kind(
        self, kind: str, *, set_remote_layer_id: bool = True, wait_works: bool = True
    ) -> None:
        st = self._ls_states[kind]
        if not st.pending_remote_broadcast:
            return
        # Wait the async kick handles before any main-stream use of the
        # scratch (set-side scatter / get-side read) or of the owner's src
        # buffer (the layer set overwrites the broadcast's source).
        if wait_works:
            for w in st.works:
                if w is not None:
                    w.wait()
            if self.layer_shard_rank != self._get_layer_owner_rank(
                st.pending_remote_layer_id
            ):
                self._split_unpack_after_broadcast(
                    kind,
                    st.pending_remote_layer_id,
                    st.pending_index,
                    st.pending_dests,
                )
        st.works = []
        st.pending_dests = []
        st.pending_remote_broadcast = False
        if set_remote_layer_id and st.pending_remote_layer_id is not None:
            st.remote_layer_id = st.pending_remote_layer_id
        st.pending_remote_layer_id = None

    def _ensure_remote_kind(
        self,
        kind: str,
        layer_id: int,
        index=None,
        *,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
    ) -> None:
        """Synchronously materialize ``layer_id``'s ``kind`` payload."""
        st = self._ls_states[kind]
        if layer_transfer_counter is not None:
            layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if st.remote_layer_id == layer_id:
            return
        if st.pending_remote_broadcast:
            self._finalize_pending_kind(
                kind, set_remote_layer_id=st.pending_remote_layer_id == layer_id
            )
            if st.remote_layer_id == layer_id:
                return
        # Fallback materialization: wait immediately, the caller reads the
        # scratch (or, on the owner, its own buffer) right after this returns.
        self._broadcast_kind(kind, layer_id, index, wait=True)
        st.remote_layer_id = layer_id

    def _prefetch_kind(
        self,
        kind: str,
        layer_id: int,
        index=None,
        *,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
    ) -> None:
        """Kick off the owner-broadcast for ``layer_id`` one layer ahead of
        the read. Skipped when already cached/pending for the same layer."""
        st = self._ls_states[kind]
        if st.remote_layer_id == layer_id:
            return
        if st.pending_remote_broadcast:
            if st.pending_remote_layer_id == layer_id:
                return
            self._finalize_pending_kind(
                kind, set_remote_layer_id=False, wait_works=False
            )
        if layer_transfer_counter is not None and layer_transfer_idx is not None:
            layer_transfer_counter.wait_until(layer_transfer_idx)
        st.works = self._broadcast_kind(kind, layer_id, index)
        st.pending_remote_layer_id = layer_id
        st.pending_remote_broadcast = True

    def _invalidate_kind_cache(self, kind: str, layer_id: int) -> None:
        st = self._ls_states[kind]
        if st.remote_layer_id == layer_id:
            st.remote_layer_id = None

    def _split_scratch(self, kind: str) -> Optional[torch.Tensor]:
        return self._ls_states[kind].scratch
