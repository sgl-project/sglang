"""Two-rank peer exchanges for MTP draft steps (``SGLANG_Q35MF_KDENSE_DRAFT_*``).

A ``PeerSlots`` owns one symmetric-memory buffer per call site, pre-filled with the
kernel's arrival marker. It is built on the first eager call (both TP ranks reach the
same call in the same order); calls before that, or under capture before it exists,
keep the stock path on every rank.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.draft_peer_xchg import (
    FP32_MARKER,
    ROW_MARKER,
    argmax_slots_shape,
    as_int32_word,
    fc_epoch_shape,
    fc_slots_shape,
    fc_xchg_sum,
)

logger = logging.getLogger(__name__)


class PeerSlots:
    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        marker: int,
        epoch_shape: Tuple[int, ...],
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.marker = marker
        self.epoch_shape = epoch_shape
        self.slots: Optional[torch.Tensor] = None
        self.peer_slots = 0
        self.epoch: Optional[torch.Tensor] = None
        self.rank = 0
        self.failed = False

    def ready(self) -> bool:
        if self.slots is not None:
            return True
        if self.failed or torch.cuda.is_current_stream_capturing():
            return False
        try:
            self._build()
        except Exception as e:  # every rank sees the same group and shapes
            logger.warning("KDENSE draft exchange %s disabled (%s)", self.name, e)
            self.failed = True
        return self.slots is not None

    def _build(self) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size != 2:
            raise RuntimeError(f"needs tp_size 2, got {tp_group.world_size}")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.inference_mode(False), torch.no_grad():
            slots = symm_mem.empty(self.shape, dtype=self.dtype, device=device)
        # Every word holds the marker before the peer can see this buffer (the
        # rendezvous below is collective).
        slots.view(torch.int32).fill_(as_int32_word(self.marker))
        torch.cuda.synchronize()
        hdl = symm_mem.rendezvous(slots, group=tp_group.device_group)
        self.rank = hdl.rank
        self.peer_slots = int(hdl.buffer_ptrs[1 - hdl.rank])
        self.epoch = torch.zeros(self.epoch_shape, dtype=torch.int32, device=device)
        self.slots = slots
        logger.info(
            "KDENSE draft exchange %s active (rank %d, slots %s)",
            self.name,
            self.rank,
            tuple(self.shape),
        )


def make_argmax_slots(max_rows: int, hidden: int) -> PeerSlots:
    return PeerSlots(
        name="argmax",
        shape=argmax_slots_shape(max_rows, hidden),
        dtype=torch.bfloat16,
        marker=ROW_MARKER,
        epoch_shape=(max_rows,),
    )


class FcPeerSum:
    """bf16(sum of the two ranks' fp32 fc partials) without the all-gather kernel."""

    def __init__(self, max_rows: int, cols: int) -> None:
        self.max_rows = max_rows
        self.cols = cols
        self.peer = PeerSlots(
            name="fc",
            shape=fc_slots_shape(max_rows, cols),
            dtype=torch.float32,
            marker=FP32_MARKER,
            epoch_shape=fc_epoch_shape(max_rows, cols),
        )

    def __call__(self, partial: torch.Tensor) -> Optional[torch.Tensor]:
        if not (
            0 < partial.shape[0] <= self.max_rows and partial.shape[1] == self.cols
        ):
            return None
        if not partial.is_contiguous() or not self.peer.ready():
            return None
        out = partial.new_empty(partial.shape, dtype=torch.bfloat16)
        return fc_xchg_sum(
            partial,
            out,
            slots=self.peer.slots,
            peer_slots=self.peer.peer_slots,
            epoch=self.peer.epoch,
        )


# Draft-graph attention metadata on a side branch (SGLANG_Q35MF_KDENSE_DRAFT_META_BRANCH):
# captured on its own stream from the graph's start, joined before the first attention kernel.
_meta_stream: Optional[torch.cuda.Stream] = None
_meta_join: Optional[torch.cuda.Event] = None


def fork_metadata(launch: Callable[[], None]) -> None:
    global _meta_stream, _meta_join
    if _meta_stream is None:
        _meta_stream = torch.cuda.Stream()
    _meta_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_meta_stream):
        launch()
    event = torch.cuda.Event()
    event.record(_meta_stream)
    _meta_join = event


def join_metadata() -> None:
    global _meta_join
    if _meta_join is not None:
        torch.cuda.current_stream().wait_event(_meta_join)
        _meta_join = None
