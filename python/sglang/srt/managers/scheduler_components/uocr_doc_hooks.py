"""Scheduler hooks for Unlimited-OCR v2 split-document windows and decode graph I/O.

The scheduler builds ``UocrDocHooks`` only for Unlimited-OCR v2 and guards every
call site with a ``None`` check, so other models never enter the document modules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from sglang.srt.session.uocr_doc_prefetch import DocPrefetch, EarlyEmbeddings
from sglang.srt.session.uocr_doc_reuse import (
    META_FORCED_TOKENS,
    DocReuse,
    forced_len,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.model_executor.runner.uocr_decode_graph_io import (
        UOCRDecodeGraphIO,
    )

logger = logging.getLogger(__name__)


def _v2_backend(attn_backend: AttentionBackend):
    from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
        UnlimitedOCRV2AttnBackend,
    )

    if isinstance(attn_backend, UnlimitedOCRV2AttnBackend):
        return attn_backend
    logger.warning(
        "[Unlimited-OCR v2] attention backend %s cannot run document KV reuse; "
        "every window recomputes",
        type(attn_backend).__name__,
    )
    return None


class UocrDocHooks:
    """Route scheduler events to Unlimited-OCR v2 document KV reuse and prefetch.

    Owns the scheduler's ``DocReuse`` and ``DocPrefetch`` state; the radix cache
    and the attention backend get the reuse state from here at construction.
    """

    def __init__(
        self,
        *,
        tree_cache: UnifiedRadixCache,
        attn_backend: AttentionBackend,
        early_embeddings: Optional[EarlyEmbeddings],
        device: torch.device,
        forward_stream: Optional[torch.cuda.Stream],
        tp_size: int,
        decode_graph_io: Callable[[], Optional[UOCRDecodeGraphIO]],
    ):
        self._prefetch = DocPrefetch(
            early=early_embeddings,
            device=device,
            forward_stream=forward_stream,
            tp_size=tp_size,
        )
        backend = _v2_backend(attn_backend)
        self._reuse = DocReuse(
            cache=tree_cache,
            backend=backend,
            on_window_finished=self._prefetch.on_window_finished,
        )
        if backend is not None:
            backend.attach_doc_reuse(self._reuse)
        tree_cache.enable_uocr_doc_reuse(self._reuse)
        # Resolved per step: a decode-graph recapture replaces the runner's I/O.
        self._decode_graph_io = decode_graph_io
        self._last_graph_io: Optional[UOCRDecodeGraphIO] = None

    # -- Waiting queue and admission --

    def park(self, req: Req, *, is_retracted: bool) -> bool:
        """Hold a split document's next window off-queue; True if parked."""
        return not is_retracted and self._prefetch.park(req)

    def take_released(self) -> List[Req]:
        """Parked windows whose predecessor finished, to queue in this order."""
        return self._prefetch.take_released()

    def on_retracted(self, req: Req) -> None:
        """Restore a retracted window's original prompt before it is re-queued."""
        self._reuse.on_retracted(req)

    def queue_position(self, waiting_queue: List[Req], req: Req) -> Optional[int]:
        """Waiting-queue index for a follow-up window, or None to append."""
        return self._reuse.queue_position(waiting_queue, req)

    def on_req_dropped(self, req: Req) -> None:
        """Release reuse and prefetch state of a request aborted before running."""
        self._reuse.on_req_dropped(req)
        self._prefetch.release_early(req.rid)

    def free_rows_for_admission(self, waiting_queue: List[Req]) -> None:
        """Free a retained window when retained windows hold every request row."""
        self._reuse.free_for_admission(waiting_queue)

    def defer_new_document(
        self, *, req: Req, adder: PrefillAdder, nothing_running: bool
    ) -> bool:
        """Whether a new document must wait so retained windows' follow-ups fit."""
        return self._reuse.defer_new_document(
            req=req, adder=adder, nothing_running=nothing_running
        )

    def on_no_token(
        self, *, req: Req, adder: PrefillAdder, nothing_running: bool
    ) -> bool:
        """Evict retained windows for a rejected request; True if memory was freed."""
        return self._reuse.on_no_token(
            req=req, adder=adder, nothing_running=nothing_running
        )

    def free_for_decode(self, batch: ScheduleBatch) -> None:
        """Evict retained windows before the decode batch would retract."""
        self._reuse.free_for_decode(batch)

    def drop_parked(self, rid_prefix: str) -> List[Req]:
        """Remove and return the held-off windows an abort names."""
        return self._prefetch.drop(rid_prefix)

    def release_doc(self, doc_id: str) -> None:
        """Free a document's retained window once no follow-up will consume it."""
        self._reuse.release_doc(doc_id)
        self._prefetch.forget_doc(doc_id)

    # -- Flush --

    def blocks_flush(self, *, ignore_waiting: bool) -> bool:
        """Whether document state a cache flush cannot drop is live.

        A matched follow-up holds copied prefix slots and its source window, so
        a pool clear under it would leave both dangling. Held-off windows are
        waiting requests: they block like the waiting queue does.
        """
        return self._reuse.has_sources() or (
            not ignore_waiting and self._prefetch.held_count() > 0
        )

    def reset(self) -> List[Req]:
        """Drop all document state before a pool clear; returns windows to abort.

        Retained windows are forgotten, not freed: the clear returns their KV.
        """
        self._reuse.reset()
        return self._prefetch.reset()

    # -- Forward --

    def _resolve_graph_io(
        self, output_tokens_buf: Optional[torch.Tensor]
    ) -> Optional[UOCRDecodeGraphIO]:
        io = self._decode_graph_io()
        if io is not self._last_graph_io:
            if (
                io is not None
                and output_tokens_buf is not None
                and io.relay.shape == output_tokens_buf.shape
            ):
                # A recaptured graph starts from an empty relay; every replay
                # loads its inputs from it, and the future map holds them all.
                io.relay.copy_(output_tokens_buf)
            self._last_graph_io = io
        return io

    def prepare_forward_inputs(
        self, batch: ScheduleBatch, future_map: FutureMap
    ) -> bool:
        """Apply forced decode inputs; True if the decode graph loads the inputs."""
        output_tokens_buf = (
            future_map.output_tokens_buf if future_map is not None else None
        )
        graph_io = self._resolve_graph_io(output_tokens_buf)
        # Forced inputs must land before any gather of the relayed tokens.
        self._reuse.apply_input_overrides(
            output_tokens_buf=output_tokens_buf,
            relay=graph_io.relay if graph_io is not None else None,
        )
        self._reuse.note_prompt_lens(batch)
        return graph_io is not None and graph_io.can_defer(batch)

    def relay_result(
        self,
        batch: ScheduleBatch,
        batch_result: GenerationBatchResult,
        future_indices: torch.Tensor,
    ) -> None:
        """Seed the graph relay from a result the decode graph did not publish."""
        graph_io = self._decode_graph_io()
        if graph_io is not None:
            graph_io.relay_result(batch, batch_result, future_indices)

    def after_forward(
        self, batch: ScheduleBatch, *, chunked_req: Optional[Req]
    ) -> None:
        """Splice the overlap pages' text into windows whose prefill just ran."""
        self._reuse.after_prefill_forward(batch, chunked_req=chunked_req)

    # -- Results and accounting --

    def consume_spliced_prefill(self, req: Req) -> bool:
        """True once for a window whose first outputs were spliced in, not sampled."""
        st = req.uocr_doc
        if st is None or not st.skip_prefill_append:
            return False
        st.skip_prefill_append = False
        return True

    def forced_tokens(self, req: Req) -> int:
        """Output tokens copied from the previous window rather than generated."""
        return forced_len(req)

    def with_forced_tokens(self, customized_info: dict, forced: List[int]) -> dict:
        """Report spliced token counts so the caller can trim the window's answer."""
        if not any(forced):
            return customized_info
        return {
            **customized_info,
            META_FORCED_TOKENS: [[n] for n in forced],
        }

    def held_tokens(self) -> int:
        """KV tokens retained windows hold outside any session record."""
        return self._reuse.held_tokens()

    def held_req_count(self) -> int:
        """Request rows retained windows hold."""
        return len(self._reuse.held_slots())
