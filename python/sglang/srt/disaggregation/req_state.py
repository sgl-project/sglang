from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.disaggregation.kv_events import OffloadedState


@dataclasses.dataclass(slots=True, kw_only=True)
class DisaggReqState:
    """PD per-request bookkeeping, ``req.disagg``; it survives a retraction."""

    # PD prefill.
    # the start index of the sent kv cache
    # We want to send it chunk by chunk for chunked prefill.
    # After every chunk forward, we do the following:
    # kv_send(req.input_ids[req.disagg.start_send_idx:req.extend_end])
    # start_send_idx = req.extend_end
    start_send_idx: int = 0
    # Decode-side cached-prefix length; base of the staging chunk grid
    # (start_send_idx starts here but advances with every send).
    decode_prefix_len: int = 0
    # At-rest device-resident prefix end, snapshotted on the request's
    # first prefill batch; the cached-prefix early-send never goes past it.
    early_send_prefix_end: Optional[int] = None
    metadata_buffer_index: int = -1
    # Used in overlap sequence to signal that an optimistic request should
    # abort chunking. Set in create_sender, consumed in process_batch_result.
    pending_bootstrap: bool = False
    # Number of optimistic prefill forward passes started. preserved across retracts.
    prefill_attempt_count: int = 0

    # PD decode.
    expected_kv_checksum: int = 0
    # Decode-local: the already-emitted boundary token to replay when a
    # retracted request is rebootstrapped. Set in pause_generation(retract)
    # and consumed in the decode transfer commit; never plumbed to prefill.
    rebootstrap_forced_output_id: Optional[int] = None
    # A retracted request whose prefix KV the prefill recomputes.
    rebootstrap_in_progress: bool = False
    # Under PP, the micro-batch whose consensus resumes this retracted request.
    retraction_mb_id: Optional[int] = None
    # Decode KV offload progress, from the first offload until the release.
    offload: Optional[OffloadedState] = None
