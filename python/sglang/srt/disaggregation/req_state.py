from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(slots=True, kw_only=True)
class DisaggReqState:
    """A PD request's transfer state, ``req.disagg``: plain data, attached by
    the PD queue that takes the request and written by the stages that move it.
    It survives a retraction, as the KV transfer state did on ``Req``."""

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
