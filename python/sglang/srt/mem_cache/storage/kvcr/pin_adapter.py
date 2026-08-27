# SPDX-License-Identifier: Apache-2.0
"""The framework-pinning half of the KVCR bindings, deliberately empty.

``request_pin`` / ``poll_pin_results`` / ``release_pin`` are KVCR-to-framework
callbacks, not the other way round: when a peer asks this worker to serve a
prefix, KVCR first claims what its *own* local DRAM tier holds
(``_claim_local_dram_sources``) and only then asks the framework "do you hold
the rest, and if so pin it and give me addresses". Answering that question
means offering SGLang's ``HostKVCache`` pages as NIXL source memory.

We answer "we hold nothing", because SGLang gives us no way to answer it
safely. A host page is owned by HiCache's own allocator, and a KVCR source
write is asynchronous: between handing over the address and the peer's transfer
completing, HiRadixCache is free to evict that page and refill it for a
different sequence. Nothing on that path errors -- KVCR block keys are token
hashes with no content check, so the peer would accept and decode from whatever
happened to land there. Pinning a host page properly means holding its owning
``TreeNode`` via ``protect_host`` / ``unprotect_host``, which needs a residency
index inside HiRadixCache that this backend does not have (that is the
Shared-HiCache adapter -- separate work).

The cost of the empty answer is a miss, not a wrong result: a key KVCR's local
tier has already evicted is simply not served, and the peer recomputes that
page. Everything this backend deposits lands in that tier, so the common case
is served from there, with KVCR's own claim/refcount holding the slot for the
duration of the write.

The request still has to round-trip -- ``_pin_framework_keys`` has no
synchronous "nothing to pin" return -- so we hand back an id immediately and a
``None`` result on the next poll, which KVCR reads as "the framework holds none
of these" and submits the source write with the local-tier sources alone.
"""

from __future__ import annotations

import logging
import threading
from typing import Collection, List, Tuple

from kvcr.types import BlockKey, PinHandle, PinRequestId, PinResult

logger = logging.getLogger(__name__)


class NoFrameworkPinning:
    """Declines every pin request, one poll after it is made.

    Thread contract: ``request_pin`` / ``poll_pin_results`` /
    ``cancel_pin_request`` are all called from whichever thread is inside
    ``poll_completed`` -- the source pump or the prefetch thread's
    ``_drain_until``, which the store serializes -- but the lock is kept so a
    future caller outside that seam cannot corrupt the queue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_request = 0
        self._declined: List[Tuple[PinRequestId, PinResult]] = []

    def request_pin(self, keys: Collection[BlockKey]) -> PinRequestId:
        with self._lock:
            request = PinRequestId(self._next_request)
            self._next_request += 1
            self._declined.append((request, None))
            return request

    def poll_pin_results(self) -> List[Tuple[PinRequestId, PinResult]]:
        with self._lock:
            declined, self._declined = self._declined, []
        return declined

    def cancel_pin_request(self, request: PinRequestId) -> None:
        """Drop a request KVCR gave up on before we reported it.

        Only reachable when the source op's deadline expires inside the single
        poll interval between ``request_pin`` and ``poll_pin_results``.
        """
        with self._lock:
            self._declined = [entry for entry in self._declined if entry[0] != request]

    def release_pin(self, pin_handle: PinHandle) -> bool:
        """Unreachable: KVCR only releases handles a non-``None`` result gave it.

        Reaching this means KVCR installed a pin we never issued, so the
        contract in the module docstring no longer holds. Say so instead of
        returning a reassuring True.
        """
        logger.error(
            "KVCRStore: asked to release framework pin %r, but this backend "
            "never offers framework memory as a source.",
            pin_handle,
        )
        return False
