from __future__ import annotations

from collections import OrderedDict


class PrefetchAdmissionBudget:
    """Conservative request-span reservations, held through L2 admission.

    Waiting entries are metadata only: no tree nodes, allocator indices or lock
    receipts. A fresh tree match is required after granting a reservation.
    """

    def __init__(self):
        self.waiting = OrderedDict()
        self.granted = {}
        self.ready = set()

    def enqueue(self, handle, tokens, payload):
        if handle not in self.granted:
            self.waiting.setdefault(handle, (tokens, payload))

    def release(self, handle):
        self.waiting.pop(handle, None)
        self.granted.pop(handle, None)
        self.ready.discard(handle)

    def clear(self):
        self.waiting.clear()
        self.granted.clear()
        self.ready.clear()

    def candidates(self):
        return sorted(self.waiting, key=lambda h: h not in self.ready)

    def grant(self, handle, capacity, reserve, slots=None):
        tokens, payload = self.waiting[handle]
        # Each Mamba request can own an anchor and an incoming state at once.
        limit = capacity if handle in self.ready else max(0, capacity - reserve)
        slot_limit = (
            slots if handle in self.ready or slots is None else max(0, slots - 2)
        )
        if sum(self.granted.values()) + tokens > limit:
            return None
        if slot_limit is not None and 2 * (len(self.granted) + 1) > slot_limit:
            return None
        self.waiting.pop(handle)
        self.granted[handle] = tokens
        return payload
