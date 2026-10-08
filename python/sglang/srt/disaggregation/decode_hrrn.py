"""Opt-in HRRN ordering before decode KV reservation.

Age advances with admitted estimated prefill work, as engine HRRN advances with
processed prefill work. Every TP rank sees the same requests and counter.
The text-cache estimate only controls ordering; allocator budgets are unchanged.
"""

import math


def parse_uncached_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 1.0
    return fraction if math.isfinite(fraction) and 0 <= fraction <= 1 else 1.0


class DecodeHrrn:
    def __init__(self):
        self.admitted_work = 0
        self.waiting = {}
        self.sequence = 0

    def order(self, entries):
        live = {entry.req.rid for entry in entries}
        self.waiting = {
            rid: state for rid, state in self.waiting.items() if rid in live
        }
        for entry in entries:
            req = entry.req
            if req.rid not in self.waiting:
                # Missing/invalid estimates conservatively mean a cold prompt.
                fraction = parse_uncached_fraction(
                    getattr(req, "prefill_uncached_fraction", None)
                )
                cost = max(1, math.ceil(len(req.origin_input_ids) * fraction))
                self.waiting[req.rid] = (self.admitted_work, cost, self.sequence)
                self.sequence += 1

        def key(entry):
            arrived, cost, sequence = self.waiting[entry.req.rid]
            return (-(self.admitted_work - arrived) / cost, sequence)

        entries.sort(key=key)

    def admitted(self, rid):
        state = self.waiting.pop(rid, None)
        if state is not None:
            self.admitted_work += state[1]
