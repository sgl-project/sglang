from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.repetition import override_repeating_draft_tokens


class RepetitionDraftFallback:
    """Reuse a stable output cycle after sustained greedy draft rejection."""

    _history_size = 32
    _max_period = 8
    _min_verify = 32

    def __init__(self):
        self._active: dict[str, object] = {}

    def select_patterns(self, reqs):
        self._active = {
            req.rid: req for req in reqs if self._active.get(req.rid) is req
        }
        rows = []
        any_active = False
        for req in reqs:
            period = 0
            history = req.output_ids
            eligible = self._active.get(req.rid) is req or (
                req.spec_verify_ct >= self._min_verify
                and req.spec_num_correct_drafts * 4 <= req.spec_verify_ct
            )
            tail = history[-self._history_size :] if eligible else ()
            if eligible and len(tail) == self._history_size:
                for candidate in range(1, self._max_period + 1):
                    if all(
                        tail[i] == tail[i - candidate]
                        for i in range(candidate, len(tail))
                    ):
                        period = candidate
                        break
            if period:
                self._active[req.rid] = req
                any_active = True
                # Overlap output history can trail the live GPU prefix.
                start = len(req.origin_input_ids) + len(history) - len(tail)
                rows.append(
                    [
                        period,
                        start,
                        *tail[:period],
                        *([0] * (self._max_period - period)),
                    ]
                )
            else:
                self._active.pop(req.rid, None)
                rows.append([0] * (self._max_period + 2))
        return rows if any_active else None

    def apply(self, *, reqs, prefix_lens, draft_tokens):
        rows = self.select_patterns(reqs)
        if rows is None:
            return
        patterns = torch.tensor(rows, dtype=torch.int64, device=draft_tokens.device)
        # In-place also updates the static proposal buffer read by graph verify.
        override_repeating_draft_tokens(draft_tokens, prefix_lens, patterns)
