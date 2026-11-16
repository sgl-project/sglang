"""Experimental, soft host-step budget for synchronous prefill admission."""

import math


class AdaptivePrefillController:
    def __init__(self, target_ms=40.0, token_ms=0.02, max_decode_rounds=8, floor=1):
        if not all(math.isfinite(x) and x > 0 for x in (target_ms, token_ms)):
            raise ValueError("Adaptive prefill costs must be finite and positive")
        if not 1 <= floor <= max_decode_rounds:
            raise ValueError(
                "Adaptive decode cadence must have positive bounded limits"
            )
        self.target_ms = target_ms
        self.token_ms = token_ms
        self.decode_ms = 0.0
        self.floor = floor
        self.max_decode_rounds = max_decode_rounds
        self.interval = floor

    def chunk_budget(self, configured, page_size, has_decode):
        if not has_decode:
            return configured
        available_ms = max(self.target_ms * 0.1, self.target_ms - self.decode_ms)
        tokens = int(available_ms / self.token_ms)
        # One page permits progress even when it exceeds the soft time target.
        return min(configured, max(page_size, tokens // page_size * page_size))

    def observe(self, elapsed_ms, prefill_tokens):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        if prefill_tokens > 0:
            measured = elapsed_ms / prefill_tokens
            self.token_ms = max(measured, 0.9 * self.token_ms + 0.1 * measured)
            if elapsed_ms > self.target_ms:
                self.interval = min(self.max_decode_rounds, self.interval + 1)
            elif elapsed_ms < self.target_ms * 0.8:
                self.interval = max(self.floor, self.interval - 1)
        else:
            self.decode_ms = max(elapsed_ms, 0.9 * self.decode_ms + 0.1 * elapsed_ms)
