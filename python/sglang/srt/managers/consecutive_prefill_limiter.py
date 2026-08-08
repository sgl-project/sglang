class ConsecutivePrefillLimiter:
    """Reserve a decode turn after a bounded run of prefill batches.

    The scheduler normally prioritizes prefill whenever one is available. This
    opt-in limiter preserves that default at a limit of zero, while allowing a
    runnable decode batch to make progress after a configured number of
    consecutive prefill batches.
    """

    def __init__(self, max_consecutive_prefill_batches: int):
        if max_consecutive_prefill_batches < 0:
            raise ValueError("max_consecutive_prefill_batches must be non-negative")
        self._limit = max_consecutive_prefill_batches
        self._consecutive_prefill_batches = 0

    def should_force_decode(self, *, has_runnable_decode: bool) -> bool:
        return (
            has_runnable_decode
            and self._limit > 0
            and self._consecutive_prefill_batches >= self._limit
        )

    def on_prefill(self) -> None:
        if self._limit > 0:
            self._consecutive_prefill_batches = min(
                self._consecutive_prefill_batches + 1, self._limit
            )

    def on_decode(self) -> None:
        self._consecutive_prefill_batches = 0
