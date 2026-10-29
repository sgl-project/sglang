"""Request state-slot demand shared by sizing, admission and row acquisition."""

import msgspec

from sglang.srt.mem_cache.allocation_sizing import get_mamba_tracking_slots


class StateSlotStrategy(msgspec.Struct, frozen=True, kw_only=True):
    radix_cache: bool = True
    decode_only: bool = False
    extra_buffer: bool = False
    overlap: bool = False
    lazy: bool = False
    skip_decode_lock: bool = False

    def __post_init__(self):
        if self.lazy and (not self.overlap or not self.extra_buffer):
            raise ValueError("lazy state tracking requires overlapping extra buffers")

    @property
    def tracking_slots(self) -> int:
        return get_mamba_tracking_slots(
            extra_buffer=self.extra_buffer, overlap=self.overlap
        )

    @property
    def initial_tracking_slots(self) -> int:
        return get_mamba_tracking_slots(
            extra_buffer=self.extra_buffer, overlap=self.overlap, lazy=self.lazy
        )

    def slots_per_request(self) -> int:
        if not self.radix_cache or self.decode_only:
            return 1 + self.initial_tracking_slots
        # Prefill can hold an active slot and two radix checkpoints. With
        # tracked buffers, skipping the decode lock removes one checkpoint.
        checkpoints = 1 if self.skip_decode_lock and self.extra_buffer else 2
        return 1 + checkpoints + self.initial_tracking_slots
