from typing import Optional


class StateManager:
    _iteration: int = 0
    _global_clock: float = 0
    _last_inference_dur: float = 0
    _current_inference_dur: float = 0
    _hicache_l2_load_dur: float = 0
    _hicache_l2_backup_dur: float = 0
    _hicache_l2_load_call_count: int = 0
    _hicache_l2_load_segment_count: int = 0
    _hicache_l2_load_units: int = 0
    _hicache_l2_load_bytes: float = 0
    _last_real_time_ts: float = 0
    _last_flush_time_ts: float = 0

    # In-flight KV cache transfers, keyed by request id, valued by the simulated
    # clock time the transfer completes. A decode instance waiting on these has
    # no batch to run, so they are the only thing that can move its clock.
    _pending_transfer_completions: dict[str, float] = {}

    @classmethod
    def set_pending_transfer_completion(cls, rid: str, completion_time: float) -> None:
        cls._pending_transfer_completions[rid] = completion_time

    @classmethod
    def get_pending_transfer_completion(cls, rid: str) -> Optional[float]:
        return cls._pending_transfer_completions.get(rid)

    @classmethod
    def clear_pending_transfer_completion(cls, rid: str) -> None:
        cls._pending_transfer_completions.pop(rid, None)

    @classmethod
    def next_pending_transfer_completion(cls, after: float) -> Optional[float]:
        """Earliest in-flight transfer completion at or after `after`.

        Equality matters for transfers that complete immediately (for example
        without a KV transfer model): one zero-length idle turn lets the
        scheduler poll and commit them without an artificial fixed step.
        """
        pending = [
            ts for ts in cls._pending_transfer_completions.values() if ts >= after
        ]
        return min(pending) if pending else None

    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0
        cls._last_inference_dur = 0
        cls._current_inference_dur = 0
        cls._hicache_l2_backup_dur = 0
        cls._hicache_l2_load_dur = 0
        cls._hicache_l2_load_call_count = 0
        cls._hicache_l2_load_segment_count = 0
        cls._hicache_l2_load_units = 0
        cls._hicache_l2_load_bytes = 0
        cls._last_real_time_ts = 0

        cls._pending_transfer_completions.clear()

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_load_stats(
        cls,
        call_count: int = 0,
        segment_count: int = 0,
        units: int = 0,
        bytes_: float = 0,
    ) -> None:
        cls._hicache_l2_load_call_count += call_count
        cls._hicache_l2_load_segment_count += segment_count
        cls._hicache_l2_load_units += units
        cls._hicache_l2_load_bytes += bytes_

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0
        return dur

    @classmethod
    def pop_hicache_l2_load_stats(cls) -> dict:
        stats = {
            "h2d_load_call_count": cls._hicache_l2_load_call_count,
            "h2d_load_segment_count": cls._hicache_l2_load_segment_count,
            "h2d_load_units": cls._hicache_l2_load_units,
            "h2d_load_bytes": cls._hicache_l2_load_bytes,
        }
        cls._hicache_l2_load_call_count = 0
        cls._hicache_l2_load_segment_count = 0
        cls._hicache_l2_load_units = 0
        cls._hicache_l2_load_bytes = 0
        return stats

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0
        return dur

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur

    @classmethod
    def set_last_real_time_ts(cls, ts):
        cls._last_real_time_ts = ts

    @classmethod
    def get_last_real_time_ts(cls):
        return cls._last_real_time_ts

    @classmethod
    def set_last_flush_time_ts(cls, ts: float):
        cls._last_flush_time_ts = ts

    @classmethod
    def get_last_flush_time_ts(cls) -> float:
        return cls._last_flush_time_ts
