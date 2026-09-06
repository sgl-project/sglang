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
