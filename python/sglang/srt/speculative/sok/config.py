"""SOK configuration and safety limits."""

from dataclasses import dataclass, field


@dataclass
class SOKConfig:
    """Configuration for PHANTOM-SOK subsystem.

    Safety-first defaults: cache and prewarm always on, selection and
    autotuning require explicit opt-in.
    """

    # F1: Cache provenance + prewarm (always on)
    enable_cache: bool = True
    enable_prewarm: bool = True
    cache_dir: str = ""  # empty = auto (~/.cache/phantom_sok/)
    max_cache_size_mb: int = 512
    prewarm_top_n: int = 20
    prewarm_timeout_s: float = 30.0

    # F2: Telemetry (always on when available)
    enable_telemetry: bool = True
    telemetry_sample_rate: float = 1.0  # 1.0 = every dispatch
    profile_persist_interval: int = 500  # rounds between profile snapshots

    # F3: Replay/validation (future)
    enable_replay: bool = False
    max_trace_size_mb: int = 128
    trace_sample_rate: float = 0.01  # 1% of dispatches

    # F4: Static kernel selection (future, opt-in)
    enable_selection: bool = False

    # F5: Autotuning (future, developer-only)
    enable_tuning: bool = False
    min_shape_observations: int = 128
    max_candidates_per_epoch: int = 8
    min_promotion_gain_pct: float = 5.0
    max_p95_regression_pct: float = 3.0
    require_exact_acceptance: bool = True
    quarantine_on_first_failure: bool = True

    # Invariant safety rails
    tune_schedules_only: bool = True  # NEVER auto-modify semantics
