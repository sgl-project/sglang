"""Resolved configuration for shared LayerSplit staging.

window_size and page_size count tokens, matching native HiCache units.
Both directions use the same window, one fixed buffer each and one data PG pool.
Direction-specific names remain only for genuinely different policies (GET/SET
deadlines and the backup outstanding-work limit).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class StagingBufferConfig:
    window_size: int = 8192
    requested_exchange_group_count: int = 8
    get_timeout_s: float = 60.0
    set_timeout_s: float = 60.0
    exchange_timeout_s: float = 5.0
    window_agreement_timeout_s: float = 60.0
    control_timeout_s: float = 60.0
    write_ack_stall_timeout_s: float = 120.0
    backup_max_outstanding_operations: int = 256
    backup_drop_log_interval: int = 64
    page_size: Optional[int] = None
    shard_size: Optional[int] = None

    def __post_init__(self):
        for name in (
            "window_size",
            "requested_exchange_group_count",
            "backup_max_outstanding_operations",
            "backup_drop_log_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "get_timeout_s",
            "set_timeout_s",
            "exchange_timeout_s",
            "window_agreement_timeout_s",
            "control_timeout_s",
            "write_ack_stall_timeout_s",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("page_size", "shard_size"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.page_size is not None and self.window_size % self.page_size:
            raise ValueError("window_size must be a multiple of page_size")

    @classmethod
    def from_environment(cls):
        # Resolve the common window and data-group budget once at startup.
        defaults = cls()
        return cls(
            window_size=int(
                os.getenv("SGLANG_L3_STAGING_WINDOW_TOKENS", defaults.window_size)
            ),
            requested_exchange_group_count=int(
                os.getenv(
                    "SGLANG_L3_STAGING_EXCHANGE_GROUPS",
                    defaults.requested_exchange_group_count,
                )
            ),
        )

    def with_host_layout(
        self, *, page_size: int, shard_size: int
    ) -> StagingBufferConfig:
        return replace(self, page_size=page_size, shard_size=shard_size)

    def require_host_layout(self):
        if self.page_size is None or self.shard_size is None:
            raise ValueError("staging_buffer_config needs the native host layout")

    @property
    def pages_per_window(self):
        self.require_host_layout()
        return self.window_size // self.page_size

    @property
    def pages_per_rank_per_window(self):
        return -(-self.pages_per_window // self.shard_size)

    @property
    def exchange_group_count(self):
        return min(self.requested_exchange_group_count, self.pages_per_rank_per_window)
