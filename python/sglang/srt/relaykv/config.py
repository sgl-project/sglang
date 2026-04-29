from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RelayKVMode = Literal["off", "shadow"]


@dataclass(frozen=True)
class RelayKVConfig:
    """Runtime knobs for the RelayKV MVP-0 shadow planner.

    Shadow mode must never mutate KV tensors. It only records the logical plan:
    resident budget, recent window, anchor pages, and cold/resident token split.
    """

    enabled: bool = False
    mode: RelayKVMode = "off"
    resident_budget_tokens: int = 0
    recent_window: int = 0
    anchor_pages: int = 0
    log_interval: int = 1
    host_backup_shadow: bool = False
    host_backup_max_mib: float = 0.0

    @classmethod
    def from_server_args(cls, server_args: object) -> "RelayKVConfig":
        enabled = bool(getattr(server_args, "enable_relaykv", False))
        mode = str(getattr(server_args, "relaykv_mode", "off"))
        if not enabled:
            mode = "off"
        if mode not in ("off", "shadow"):
            raise ValueError(f"Unsupported RelayKV mode for MVP-0: {mode!r}")
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            resident_budget_tokens=int(
                getattr(server_args, "relaykv_resident_budget_tokens", 0) or 0
            ),
            recent_window=int(getattr(server_args, "relaykv_recent_window", 0) or 0),
            anchor_pages=int(getattr(server_args, "relaykv_anchor_pages", 0) or 0),
            log_interval=int(getattr(server_args, "relaykv_log_interval", 1) or 1),
            host_backup_shadow=bool(
                getattr(server_args, "relaykv_host_backup_shadow", False)
            ),
            host_backup_max_mib=float(
                getattr(server_args, "relaykv_host_backup_max_mib", 0.0) or 0.0
            ),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.mode != "shadow":
            raise ValueError("MVP-0 supports only --relaykv-mode shadow")
        if self.resident_budget_tokens <= 0:
            raise ValueError("--relaykv-resident-budget-tokens must be > 0")
        if self.recent_window < 0:
            raise ValueError("--relaykv-recent-window must be >= 0")
        if self.anchor_pages < 0:
            raise ValueError("--relaykv-anchor-pages must be >= 0")
        if self.log_interval <= 0:
            raise ValueError("--relaykv-log-interval must be > 0")
        if self.host_backup_max_mib < 0:
            raise ValueError("--relaykv-host-backup-max-mib must be >= 0")
