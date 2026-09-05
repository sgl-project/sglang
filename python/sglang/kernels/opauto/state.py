"""Process-local OpAuto backend state (cold / warm / failed / measured)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class BackendStatus(str, Enum):
    COLD = "cold"
    WARM = "warm"
    FAILED = "failed"


Key = Tuple[str, str, str, str]  # op_id, backend, arch, shape_bucket


@dataclass
class BackendState:
    status: BackendStatus = BackendStatus.COLD
    measured_us: Optional[float] = None
    reason: str = ""


@dataclass
class OpAutoState:
    """In-process sticky state for OpAuto decisions."""

    _entries: Dict[Key, BackendState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def key(
        self,
        op_id: str,
        backend: str,
        *,
        arch: str = "",
        shape_bucket: str = "*",
    ) -> Key:
        return (op_id, backend, arch or _default_arch(), shape_bucket)

    def get(
        self,
        op_id: str,
        backend: str,
        *,
        arch: str = "",
        shape_bucket: str = "*",
    ) -> BackendState:
        k = self.key(op_id, backend, arch=arch, shape_bucket=shape_bucket)
        with self._lock:
            return self._entries.get(k, BackendState())

    def mark_warm(
        self,
        op_id: str,
        backend: str,
        *,
        arch: str = "",
        shape_bucket: str = "*",
        measured_us: Optional[float] = None,
    ) -> None:
        k = self.key(op_id, backend, arch=arch, shape_bucket=shape_bucket)
        with self._lock:
            st = self._entries.get(k, BackendState())
            st.status = BackendStatus.WARM
            if measured_us is not None:
                st.measured_us = measured_us
            self._entries[k] = st

    def mark_failed(
        self,
        op_id: str,
        backend: str,
        *,
        arch: str = "",
        shape_bucket: str = "*",
        reason: str = "",
    ) -> None:
        k = self.key(op_id, backend, arch=arch, shape_bucket=shape_bucket)
        with self._lock:
            self._entries[k] = BackendState(
                status=BackendStatus.FAILED, reason=reason[:200]
            )

    def is_failed(
        self,
        op_id: str,
        backend: str,
        *,
        arch: str = "",
        shape_bucket: str = "*",
    ) -> bool:
        return (
            self.get(op_id, backend, arch=arch, shape_bucket=shape_bucket).status
            is BackendStatus.FAILED
        )

    def demoted_backends(self) -> Dict[str, list[str]]:
        """Map op_id -> list of demoted backend names (any arch/bucket)."""
        out: Dict[str, list[str]] = {}
        with self._lock:
            for (op_id, backend, _arch, _bucket), st in self._entries.items():
                if st.status is BackendStatus.FAILED:
                    out.setdefault(op_id, [])
                    if backend not in out[op_id]:
                        out[op_id].append(backend)
        return out

    def snapshot(self) -> dict:
        with self._lock:
            return {
                f"{op}|{backend}|{arch}|{bucket}": {
                    "status": st.status.value,
                    "measured_us": st.measured_us,
                    "reason": st.reason,
                }
                for (op, backend, arch, bucket), st in self._entries.items()
            }

    def load_snapshot(self, data: dict) -> None:
        with self._lock:
            for key, payload in (data or {}).items():
                parts = str(key).split("|")
                if len(parts) != 4:
                    continue
                op, backend, arch, bucket = parts
                status = payload.get("status", "cold")
                try:
                    st = BackendStatus(status)
                except ValueError:
                    continue
                self._entries[(op, backend, arch, bucket)] = BackendState(
                    status=st,
                    measured_us=payload.get("measured_us"),
                    reason=payload.get("reason") or "",
                )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_STATE = OpAutoState()


def get_state() -> OpAutoState:
    return _STATE


def _default_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception:
        pass
    return "unknown"
