"""Strict registration schema, value objects, and HTTP router for the reporter.

This module owns the control-plane surface of the embedded load reporter:

* ``StartReportingRequest`` / ``StartReportingResponse`` -- the strict Pydantic
  v2 wire contract for ``POST /v1/start_reporting``.
* ``MonitorKey`` / ``MonitorRegistration`` -- normalized, immutable value
  objects keyed by the Router's ``ip:port``.
* ``normalize_worker_origin`` -- derives the ``Worker.worker_addr`` from the
  registration request origin (never from ``Forwarded``/``X-Forwarded-*``).
* ``WorkerIdentityConflict`` / ``RuntimeClosingError`` -- typed errors mapped to
  HTTP status codes by the route handler.

It intentionally imports nothing from ``monitor.py`` or ``runtime.py`` so those
modules can import the value objects and exceptions here without a cycle; the
route handler reaches the runtime through FastAPI ``app.state``.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress

router = APIRouter()


class StartReportingRequest(BaseModel):
    """Strict registration payload; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    ip: IPvAnyAddress
    port: Annotated[int, Field(strict=True, ge=1, le=65535)]
    report_interval_ms: Annotated[int, Field(strict=True, gt=0)]
    lease_ttl_ms: Annotated[int, Field(strict=True, gt=0)]


class StartReportingResponse(BaseModel):
    """Lease response returned after a reporter registration succeeds."""

    status: str
    lease_ttl_ms: int
    renew_after_ms: int


# ---------------------------------------------------------------------------
# Normalized value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MonitorKey:
    """Canonical Router identity: normalized ``ip`` + ``port``."""

    host: str
    port: int

    @classmethod
    def from_request(cls, value: StartReportingRequest) -> MonitorKey:
        """Normalize a registration payload into a monitor key.

        Args:
            value: Validated Router registration request.

        Returns:
            Canonical IP and port identity for the Router target.
        """
        # ``str(IPvAnyAddress)`` already yields the canonical textual form.
        return cls(str(value.ip), value.port)

    @property
    def authority(self) -> str:
        """gRPC dial target; IPv6 hosts are bracketed."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerIdentity:
    """Stable worker identity included in every load report."""

    worker_addr: str
    worker_type: int
    model: str | None
    zone: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class MonitorRegistration:
    """Immutable, revisioned registration state for one Router target."""

    key: MonitorKey
    worker_identity: WorkerIdentity
    report_interval_ms: int
    lease_expires_at: float
    updated_at: float
    revision: int


# ---------------------------------------------------------------------------
# Control-plane errors
# ---------------------------------------------------------------------------


class WorkerIdentityConflict(Exception):
    """Raised when a live monitor key is re-registered from a different origin."""

    def __init__(
        self,
        key: MonitorKey | None = None,
        *,
        message: str | None = None,
    ) -> None:
        """Initialize a local-key or remote-message identity conflict.

        Args:
            key: Conflicting monitor key when raised by the owner runtime.
            message: Owner-provided message when reconstructed by an IPC proxy.

        Returns:
            None.
        """
        if message is None:
            message = (
                f"monitor {key.authority} is already owned by a different worker origin"
                if key is not None
                else "monitor is already owned by a different worker origin"
            )
        super().__init__(message)
        self.key = key


class RuntimeClosingError(Exception):
    """Raised when registration arrives while the runtime is shutting down."""


# ---------------------------------------------------------------------------
# Origin normalization
# ---------------------------------------------------------------------------


def normalize_worker_origin(request: Request) -> str:
    """Derive ``scheme://host:port`` from the ASGI request URL only.

    Reads ``request.url`` (scheme/hostname/port) and never trusts
    ``Forwarded`` / ``X-Forwarded-*`` headers. A missing port falls back to the
    scheme default (HTTP=80, HTTPS=443). The output is always fully qualified so
    two Routers hitting the same worker record an identical ``worker_addr``.
    """
    url = request.url
    scheme = (url.scheme or "http").lower()
    host = url.hostname or "unknown"
    port = url.port
    if port is None:
        port = 443 if scheme == "https" else 80
    bracketed = f"[{host}]" if ":" in host else host
    return f"{scheme}://{bracketed}:{port}"


@router.post("/v1/start_reporting", response_model=StartReportingResponse)
async def start_reporting(payload: StartReportingRequest, request: Request):
    """Register or renew one internal Router load-reporting target.

    Args:
        payload: Strict Router endpoint and lease configuration.
        request: Incoming FastAPI request used to derive the worker identity.

    Returns:
        The accepted lease and renewal timing.

    Raises:
        HTTPException: For unsupported runtime, identity conflict, or shutdown failures.
    """
    runtime = getattr(request.app.state, "load_reporter_runtime", None)
    unsupported = getattr(request.app.state, "load_reporter_unsupported_reason", None)
    if runtime is None:
        raise HTTPException(
            status_code=501,
            detail=unsupported or "load reporting is unavailable",
        )
    # Local import to break the potential cycle: ipc.py imports
    # StartReportingRequest / StartReportingResponse from this module, so a
    # top-level import of ipc here would create a circular dependency.
    from sglang.srt.load_reporter.ipc import (  # noqa: PLC0415
        LoadReporterDependencyUnavailableError,
        LoadReporterInternalError,
        LoadReporterUnavailableError,
    )

    try:
        return await runtime.start_reporting(payload, normalize_worker_origin(request))
    except WorkerIdentityConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeClosingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LoadReporterUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LoadReporterDependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except LoadReporterInternalError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
