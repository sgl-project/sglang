# SPDX-License-Identifier: Apache-2.0
"""Configuration for the KVCR direct linker.

Parsed from ``--hicache-storage-backend-extra-config`` (JSON, or ``@file``),
the same channel the Mooncake and UMBP linkers use despite its name.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import msgspec

from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    MAX_TCP_PORT,
    split_control_endpoint,
)

_UNROUTABLE_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})

# Options the KVCR HiCache adapter (#36409) understood whose meaning changed or
# does not apply here. Each maps to the message the operator needs.
_RETIRED_OPTIONS = {
    "local_dram_bytes": (
        "local_dram_bytes is ambiguous in linker mode; set "
        "local_dram_bytes_per_worker (the total for every scheduler rank on "
        "this worker, divided among them)."
    ),
    "local_dram_slots": "local_dram_slots is derived from the physical layout.",
    "get_timeout_s": "get_timeout_s is replaced by preparation_deadline_ms.",
}


class KVCRLinkerConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Operator-facing settings for ``--unified-cache-external-linker-backend kvcr``."""

    # Total KVCR-owned DRAM for this worker; each local scheduler rank gets an
    # equal share. Required: there is no sensible default for a cache tier.
    local_dram_bytes_per_worker: int
    pin_local_dram: bool = True
    nixl_backend: str = "UCX"

    # Peer control channel. control_port is a base; each rank adds its
    # engine-global attention rank so colocated ranks never collide.
    control_host: str = "0.0.0.0"
    control_port: int = 0
    control_advertise_host: Optional[str] = None
    enable_remote_hint: bool = False

    # KVCR core knobs.
    operation_timeout_ms: int = 20000
    abandon_timeout_ms: int = 60000
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100
    policy: str = "lru"
    enable_telemetry: bool = False

    # Preparation bounds. A request stops waiting at the deadline and admits
    # whatever prefix was confirmed; late completions are drained afterwards.
    preparation_deadline_ms: int = 2000
    max_inflight_prepare_requests: int = 64
    max_inflight_prepare_bytes: int = 8 << 30
    max_prepare_bytes_per_request: int = 2 << 30
    fetch_chunk_pages: int = 32
    # Offloads beyond this many in-flight bytes are declined; the tree retries.
    max_inflight_offload_bytes: int = 8 << 30
    # Late (abandoned) work above this stops new preparation until it drains.
    max_abandoned_bytes: int = 4 << 30
    # Owner-thread idle poll interval; shorter finishes small transfers sooner.
    poll_interval_ms: float = 0.5
    stats_log_interval_s: float = 30.0

    def __post_init__(self) -> None:
        if self.local_dram_bytes_per_worker <= 0:
            raise ValueError("KVCR linker requires local_dram_bytes_per_worker > 0.")
        if self.control_port < 0 or self.control_port > MAX_TCP_PORT:
            raise ValueError(
                f"KVCR control_port ({self.control_port}) is out of range; use 0 "
                f"(OS-assigned, local-only) or 1..{MAX_TCP_PORT}."
            )
        if self.operation_timeout_ms <= 0:
            raise ValueError("KVCR operation_timeout_ms must be positive.")
        if self.abandon_timeout_ms < 2 * self.operation_timeout_ms:
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms."
            )
        if self.preparation_deadline_ms <= 0:
            raise ValueError("KVCR preparation_deadline_ms must be positive.")
        if self.fetch_chunk_pages <= 0:
            raise ValueError("KVCR fetch_chunk_pages must be positive.")
        for name in (
            "max_inflight_prepare_requests",
            "max_inflight_prepare_bytes",
            "max_prepare_bytes_per_request",
            "max_inflight_offload_bytes",
            "max_abandoned_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"KVCR {name} must be positive.")
        if self.poll_interval_ms <= 0:
            raise ValueError("KVCR poll_interval_ms must be positive.")
        self._validate_remote_hint_endpoint()

    def _validate_remote_hint_endpoint(self) -> None:
        """A hint source must be dialable before it binds.

        Port 0 exists only inside this process and cannot be advertised, and
        the bind host is legitimately a wildcard, so remote hints require an
        explicit advertise host and base port.
        """
        if not self.enable_remote_hint:
            return
        if self.control_port <= 0:
            raise ValueError(
                "KVCR enable_remote_hint requires an explicit control_port: an "
                "OS-assigned port cannot be registered for peers to dial."
            )
        advertise = self.control_advertise_host
        if not advertise or advertise in _UNROUTABLE_HOSTS:
            raise ValueError(
                f"KVCR enable_remote_hint cannot advertise {advertise!r}; set "
                "control_advertise_host to an address peers can dial."
            )
        if split_control_endpoint(f"tcp://{advertise}:{self.control_port}") is None:
            raise ValueError(
                f"KVCR control endpoint tcp://{advertise}:{self.control_port} "
                "is not dialable."
            )

    @classmethod
    def from_extra_config(
        cls, extra_config: Optional[Mapping[str, Any]]
    ) -> KVCRLinkerConfig:
        extra_config = dict(extra_config or {})
        for name, message in _RETIRED_OPTIONS.items():
            if name in extra_config:
                raise ValueError(f"KVCR linker config: {message}")
        known = set(cls.__struct_fields__)
        unknown = sorted(set(extra_config) - known)
        if unknown:
            raise ValueError(
                f"KVCR linker config has unknown options {unknown}; known "
                f"options: {sorted(known)}."
            )
        if "local_dram_bytes_per_worker" not in extra_config:
            raise ValueError(
                "KVCR linker config requires local_dram_bytes_per_worker in "
                "--hicache-storage-backend-extra-config."
            )
        try:
            return msgspec.convert(extra_config, cls)
        except msgspec.ValidationError as error:
            raise ValueError(f"KVCR linker config is invalid: {error}") from error
