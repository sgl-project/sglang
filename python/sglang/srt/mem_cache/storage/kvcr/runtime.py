from __future__ import annotations

import ctypes
import json
import logging
import math
import mmap
import socket
import time
from collections import Counter, deque

logger = logging.getLogger(__name__)


class _TelemetryStats:
    """Bounded interval aggregates recorded on the KVCR API owner thread."""

    def __init__(self):
        self.counters = Counter()
        self.gauges = {}
        self.histograms = {}

    def increase_counter(self, name, value, labelvalues=()):
        self.counters[name, labelvalues] += value

    def set_gauge(self, name, value, labelvalues=()):
        self.gauges[name, labelvalues] = value

    def observe_histogram(self, name, value, labelvalues=()):
        key = name, labelvalues
        count, total, maximum = self.histograms.get(key, (0, 0, value))
        self.histograms[key] = count + 1, total + value, max(maximum, value)

    def is_empty(self):
        return not (self.counters or self.gauges or self.histograms)

    def reduce(self):
        values = {
            f"{name}[{','.join(labels)}].{kind}": value
            for kind, metrics in (("counter", self.counters), ("gauge", self.gauges))
            for (name, labels), value in metrics.items()
        }
        for (name, labels), summary in self.histograms.items():
            for kind, value in zip(("count", "sum", "max"), summary):
                values[f"{name}[{','.join(labels)}].{kind}"] = value
        return values


class KVCRRuntime:
    """KVCR resources constructed, polled, and closed by the linker owner thread."""

    def __init__(self, layout, options: dict, *, rank: int):
        defaults = {
            "dram_size_gb": 1,
            "kvcr_service_socket_path": "/run/kvcr/memory.sock",
            "guard_index": 0,
            "compatibility_digest": layout.namespace,
            "control_host": "0.0.0.0",
            "advertise_host": None,
            "control_port": 19500,
            "nixl_port": 20500,
            "operation_timeout_ms": 1000,
            "abandon_timeout_ms": 5000,
            "enable_telemetry": False,
            "trailing_capacity_divisor": 1,
        }
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown KVCR options: {sorted(unknown)}")
        config = defaults | options
        if type(rank) is not int or rank < 0:
            raise ValueError("KVCR rank must be a nonnegative integer")
        if type(config["enable_telemetry"]) is not bool:
            raise ValueError("KVCR enable_telemetry must be a boolean")
        guard_socket = config["kvcr_service_socket_path"]
        if guard_socket is not None and (
            not isinstance(guard_socket, str) or not guard_socket
        ):
            raise ValueError("KVCR kvcr_service_socket_path must be a path or null")
        if type(config["guard_index"]) is not int or config["guard_index"] < 0:
            raise ValueError("KVCR guard_index must be a nonnegative integer")
        if (
            not isinstance(config["compatibility_digest"], str)
            or not config["compatibility_digest"]
        ):
            raise ValueError("KVCR compatibility_digest must be a nonempty string")
        size_gb = config["dram_size_gb"]
        if (
            isinstance(size_gb, bool)
            or not isinstance(size_gb, (int, float))
            or not math.isfinite(size_gb)
            or size_gb <= 0
        ):
            raise ValueError("KVCR dram_size_gb must be finite and positive")
        divisor = config["trailing_capacity_divisor"]
        if type(divisor) is not int or divisor < 1:
            raise ValueError(
                "KVCR trailing_capacity_divisor must be a positive integer"
            )
        full_bytes = sum(
            size * (layout.pool_counts[name] - layout.trailing_counts[name])
            for name, size in layout.pool_layouts
        )
        trailing_bytes = sum(
            size * layout.trailing_counts[name] for name, size in layout.pool_layouts
        )
        capacity = int(size_gb * (1 << 30))
        if capacity < full_bytes + trailing_bytes:
            raise ValueError("KVCR DRAM capacity must fit at least one complete page")
        page_count = capacity * divisor // (full_bytes * divisor + trailing_bytes)
        if full_bytes:
            # Even a large divisor must leave room for one full trailing page.
            page_count = min(page_count, (capacity - trailing_bytes) // full_bytes)
        trailing_pages = max(1, page_count // divisor)
        for name in ("control_port", "nixl_port"):
            port = config[name]
            if type(port) is not int or not 1 <= port + rank <= 65535:
                raise ValueError(f"KVCR {name} plus rank must be in [1, 65535]")
            config[name] += rank
        if config["control_port"] == config["nixl_port"]:
            raise ValueError("KVCR control_port and nixl_port must differ")
        for name in ("operation_timeout_ms", "abandon_timeout_ms"):
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError(f"KVCR {name} must be a positive integer")
        if config["abandon_timeout_ms"] < 2 * config["operation_timeout_ms"]:
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms"
            )
        if config["advertise_host"] is None:
            config["advertise_host"] = socket.gethostbyname(socket.gethostname())
        for name in ("control_host", "advertise_host"):
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"KVCR {name} must be a nonempty host")

        from kvcr import KVCR, KVCRBindings, KVCRSocketError
        from kvcr.config import (
            KVCRBackendConfigs,
            KVCRConfig,
            KVCRGuardConfig,
            LocalDramOptions,
            RemoteFWDramOptions,
        )
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import CacheTier, KVCRStartupError

        self._layout = layout
        self._rank = rank
        self._telemetry_enabled = config["enable_telemetry"]
        self._last_stats_log = time.monotonic()
        self.events = deque()
        self.removals = {}
        self._buffers = []
        self._pin_results = {}
        self._next_pin = 0
        self.client = None

        def inventory(event):
            if event.tier is CacheTier.LOCAL_G2:
                for key in event.keys:
                    decoded = layout.decode_key(key)
                    if decoded is not None:
                        if event.removed:
                            self.removals[key] = decoded[1]
                        else:
                            self.removals.pop(key, None)

        try:
            kvcr_config = KVCRConfig(
                nixl_agent_name=layout.endpoint_name,
                pool_layouts=layout.pool_layouts,
                nixl_listen_port=config["nixl_port"],
                operation_timeout_ms=config["operation_timeout_ms"],
                abandon_timeout_ms=config["abandon_timeout_ms"],
                enable_telemetry=self._telemetry_enabled,
            )
            bindings = KVCRBindings(
                request_pin=self._decline_pin,
                poll_pin_results=self._poll_pin_results,
                release_pin=lambda _: True,
                cancel_pin_request=lambda request: self._pin_results.pop(request, None),
                framework_control=ZmqPeerControlChannel(
                    config["control_host"],
                    config["control_port"],
                    config["advertise_host"],
                ),
                key_adapter=self,
                inventory_sink=inventory,
                on_resilience_event=self.events.append,
                stats_factory=_TelemetryStats if self._telemetry_enabled else None,
            )
            if guard_socket is not None:
                guard = KVCRGuardConfig(
                    kvcr_service_socket_path=guard_socket,
                    guard_index=config["guard_index"] + rank,
                    compatibility_digest=config["compatibility_digest"],
                )
                logger.info(
                    "KVCR Guard: %s, pool layouts: %s", guard, layout.pool_layouts
                )
                try:
                    self.client = KVCR(
                        kvcr_config,
                        bindings,
                        KVCRBackendConfigs(
                            framework_regions=layout.registrations,
                            local_dram=None,
                            remote_fw_dram=RemoteFWDramOptions(
                                opportunistic_query=False
                            ),
                        ),
                        guard,
                    )
                    return
                except KVCRSocketError as error:
                    # A lost claim reply can leave a live lease; only retry before connect.
                    if not isinstance(
                        error.__cause__, (FileNotFoundError, ConnectionRefusedError)
                    ):
                        raise
                    logger.warning("%s; using in-process KVCR DRAM", error)
            pools = []
            for name, size in layout.pool_layouts:
                length = (
                    page_count
                    * (layout.pool_counts[name] - layout.trailing_counts[name])
                    + trailing_pages * layout.trailing_counts[name]
                ) * size
                buffer = mmap.mmap(-1, length)
                self._buffers.append(buffer)
                address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
                pools.append((name, address, length))
            self.client = KVCR(
                kvcr_config,
                bindings,
                KVCRBackendConfigs(
                    framework_regions=layout.registrations,
                    local_dram=LocalDramOptions(pools),
                    remote_fw_dram=RemoteFWDramOptions(opportunistic_query=False),
                ),
            )
        except BaseException as error:
            if isinstance(error, KVCRStartupError) or not isinstance(error, Exception):
                # The owner retains this error until process exit; native work may survive.
                error.kvcr_runtime = self
            else:
                self._close_buffers()
            raise

    def _decline_pin(self, keys):
        self._next_pin += 1
        self._pin_results[self._next_pin] = None
        return self._next_pin

    def _poll_pin_results(self):
        results = list(self._pin_results.items())
        self._pin_results.clear()
        return results

    def encode(self, key: object) -> bytes:
        if not isinstance(key, bytes):
            raise TypeError("KVCR keys must be encoded layout keys")
        return key

    def decode(self, key: bytes) -> int:
        decoded = self._layout.decode_key(key)
        if decoded is None:
            raise ValueError("Unknown KVCR layout key")
        return int(decoded[1][:16], 16)

    def log_stats(self, *, force=False):
        if not self._telemetry_enabled:
            return
        now = time.monotonic()
        elapsed = now - self._last_stats_log
        if not force and elapsed < 5:
            return
        self._last_stats_log = now
        try:
            stats = self.client.get_stats()
            if stats is not None and not stats.is_empty():
                logger.info(
                    "KVCR telemetry rank=%d interval_s=%.6f stats=%s",
                    self._rank,
                    elapsed,
                    json.dumps(stats.reduce(), sort_keys=True),
                )
        except Exception:
            # Optional diagnostics must not prevent transfer progress or shutdown.
            self._telemetry_enabled = False
            logger.warning(
                "KVCR telemetry logging disabled after failure", exc_info=True
            )

    def close(self) -> None:
        if self.client is None:
            raise RuntimeError("KVCR startup did not establish safe resource ownership")
        self.log_stats(force=True)
        self.client.close()
        self._close_buffers()

    def _close_buffers(self) -> None:
        while self._buffers:
            self._buffers[-1].close()
            self._buffers.pop()
