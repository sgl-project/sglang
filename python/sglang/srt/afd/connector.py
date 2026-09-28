"""Connector ownership for transport, role graphs, and aggregate CLOSE."""

from __future__ import annotations

import threading
from typing import Any, Callable

from .contracts import AFDGraphStrategy, AFDTransport


class AFDConnector:
    """Own exactly one transport and one whole-role graph service."""

    def __init__(
        self,
        *,
        transport: AFDTransport,
        graph_strategy: AFDGraphStrategy,
        usage_emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        transport.validate_invariants()
        self.transport = transport
        self.graph_strategy = graph_strategy
        self._usage_emitter = usage_emitter
        self._close_receipt: dict[str, Any] | None = None
        self._close_error: RuntimeError | None = None
        self._close_lock = threading.Lock()
        self._shutdown_requested = False

    def close(self) -> dict[str, Any]:
        """Emit aggregate final usage and close owned resources exactly once."""

        with self._close_lock:
            return self._close_locked()

    def _close_locked(self) -> dict[str, Any]:
        if self._close_receipt is not None:
            if self._close_error is not None:
                raise self._close_error
            return self._close_receipt
        failures: list[tuple[str, Exception]] = []
        try:
            graph_usage = self.graph_strategy.close()
        except Exception as exc:
            failures.append(("graph", exc))
            graph_usage = self.graph_strategy.usage(status="CLOSE_FAILED")
        try:
            peer_usage = self.transport.exchange_close(usage=graph_usage)
        except Exception as exc:
            failures.append(("close_exchange", exc))
            peer_usage = {}
        try:
            transport_usage = self.transport.close()
        except Exception as exc:
            failures.append(("transport", exc))
            transport_usage = {}
        receipt = {
            "schema": "afd-aggregate-close-v1",
            "event": "CLOSE",
            "usage_final": True,
            "role": self.transport.role.value,
            "graph": graph_usage,
            "peer_graph": peer_usage,
            "transport": transport_usage,
            "failures": tuple(component for component, _ in failures),
        }
        if self._usage_emitter is not None:
            try:
                self._usage_emitter(receipt)
            except Exception as exc:
                failures.append(("usage_emitter", exc))
                receipt["failures"] = tuple(component for component, _ in failures)
        self._close_receipt = receipt
        if failures:
            components = tuple(component for component, _ in failures)
            self._close_error = RuntimeError(
                f"AFD_CONNECTOR_AGGREGATE_CLOSE_FAILED components={components!r}"
            )
            raise self._close_error from failures[0][1]
        return receipt

    def request_shutdown(self) -> None:
        """Signal-safe request; the bounded handshake runs at a safe point."""

        self._shutdown_requested = True

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested
