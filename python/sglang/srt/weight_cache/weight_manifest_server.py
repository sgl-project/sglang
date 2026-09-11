# SPDX-License-Identifier: Apache-2.0
"""Lightweight TCP registry for source weight manifests."""

import logging
import socket
import threading
from typing import Any, Dict, Optional

from sglang.srt.utils.network import NetworkAddress

from .protocol import recv_msg, send_msg

logger = logging.getLogger(__name__)

CLIENT_CONNECTION_TIMEOUT = 5.0


class WeightManifestServer:
    """Collect per-rank source manifests and publish one complete manifest."""

    def __init__(self, host: str, port: int, expected_rank_count: int):
        if expected_rank_count <= 0:
            raise ValueError("expected_rank_count must be positive")

        self.host = host
        self.port = port
        self.expected_rank_count = expected_rank_count
        self._node_id: Optional[str] = None
        self._parallel_layout: Optional[dict[str, int]] = None
        self._model_identity: Optional[tuple[str, str]] = None
        self._rank_manifests: Dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

        address = NetworkAddress(host, port)
        self._listener = socket.socket(address.family, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(address.to_bind_tuple())
        self.port = self._listener.getsockname()[1]
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="weight-manifest-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "WeightManifestServer started on %s:%s, expected_ranks=%s",
            host,
            self.port,
            expected_rank_count,
        )

    def _serve(self) -> None:
        while not self._stopped.is_set():
            try:
                conn, peer = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stopped.is_set():
                    return
                raise

            conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
            try:
                self._handle_connection(conn)
            except Exception:
                logger.exception(
                    "Failed to handle weight manifest request from %s", peer
                )
            finally:
                conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        if not isinstance(request, dict):
            send_msg(
                conn,
                {"status": "error", "message": "manifest request must be a dict"},
            )
            return

        request_type = request.get("type")
        if request_type == "register_weight_manifest":
            try:
                self.register_weight_manifest(request["manifest"])
            except ValueError as error:
                send_msg(
                    conn,
                    {"status": "conflict", "message": str(error)},
                )
                return
            except Exception as error:
                logger.exception("Failed to register source weight manifest")
                send_msg(conn, {"status": "error", "message": str(error)})
                return
            send_msg(conn, {"status": "ok"})
            return

        if request_type == "get_weight_manifest":
            try:
                manifest = self.get_weight_manifest()
            except RuntimeError as error:
                send_msg(conn, {"status": "retry", "message": str(error)})
                return
            send_msg(
                conn,
                {"status": "ok", "weight_manifest": manifest},
            )
            return

        send_msg(
            conn,
            {
                "status": "error",
                "message": f"unknown manifest request type: {request_type!r}",
            },
        )

    def register_weight_manifest(self, data: dict[str, Any]) -> None:
        node_id = str(data["node_id"])
        global_rank = int(data["global_rank"])
        if not 0 <= global_rank < self.expected_rank_count:
            raise ValueError(f"source global rank is out of range: {global_rank}")

        parallel_layout = {
            "tp_size": int(data["parallel_layout"]["tp_size"]),
            "dp_size": int(data["parallel_layout"].get("dp_size", 1)),
            "pp_size": int(data["parallel_layout"]["pp_size"]),
            "ep_size": int(data["parallel_layout"]["ep_size"]),
        }
        if parallel_layout["tp_size"] * parallel_layout["pp_size"] != (
            self.expected_rank_count
        ):
            raise ValueError(
                "source parallel layout does not match expected rank count"
            )

        runtime_manifest = data["runtime_manifest"]
        model_identity = (
            str(runtime_manifest["model_id"]),
            str(runtime_manifest["revision"]),
        )
        rank_manifest = {
            "device_uuid": str(data["device_uuid"]),
            "runtime_manifest": runtime_manifest,
        }

        with self._lock:
            if self._parallel_layout is None:
                self._node_id = node_id
                self._parallel_layout = parallel_layout
                self._model_identity = model_identity
            elif (
                self._node_id != node_id
                or self._parallel_layout != parallel_layout
                or self._model_identity != model_identity
            ):
                raise ValueError(
                    "source rank manifest node, model identity, or parallel layout differs"
                )

            previous = self._rank_manifests.get(global_rank)
            if previous is not None and previous != rank_manifest:
                raise ValueError(
                    f"source global rank {global_rank} registered conflicting data"
                )
            self._rank_manifests[global_rank] = rank_manifest

        logger.info("Registered source weight manifest global_rank=%s", global_rank)

    def get_weight_manifest(self) -> dict[str, Any]:
        with self._lock:
            if len(self._rank_manifests) != self.expected_rank_count:
                raise RuntimeError(
                    "source weight manifest is not ready: "
                    f"registered={len(self._rank_manifests)}, "
                    f"expected={self.expected_rank_count}"
                )
            rank_manifests = tuple(
                self._rank_manifests[rank] for rank in range(self.expected_rank_count)
            )
            parallel_layout = dict(self._parallel_layout)

        device_uuids = tuple(item["device_uuid"] for item in rank_manifests)
        if len(device_uuids) != len(set(device_uuids)):
            raise RuntimeError(
                f"source device UUIDs are not unique: {device_uuids}"
            )

        return {
            "parallel_layout": parallel_layout,
            "device_uuids": device_uuids,
            "runtime_manifests": tuple(
                item["runtime_manifest"] for item in rank_manifests
            ),
        }

    def close(self) -> None:
        self._stopped.set()
        self._listener.close()
        self._thread.join(timeout=5)
