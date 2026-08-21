# SPDX-License-Identifier: Apache-2.0
"""Lightweight HTTP registry for source weight manifests."""

import logging
import threading
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)


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

        app = FastAPI()

        @app.get("/health")
        def health():
            return PlainTextResponse("OK")

        @app.put("/register_weight_manifest")
        def register_weight_manifest(data: dict):
            try:
                self.register_weight_manifest(data)
                return PlainTextResponse("OK")
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except Exception as error:
                logger.exception("Failed to register source weight manifest")
                raise HTTPException(status_code=400, detail=str(error)) from error

        @app.get("/get_source_weights_manifest")
        def get_source_weights_manifest():
            try:
                return {"source_weights_manifest": self.get_source_weights_manifest()}
            except RuntimeError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(
            "WeightManifestServer started on %s:%s, expected_ranks=%s",
            host,
            port,
            expected_rank_count,
        )

    def register_weight_manifest(self, data: dict[str, Any]) -> None:
        node_id = str(data["node_id"])
        global_rank = int(data["global_rank"])
        if not 0 <= global_rank < self.expected_rank_count:
            raise ValueError(f"source global rank is out of range: {global_rank}")

        parallel_layout = {
            "tp_size": int(data["parallel_layout"]["tp_size"]),
            "pp_size": int(data["parallel_layout"]["pp_size"]),
            "ep_size": int(data["parallel_layout"]["ep_size"]),
        }
        if parallel_layout["tp_size"] * parallel_layout["pp_size"] != (
            self.expected_rank_count
        ):
            raise ValueError(
                "source parallel layout does not match expected rank count"
            )

        runtime_inventory = data["runtime_inventory"]
        model_identity = (
            str(runtime_inventory["model_id"]),
            str(runtime_inventory["revision"]),
        )
        rank_manifest = {
            "gpu_id": int(data["gpu_id"]),
            "runtime_inventory": runtime_inventory,
            "content_checksums": data["content_checksums"],
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

    def get_source_weights_manifest(self) -> dict[str, Any]:
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

        gpu_ids = tuple(item["gpu_id"] for item in rank_manifests)
        if len(gpu_ids) != len(set(gpu_ids)):
            raise RuntimeError(f"source GPU ranks are not unique: {gpu_ids}")

        return {
            "node_id": self._node_id,
            "parallel_layout": parallel_layout,
            "gpu_ids": gpu_ids,
            "runtime_inventories": tuple(
                item["runtime_inventory"] for item in rank_manifests
            ),
            "content_checksum_groups": tuple(
                item["content_checksums"] for item in rank_manifests
            ),
        }

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
