# SPDX-License-Identifier: Apache-2.0
"""HTTP client for an exact causal realtime VAE decoder."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
    RAW_RGB_CONTENT_TYPE,
    SCHEMA_VERSION,
    materialize_raw_transport_batches_from_shared_memory,
    packb,
    unpackb,
)

REMOTE_VAE_URL_ENV = "SGLANG_REALTIME_REMOTE_VAE_URL"
REMOTE_VAE_TIMEOUT_ENV = "SGLANG_REALTIME_REMOTE_VAE_TIMEOUT"


def get_remote_vae_url() -> str | None:
    value = os.environ.get(REMOTE_VAE_URL_ENV, "").strip()
    return value.rstrip("/") or None


def get_remote_vae_response_transport(url: str) -> str:
    hostname = urlparse(url).hostname
    return "shared_memory" if hostname in {"127.0.0.1", "localhost", "::1"} else "http"


@dataclass(frozen=True)
class RemoteVAEDecodeResult:
    raw_frame_batches: list[list[bytes]] | None
    raw_transport_batches: list[list[dict[str, Any]]] | None
    raw_frame_metadata: dict[str, Any]
    raw_frame_content_type: str
    stats: dict[str, Any]


class RemoteVAEDecodeClient:
    def __init__(self, url: str, *, timeout: float | None = None) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout or float(os.environ.get(REMOTE_VAE_TIMEOUT_ENV, "300"))
        self.session = requests.Session()

    def decode(self, request: dict[str, Any]) -> RemoteVAEDecodeResult:
        body = packb(request)
        started_at = time.monotonic()
        response = self.session.post(
            f"{self.url}/decode",
            data=body,
            headers={"content-type": "application/msgpack"},
            timeout=self.timeout,
        )
        roundtrip_ms = (time.monotonic() - started_at) * 1000.0
        response.raise_for_status()
        result = unpackb(response.content)
        if result.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"remote VAE schema mismatch: {result.get('schema_version')}"
            )
        if result.get("status") != "ok":
            raise RuntimeError(f"remote VAE decode failed: {result}")
        if (
            result.get("raw_frame_batches") is None
            and result.get("raw_transport_batches") is None
        ):
            raise RuntimeError("remote VAE response has no frame transport")
        transport_batches = result.get("raw_transport_batches")
        materialize_ms = 0.0
        if result.get("raw_transport_storage") == "shared_memory":
            materialize_start = time.monotonic()
            transport_batches = materialize_raw_transport_batches_from_shared_memory(
                transport_batches
            )
            materialize_ms = (time.monotonic() - materialize_start) * 1000.0
        return RemoteVAEDecodeResult(
            raw_frame_batches=result.get("raw_frame_batches"),
            raw_transport_batches=transport_batches,
            raw_frame_metadata=result.get("raw_frame_metadata") or {},
            raw_frame_content_type=result.get("raw_frame_content_type")
            or RAW_RGB_CONTENT_TYPE,
            stats={
                **(result.get("stats") or {}),
                "client_http_roundtrip_ms": roundtrip_ms,
                "client_request_bytes": len(body),
                "client_response_bytes": len(response.content),
                "client_shared_memory_materialize_ms": materialize_ms,
            },
        )
