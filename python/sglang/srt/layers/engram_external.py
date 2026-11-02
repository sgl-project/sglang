# SPDX-License-Identifier: Apache-2.0
"""Attach SGLang Engram gathers to an independent immutable table owner."""

import hashlib
import json
import logging
import os
import socket
import threading
from contextlib import suppress
from pathlib import Path

from sglang.srt.layers.engram_shared_vmm import TableHandles, TableMapping, import_table

logger = logging.getLogger(__name__)
MAX_MANIFEST_BYTES = 1 << 20
_attachments: list["ExternalEngramTable"] = []
_attachments_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalEngramTable:
    """Keep one imported GPU mapping and its owner lease together."""

    def __init__(
        self,
        connection: socket.socket,
        mapping: TableMapping,
        lease: str,
        weight_bytes: int,
    ) -> None:
        self.connection = connection
        self.mapping = mapping
        self.lease = lease
        self.weight_ptr = mapping.address
        self.scale_ptr = mapping.address + weight_bytes

    def close(self) -> None:
        # Unmap only after the model runner has stopped and the GPU is idle.
        self.mapping.close()
        try:
            self.connection.sendall(
                json.dumps({"op": "detach", "lease": self.lease}).encode() + b"\n"
            )
        finally:
            self.connection.close()


def attach_table(
    *,
    endpoint: str,
    token_file: Path,
    generation: str,
    checkpoint_dir: Path,
    layer: int,
    rows: int,
    dim: int,
    head_sizes: tuple[int, ...],
    device: int,
) -> ExternalEngramTable:
    """Validate the owner generation before exposing a read-only GPU pointer."""
    host, port = endpoint.rsplit(":", 1)
    connection = socket.create_connection((host, int(port)), timeout=30)
    lease = None
    mapping = None
    try:
        if not checkpoint_dir.is_dir():
            raise ValueError("external Engram needs a local checkpoint directory")
        request = {
            "op": "attach",
            "token": token_file.read_text().strip(),
            "client": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "device": device,
                "layer": layer,
            },
        }
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = bytearray()
        while not response.endswith(b"\n"):
            part = connection.recv(1 << 20)
            if not part:
                raise ConnectionError("Engram owner closed during attach")
            response.extend(part)
            if len(response) > MAX_MANIFEST_BYTES:
                raise ValueError("Engram owner manifest is too large")
        reply = json.loads(response)
        if not reply.get("ok"):
            raise RuntimeError(f"Engram owner rejected attach: {reply.get('error')}")
        lease = reply["lease"]
        manifest = reply["manifest"]
        identity = {
            "config_sha256": _sha256(checkpoint_dir / "config.json"),
            "index_sha256": _sha256(checkpoint_dir / "model.safetensors.index.json"),
        }
        if manifest.get("generation") != generation or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("external Engram generation or checkpoint mismatch")
        spec = manifest["tables"][str(layer)]
        if (
            manifest.get("schema") != 1
            or manifest.get("format") != "FP8_E4M3FN_E8M0_BLOCK32"
            or spec["rows"] != rows
            or spec["dim"] != dim
            or spec["scale_width"] != dim // 32
            or tuple(spec["head_sizes"]) != head_sizes
        ):
            raise ValueError("external Engram table layout mismatch")
        table = TableHandles.from_wire(spec["table"])
        weight_bytes = rows * dim
        if table.size_bytes != rows * (dim + dim // 32):
            raise ValueError("external Engram byte extent mismatch")
        mapping = import_table(table, device)
        attachment = ExternalEngramTable(connection, mapping, lease, weight_bytes)
        with _attachments_lock:
            _attachments.append(attachment)
        return attachment
    except BaseException as error:
        if mapping is not None:
            try:
                mapping.close()
            except Exception as cleanup_error:
                # Lost control makes the owner retain an uncertain lease.
                connection.close()
                error.add_note("external Engram mapping could not be quiesced")
                raise error from cleanup_error
        if lease is not None:
            with suppress(OSError):
                connection.sendall(
                    json.dumps({"op": "detach", "lease": lease}).encode() + b"\n"
                )
        connection.close()
        raise


def close_attached_tables() -> None:
    """Called after SGLang has stopped requests and graph replay."""
    with _attachments_lock:
        for attachment in reversed(_attachments[:]):
            try:
                attachment.close()
            except Exception:
                logger.exception("Cannot safely detach external Engram table")
            else:
                _attachments.remove(attachment)
