from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional, Tuple

from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

try:
    import nvshmem  # type: ignore
    import nvshmem.core  # type: ignore
    from nvshmem.core.interop import torch as nvshmem_torch  # type: ignore
    try:
        from nvshmem import bindings as nvshmem_bindings  # type: ignore
    except Exception:  # pragma: no cover
        nvshmem_bindings = None  # type: ignore
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    nvshmem = None  # type: ignore
    nvshmem_torch = None  # type: ignore
    nvshmem_bindings = None  # type: ignore
    torch = None  # type: ignore
    dist = None  # type: ignore

_initialized = False
_rank: Optional[int] = None
_world_size: Optional[int] = None
_local_rank: Optional[int] = None
_peer_rank_cached: Optional[int] = None
NVSHMEM_UNIQUE_ID_BYTES = 128

_bootstrap_httpd: Optional[ThreadingHTTPServer] = None
_bootstrap_httpd_thread: Optional[threading.Thread] = None
_bootstrap_uid_payloads: Dict[str, str] = {}
_bootstrap_lock = threading.Lock()


def _init_device(device_index: int) -> Tuple[torch.device, Optional[CudaDevice]]:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    torch_device = torch.device("cuda", device_index)
    try:
        nv_dev = CudaDevice(device_index)
        nv_dev.set_current()
    except Exception:
        nv_dev = None
    return torch_device, nv_dev


def init_nvshmem(
    rank: Optional[int] = None,
    nranks: Optional[int] = None,
    local_rank: Optional[int] = None,
    peer_rank: Optional[int] = None,
) -> bool:
    """Best-effort NVSHMEM bootstrap for torchrun or bootstrap-assisted deployments."""

    global _initialized, _rank, _world_size, _local_rank, _peer_rank_cached

    if _initialized:
        return True

    if nvshmem is None:
        logger.error("nvshmem python package is not available.")
        return False

    try:
        if _init_via_bootstrap_bridge(rank=rank, nranks=nranks, local_rank=local_rank, peer_rank=peer_rank):
            return True
    except Exception:
        raise

def ensure_initialized() -> bool:
    """Best-effort NVSHMEM bootstrap for torchrun or bootstrap-assisted deployments."""

    global _initialized, _rank, _world_size, _local_rank, _peer_rank_cached

    if _initialized:
        return True

    if nvshmem is None:
        logger.error("nvshmem python package is not available.")
        return False

    try:
        if _init_via_bootstrap_bridge(rank=_rank, nranks=_world_size, local_rank=_local_rank, peer_rank=_peer_rank_cached):
            return True
    except Exception:
        raise

def tensor_factory(device: Optional[str] = None) -> Optional[Callable]:
    """Return a callable that allocates NVSHMEM symmetric tensors."""

    if not ensure_initialized():
        return None
    if nvshmem_torch is None:
        return None

    def _factory(shape, dtype):
        # nvshmem_torch.tensor places allocations on the current device; it does
        # not accept a device kwarg. Callers should set torch.cuda.set_device
        # before invoking the factory.
        return nvshmem_torch.tensor(shape, dtype=dtype)

    return _factory


def _maybe_start_local_bootstrap_service(bootstrap: str, role: str, tp_rank: Optional[int] = None) -> None:
            
    host_port = bootstrap.split(":")
    if len(host_port) != 2:
        logger.warning("NVSHMEM bootstrap address %s is invalid; expected host:port.", bootstrap)
        return
    host, port_str = host_port
    port = int(port_str)

    global _bootstrap_httpd, _bootstrap_httpd_thread
    with _bootstrap_lock:
        if _bootstrap_httpd is not None:
            return
        try:
            server = ThreadingHTTPServer((host, port), _BootstrapRequestHandler)
        except OSError as exc:
            logger.warning(
                "Failed to start NVSHMEM bootstrap service on %s:%s: %s",
                host,
                port,
                exc,
            )
            return
        thread = threading.Thread(
            target=server.serve_forever, name="nvshmem-bootstrap", daemon=True
        )
        thread.start()
        _bootstrap_httpd = server
        _bootstrap_httpd_thread = thread
        logger.info(
            "NVSHMEM bootstrap service listening on http://%s:%s/nvshmem_uid", host, port
        )


class _BootstrapRequestHandler(BaseHTTPRequestHandler):
    def do_PUT(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            uid_blob = payload.get("uid")
            pair_id = str(payload.get("pair_id", "default"))
            if not isinstance(uid_blob, str):
                raise ValueError("Missing uid")
            with _bootstrap_lock:
                _bootstrap_uid_payloads[pair_id] = uid_blob
            self.send_response(204)
            self.end_headers()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Invalid NVSHMEM UID payload: %s", exc)
            self.send_response(400)
            self.end_headers()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query or "")
        pair_id = params.get("pair_id", ["default"])[0]
        with _bootstrap_lock:
            uid_blob = _bootstrap_uid_payloads.get(pair_id)
        if uid_blob is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"uid": uid_blob}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):  # noqa: A003
        logger.debug("NVSHMEM bootstrap: " + format, *args)


def _init_via_bootstrap_bridge(
        rank: Optional[int] = None,
        nranks: Optional[int] = None,
        local_rank: Optional[int] = None,
        peer_rank: Optional[int] = None,
    ) -> bool:
    """Initialize NVSHMEM via HTTP bootstrap service."""

    global _initialized, _rank, _world_size, _local_rank, _peer_rank_cached

    bootstrap = os.getenv("SGLANG_NVSHMEM_BOOTSTRAP")
    role = os.getenv("SGLANG_NVSHMEM_ROLE", "").lower()
    if not bootstrap or role not in {"prefill", "decode"}:
        return False

    pair_key="nvshmem_pair_key"
    logger.debug(
        "NVSHMEM bootstrap parameters role=%s pair_key=%s rank=%s peer_rank=%s",
        role,
        pair_key,
        rank,
        peer_rank,
    )

    torch_device, nv_dev = _init_device(device_index=local_rank)
    timeout_s = float(os.getenv("SGLANG_NVSHMEM_BOOTSTRAP_TIMEOUT", "180"))
    poll_interval = float(os.getenv("SGLANG_NVSHMEM_BOOTSTRAP_INTERVAL", "1.0"))
    deadline = time.time() + timeout_s

    if role == "decode" and local_rank == 0:
        _maybe_start_local_bootstrap_service(bootstrap, role, rank)
        uid_obj = _publish_uid_to_bootstrap(
            bootstrap,
            pair_key,
            deadline,
            poll_interval,
            role,
            rank,
        )
    else:
        uid_obj = _fetch_uid_from_bootstrap(
            bootstrap,
            pair_key,
            deadline,
            poll_interval,
            role,
            rank,
        )

    init_kwargs = dict(
        uid=uid_obj,
        rank=rank,
        nranks=nranks,
        initializer_method="uid",
    )
    if nv_dev is not None:
        init_kwargs["device"] = nv_dev

    try:
        nvshmem.core.init(**init_kwargs)  # type: ignore[arg-type]
        nvshmem_bindings.barrier_all()
        _initialized = True
        _rank = rank
        _peer_rank_cached = peer_rank
        _local_rank = local_rank
        _world_size = nranks
        logger.info(
            "Initialized NVSHMEM runtime via bootstrap bridge (role=%s rank=%s pair=%s world=%s).",
            role,
            rank,
            peer_rank,
            nranks,
        )
        return True
    except Exception as exc:
        logger.error("Failed NVSHMEM init via bootstrap bridge: %s", exc)
        return False


def _encode_uid(uid_obj) -> str:
    return base64.b64encode(pickle.dumps(uid_obj)).decode("ascii")


def _publish_uid_to_bootstrap(
    bootstrap: str,
    pair_key: str,
    deadline: float,
    poll_interval: float,
    role: str,
    rank: int,
):
    """Leader publishes the UID to the bootstrap HTTP server."""

    try:
        raw_uid = nvshmem.core.get_unique_id()
    except Exception as exc:
        raise RuntimeError(f"Failed to get NVSHMEM unique id: {exc}") from exc

    payload = {"uid": _encode_uid(raw_uid), "pair_key": pair_key}
    url = f"http://{bootstrap}/nvshmem_uid"
    while time.time() < deadline:
        try:
            resp = requests.put(url, json=payload, timeout=5)
            if resp.status_code in (200, 204):
                logger.info(
                    "Published NVSHMEM UID to %s (role=%s rank=%s pair=%s).",
                    url,
                    role,
                    rank,
                    pair_key,
                )
                return raw_uid
        except Exception as exc:
            logger.warning(
                "Failed to publish NVSHMEM UID to %s (role=%s rank=%s pair=%s): %s",
                url,
                role,
                rank,
                pair_key,
                exc,
            )
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Unable to publish NVSHMEM UID to bootstrap server {url} "
        f"(role={role} rank={rank} pair={pair_key}) within allotted time."
    )


def _fetch_uid_from_bootstrap(
    bootstrap: str,
    pair_key: str,
    deadline: float,
    poll_interval: float,
    role: str,
    rank: int,
):
    """Non-leader ranks wait for the leader to publish the UID."""

    url = f"http://{bootstrap}/nvshmem_uid?pair_key={pair_key}"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                return pickle.loads(base64.b64decode(payload["uid"]))
        except Exception:
            pass
        logger.info(
            "Waiting for NVSHMEM UID from %s (role=%s rank=%s pair=%s)...",
            url,
            role,
            rank,
            pair_key,
        )
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Timed out waiting for NVSHMEM UID from bootstrap server {url} "
        f"(role={role} rank={rank} pair={pair_key})."
    )


def get_rank() -> Optional[int]:
    return _rank


def get_peer_rank(default: Optional[int] = None) -> Optional[int]:
    global _peer_rank_cached
    if _peer_rank_cached is not None:
        return _peer_rank_cached

    if _rank is not None and _world_size is not None and _world_size >= 2:
        if _world_size % 2 == 0:
            half = _world_size // 2
            if _rank < half:
                default_peer = _rank + half
            else:
                default_peer = _rank - half
        else:
            default_peer = (_rank + 1) % _world_size if _world_size > 1 else None
    else:
        default_peer = default

    _peer_rank_cached = default_peer
    return _peer_rank_cached


def get_peer_tensor_fn():
    if nvshmem_torch is None:
        return None
    try:
        from nvshmem.core.interop.torch import get_peer_tensor  # type: ignore

        return get_peer_tensor
    except Exception:
        return None


def quiet_current_stream():
    """Best-effort NVSHMEM flush on the current CUDA stream."""

    if torch is not None and torch.cuda.is_available():
        try:
            if nvshmem_bindings is not None:
                nvshmem_bindings.quiet_on_stream(torch.cuda.current_stream().cuda_stream)
                return
        except Exception:
            pass
    # Fallback to a global quiet if the bindings path is unavailable.
    try:
        nvshmem.core.quiet()  # type: ignore[attr-defined]
    except Exception:
        pass
