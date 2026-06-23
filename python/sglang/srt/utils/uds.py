"""Helpers for the ``--uds`` (Unix domain socket) HTTP-listener mode.

Used by ``sglang.srt.entrypoints.http_server`` to:

- build uvicorn/Granian bind kwargs (UDS vs host/port),
- format the listener address for startup logs,
- safely clean up stale UDS files before binding,
- self-call the server during warmup via plain HTTP over AF_UNIX,
  since the ``requests`` package can't talk UDS.

UDS is a single-tokenizer, Linux/macOS-only feature; validation in
``ServerArgs.__post_init__`` rejects the unsupported combinations
(SSL, gRPC, multi-tokenizer worker, Windows, etc.).
"""

from __future__ import annotations

import errno
import http.client
import json
import logging
import os
import socket
import stat
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class _UDSResponseWrapper:
    """Mimic the subset of ``requests.Response`` used by warmup code."""

    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self._body.decode("utf-8"))


def uvicorn_bind_kwargs(server_args: ServerArgs) -> dict:
    """UDS and host/port are mutually exclusive (enforced in ServerArgs)."""
    if server_args.uds:
        return {"uds": server_args.uds}
    return {"host": server_args.host, "port": server_args.port}


def format_listen_addr(server_args: ServerArgs) -> str:
    """Human-readable listener address for startup logs."""
    if server_args.uds:
        return f"unix:{server_args.uds}"
    return f"{server_args.host}:{server_args.port}"


def prepare_uds_path(path: str) -> None:
    """Make ``path`` safe to ``bind()`` against.

    Behavior, in order:

    - No-op if nothing exists at ``path``.
    - If a non-socket file (including a symlink, which ``lstat`` reports as
      ``S_IFLNK`` rather than ``S_IFSOCK``) exists, refuse with
      ``FileExistsError``. We refuse to follow symlinks to avoid being tricked
      into unlinking arbitrary files via a hostile path.
    - Otherwise probe with a short non-blocking connect (100 ms timeout --
      much shorter than any reasonable server startup pause, long enough that
      a routine kernel queue blip on CI hardware rarely exceeds it).
      * On ``ConnectionRefusedError`` / ``FileNotFoundError``: nobody is
        bound; treat as stale.
      * On ``TimeoutError``: probe inconclusive; treat as live (conservative
        refuse) rather than risk unlinking a slow-but-running listener.
      * On ``PermissionError``: surface the error so the operator can fix
        ownership / mode rather than have us silently clobber or refuse.
      * On any other ``OSError``: treat as live for the same conservative
        reason as ``TimeoutError``.
    - Live → raise ``OSError(EADDRINUSE)``.
    - Stale → ``os.unlink`` and log a warning. A concurrent unlink that
      already removed the file is tolerated; any other ``OSError`` during
      unlink is re-raised with the path context preserved.

    Known TOCTOU races: between ``lstat`` and ``connect``, between
    ``connect`` and ``unlink``, and between ``unlink`` and the caller's
    subsequent ``bind()``, a concurrent process could change the state at
    ``path``. Callers are expected to serialize UDS-path allocation per
    orchestrator. The worst case is a noisier-than-necessary error from
    ``bind()`` and never a silent compromise of an unrelated file.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        raise FileExistsError(
            f"{path} exists and is not a socket; refusing to overwrite"
        )
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.1)
    is_live = False
    try:
        probe.connect(path)
        is_live = True
    except (ConnectionRefusedError, FileNotFoundError):
        # Nobody bound; stale or missing file.
        pass
    except TimeoutError:
        # Probe took too long; refuse rather than risk clobbering a live
        # listener whose accept queue is temporarily saturated.
        is_live = True
    except PermissionError:
        # Surface to the operator -- a chmod/chown is the real fix.
        raise
    except OSError:
        # Any other transient probe-time error gets the conservative
        # treatment too.
        is_live = True
    finally:
        probe.close()
    if is_live:
        raise OSError(
            errno.EADDRINUSE,
            f"UDS {path} is already in use by another process",
        )
    try:
        os.unlink(path)
    except FileNotFoundError:
        # Raced with another process that already cleaned up the stale file.
        return
    except OSError as e:
        raise OSError(
            e.errno,
            f"Detected stale UDS file at {path} but failed to remove it: "
            f"{e.strerror}",
        ) from e
    logger.warning("Removed stale UDS file at %s", path)


def uds_request(
    method: str,
    uds_path: str,
    path: str,
    headers: dict,
    timeout: float,
    json_data=None,
) -> _UDSResponseWrapper:
    """Send a single HTTP request over a Unix domain socket.

    Speaks plain HTTP/1.1 via stdlib ``http.client``. ``json_data`` is
    serialized and sent as the body when provided. Always closes the
    underlying socket; ``sock.close()`` is wired before connect() so a
    failed connect doesn't leak an FD.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(uds_path)
    except OSError:
        sock.close()
        raise
    conn = http.client.HTTPConnection("localhost", timeout=timeout)
    conn.sock = sock
    try:
        body = None
        request_headers = dict(headers)
        if json_data is not None:
            body = json.dumps(json_data).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=body, headers=request_headers)
        resp = conn.getresponse()
        return _UDSResponseWrapper(resp.status, resp.read())
    finally:
        conn.close()


def server_http_get(
    server_args: ServerArgs,
    path: str,
    headers: dict,
    timeout: float,
    verify,
):
    """GET dispatcher that branches on UDS vs TCP transport."""
    if server_args.uds:
        return uds_request("GET", server_args.uds, path, headers, timeout)
    return requests.get(
        server_args.url() + path,
        timeout=timeout,
        headers=headers,
        verify=verify,
    )


def server_http_post(
    server_args: ServerArgs,
    path: str,
    headers: dict,
    timeout: float,
    verify,
    json_data,
):
    """POST dispatcher that branches on UDS vs TCP transport."""
    if server_args.uds:
        return uds_request(
            "POST", server_args.uds, path, headers, timeout, json_data=json_data
        )
    return requests.post(
        server_args.url() + path,
        json=json_data,
        timeout=timeout,
        headers=headers,
        verify=verify,
    )
