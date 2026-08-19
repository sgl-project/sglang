# SPDX-License-Identifier: Apache-2.0
"""Connector that streams safetensors weights over HTTP GET Range requests.

Motivation: on a host with several NICs, remote weights can be read faster than
local disk, but only if (a) many connections are in flight and (b) each TP rank
drives its own NIC. Neither is reachable through a POSIX/FUSE mount, because the
kernel picks the egress NIC from the routing table and all ranks end up sharing
one link. Fetching objects over HTTP moves that decision into user space: every
rank opens its own sockets and can pin them to a NIC with ``SO_BINDTODEVICE``.

URL format::

    http-range://<host>:<port>/<prefix>[?<option>=<value>&...]

The authority is the default endpoint, used for metadata objects and for any
rank without a dedicated endpoint. Supported options:

``endpoints``
    Comma separated ``host[:port]`` list; TP rank ``i`` uses entry
    ``i % len(endpoints)``. IPv6 literals must be bracketed (``[::1]:9000``).
``nics``
    Comma separated interface list bound with ``SO_BINDTODEVICE`` (Linux only),
    indexed by TP rank the same way. An empty entry leaves the egress NIC to the
    routing table.
``connections``
    Parallel connections per object (default 16). Object stores usually cap a
    single connection well below link speed.
``chunk_size``
    Range size in bytes (default 32 MiB). Keep it large: request rate, not
    bandwidth, is what object stores throttle first.
``timeout``, ``retries``
    Socket timeout in seconds (default 30) and retries per range (default 2).
``metadata``
    Extra metadata object names to fetch alongside the well-known set, for
    repositories that ship custom code or tokenizer files.

Server requirements: HTTP/1.1 with byte ranges and keep-alive. A server that
closes the connection after every response, or that ignores ``Range`` and
returns the whole object, is rejected with an explicit error. To make the
*return* path use the intended NIC the server must be split per NIC as well,
with ``SO_BINDTODEVICE`` on its listening socket -- otherwise only the client
side is spread out and the routing table still funnels every rank onto one link.

Example::

    python3 -m sglang.launch_server --tp 4 --load-format remote --model-path \\
      'http-range://[fd00::1]:9000/models/Qwen3-30B-A3B?endpoints=[fd00::1]:9000,[fd00::2]:9001,[fd00::3]:9002,[fd00::4]:9003&nics=bond0,bond1,bond2,bond3'
"""

import fnmatch
import itertools
import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlparse

import numpy as np
import torch

from sglang.srt.connector import HTTP_RANGE_SCHEME, BaseFileConnector

logger = logging.getLogger(__name__)

SCHEME = HTTP_RANGE_SCHEME

_DEFAULT_PORT = 80
_DEFAULT_CONNECTIONS = 16
_DEFAULT_CHUNK_SIZE = 32 << 20
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_RETRIES = 2

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SINGLE_SHARD = "model.safetensors"

# Metadata objects fetched by pull_files(). A plain Range server cannot be
# listed, so the well-known HuggingFace metadata set is probed by name and
# missing objects are skipped. Use the ``metadata`` URL option to add more.
_METADATA_OBJECTS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "chat_template.jinja",
    _SAFETENSORS_INDEX,
)

# safetensors dtype tag -> torch dtype attribute name.
_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "I64": "int64",
    "U64": "uint64",
    "F64": "float64",
}

Endpoint = Tuple[str, int, Optional[str]]


def parse_endpoint(spec: str, default_port: int) -> Tuple[str, int]:
    """Split ``host``, ``host:port`` or ``[v6addr]:port`` into host and port."""
    spec = spec.strip()
    if spec.startswith("["):
        host, sep, rest = spec[1:].partition("]")
        if not sep:
            raise ValueError(f"unbalanced brackets in endpoint {spec!r}")
        port = rest[1:] if rest.startswith(":") else ""
    elif spec.count(":") == 1:
        host, _, port = spec.partition(":")
    else:
        # Bare hostname, IPv4, or unbracketed IPv6 literal.
        host, port = spec, ""
    if not host:
        raise ValueError(f"missing host in endpoint {spec!r}")
    return host, int(port) if port else default_port


def _torch_dtype(tag: str) -> torch.dtype:
    name = _SAFETENSORS_DTYPES.get(tag)
    dtype = getattr(torch, name) if name else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"safetensors dtype {tag!r} is not supported by this torch build "
            f"({torch.__version__})"
        )
    return dtype


def iter_safetensors(
    buf: np.ndarray,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` from an in-memory safetensors image.

    Tensors are ``torch.frombuffer`` views onto ``buf``, so the caller must keep
    it alive while they are in use. ``safetensors.torch.load()`` would copy the
    whole image once more, which is several GB per shard at these sizes.
    """
    view = memoryview(buf.data).cast("B")
    total = len(view)
    if total < 8:
        raise ValueError(f"safetensors image is {total} bytes, too short for a header")
    header_len = int.from_bytes(bytes(view[:8]), "little")
    data_start = 8 + header_len
    if data_start > total:
        raise ValueError(
            f"safetensors header claims {header_len} bytes but the image is {total}"
        )
    header = json.loads(bytes(view[8:data_start]))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if data_start + end > total:
            raise ValueError(
                f"tensor {name!r} ends at {data_start + end} past the {total}-byte image"
            )
        dtype = _torch_dtype(meta["dtype"])
        if end == start:
            tensor = torch.empty(0, dtype=dtype)
        else:
            chunk = view[data_start + start : data_start + end]
            itemsize = torch.empty(0, dtype=dtype).element_size()
            if (data_start + start) % itemsize:
                # torch.frombuffer requires an aligned address; safetensors does
                # not guarantee one for every dtype, so copy in that rare case.
                chunk = bytearray(chunk)
            tensor = torch.frombuffer(chunk, dtype=dtype)
        yield name, tensor.reshape(meta["shape"])


class HttpRangeConnector(BaseFileConnector):
    """Streams weights straight out of an object store, one shard at a time.

    Peak host memory is one shard: the buffer is released as soon as the
    consumer has walked its tensors. Weight bytes never touch local storage --
    only metadata objects are materialized, under :meth:`get_local_dir`.

    Every rank pulls all shards, because ``load_weights`` narrows per rank on
    the consumer side; total network volume is therefore ``tp_size`` times the
    checkpoint size, which is what makes per-rank NIC pinning worthwhile here.
    """

    def __init__(self, url: str) -> None:
        super().__init__(url)
        parsed = urlparse(url)
        if parsed.scheme != SCHEME:
            raise ValueError(f"expected a {SCHEME}:// url, got {url!r}")
        if not parsed.hostname:
            raise ValueError(f"missing host in {url!r}")
        options = dict(parse_qsl(parsed.query, keep_blank_values=True))
        unknown = set(options) - {
            "endpoints",
            "nics",
            "connections",
            "chunk_size",
            "timeout",
            "retries",
            "metadata",
        }
        if unknown:
            raise ValueError(f"unknown {SCHEME} options: {sorted(unknown)}")

        self.default_endpoint = (parsed.hostname, parsed.port or _DEFAULT_PORT)
        self.prefix = parsed.path.rstrip("/")
        self.endpoints = [
            parse_endpoint(entry, self.default_endpoint[1])
            for entry in options.get("endpoints", "").split(",")
            if entry.strip()
        ] or [self.default_endpoint]
        self.nics = [
            entry.strip() or None for entry in options.get("nics", "").split(",")
        ]
        self.connections = int(options.get("connections", _DEFAULT_CONNECTIONS))
        self.chunk_size = int(options.get("chunk_size", _DEFAULT_CHUNK_SIZE))
        self.timeout = float(options.get("timeout", _DEFAULT_TIMEOUT))
        self.retries = int(options.get("retries", _DEFAULT_RETRIES))
        if self.connections < 1 or self.chunk_size < 1 or self.retries < 0:
            raise ValueError(
                "connections and chunk_size must be positive and retries non-negative"
            )
        self.metadata_objects = list(_METADATA_OBJECTS) + [
            entry.strip()
            for entry in options.get("metadata", "").split(",")
            if entry.strip()
        ]
        self._shards: Optional[list[str]] = None

    # -- endpoints and sockets ----------------------------------------------

    def endpoint_for_rank(self, rank: int) -> Endpoint:
        host, port = self.endpoints[rank % len(self.endpoints)]
        return host, port, self.nics[rank % len(self.nics)]

    def _bind_to_device(self, sock: socket.socket, nic: str) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                f"the nics option needs SO_BINDTODEVICE, unavailable on {sys.platform}"
            )
        option = getattr(socket, "SO_BINDTODEVICE", 25)
        try:
            sock.setsockopt(socket.SOL_SOCKET, option, nic.encode() + b"\0")
        except PermissionError as e:
            raise PermissionError(
                f"binding a socket to {nic!r} requires CAP_NET_RAW"
            ) from e

    def _connect(self, endpoint: Endpoint) -> socket.socket:
        host, port, nic = endpoint
        last_error: Optional[OSError] = None
        for family, kind, proto, _, address in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        ):
            sock = socket.socket(family, kind, proto)
            try:
                if nic:
                    self._bind_to_device(sock, nic)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.timeout)
                sock.connect(address)
                return sock
            except OSError as e:
                sock.close()
                last_error = e
        raise last_error or OSError(f"cannot resolve {host}:{port}")

    # -- HTTP ---------------------------------------------------------------

    def _object_path(self, name: str) -> str:
        if any(c in name for c in "\r\n") or not name:
            raise ValueError(f"illegal object name {name!r}")
        return quote(f"{self.prefix}/{name}")

    def _send_request(
        self, sock: socket.socket, name: str, byte_range: Optional[Tuple[int, int]]
    ) -> None:
        lines = [
            f"GET {self._object_path(name)} HTTP/1.1",
            f"Host: {self.default_endpoint[0]}",
            "Connection: keep-alive",
        ]
        if byte_range is not None:
            offset, length = byte_range
            lines.append(f"Range: bytes={offset}-{offset + length - 1}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())

    def _read_response_head(
        self, sock: socket.socket, pending: bytes
    ) -> Tuple[int, int, bytes, dict]:
        """Return ``(status, content_length, leftover_body_bytes, headers)``.

        ``pending`` carries bytes already read past the previous response on a
        keep-alive connection.
        """
        buf = pending
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError(
                    "connection closed while reading response headers"
                )
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        fields = lines[0].split(None, 2)
        if len(fields) < 2 or not fields[1].isdigit():
            raise ConnectionError(f"malformed HTTP status line: {lines[0]!r}")
        headers = {}
        for line in lines[1:]:
            key, sep, value = line.partition(b":")
            if sep:
                headers[key.strip().lower()] = value.strip()
        if b"transfer-encoding" in headers:
            raise ConnectionError(
                f"{SCHEME} needs Content-Length responses, got "
                f"Transfer-Encoding: {headers[b'transfer-encoding']!r}"
            )
        if headers.get(b"connection", b"").lower() == b"close":
            raise ConnectionError(
                f"{SCHEME} needs HTTP/1.1 keep-alive, server asked to close"
            )
        if b"content-length" not in headers:
            raise ConnectionError("HTTP response without Content-Length")
        return int(fields[1]), int(headers[b"content-length"]), rest, headers

    def _recv_body(
        self,
        sock: socket.socket,
        destination: memoryview,
        expected: int,
        pending: bytes,
    ) -> bytes:
        got = 0
        if pending:
            take = min(len(pending), expected)
            destination[:take] = pending[:take]
            got = take
            pending = pending[take:]
        while got < expected:
            read = sock.recv_into(destination[got:])
            if read == 0:
                raise ConnectionError(
                    f"connection closed after {got} of {expected} body bytes"
                )
            got += read
        return pending

    def _get_object(self, name: str) -> Optional[bytes]:
        """Fetch a whole (small) object, or None when the server reports 404."""
        sock = self._connect((*self.default_endpoint, None))
        try:
            self._send_request(sock, name, None)
            status, length, pending, _ = self._read_response_head(sock, b"")
            if status == 404:
                return None
            if status != 200:
                raise ConnectionError(f"GET {name} returned HTTP {status}")
            body = bytearray(length)
            self._recv_body(sock, memoryview(body), length, pending)
            return bytes(body)
        finally:
            sock.close()

    def _object_size(self, endpoint: Endpoint, name: str) -> int:
        """Probe an object's size, and confirm the server honours Range."""
        sock = self._connect(endpoint)
        try:
            self._send_request(sock, name, (0, 1))
            status, length, _, headers = self._read_response_head(sock, b"")
            if status == 404:
                raise FileNotFoundError(f"object {name!r} not found at {endpoint}")
            if status == 200:
                raise ConnectionError(
                    f"server ignored the Range header for {name!r} and returned the "
                    f"whole object; {SCHEME} requires byte-range support"
                )
            if status != 206 or length != 1:
                raise ConnectionError(
                    f"unexpected range response for {name!r}: HTTP {status}, "
                    f"{length} bytes"
                )
            # "Content-Range: bytes 0-0/<total>" is the only place a range
            # response states the object size.
            content_range = headers.get(b"content-range", b"")
            total = content_range.rpartition(b"/")[2]
            if not total.isdigit():
                raise ConnectionError(
                    f"range response for {name!r} has no usable Content-Range: "
                    f"{content_range!r}"
                )
            return int(total)
        finally:
            sock.close()

    def _fetch_object(self, endpoint: Endpoint, name: str, size: int) -> np.ndarray:
        """Pull one object into a uint8 array with parallel keep-alive readers.

        Ranges are handed out from a shared cursor rather than split per worker,
        so every connection stays busy even though shards differ in size.
        """
        out = np.empty(size, dtype=np.uint8)
        view = memoryview(out.data).cast("B")
        ranges = [
            (offset, min(self.chunk_size, size - offset))
            for offset in range(0, size, self.chunk_size)
        ]
        cursor = itertools.count()
        stop = threading.Event()
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            sock = None
            pending = b""
            try:
                for index in cursor:
                    if index >= len(ranges) or stop.is_set():
                        return
                    offset, length = ranges[index]
                    for attempt in range(self.retries + 1):
                        try:
                            if sock is None:
                                sock = self._connect(endpoint)
                                pending = b""
                            self._send_request(sock, name, (offset, length))
                            status, expected, pending, _ = self._read_response_head(
                                sock, pending
                            )
                            if status != 206 or expected != length:
                                raise ConnectionError(
                                    f"range {offset}+{length} of {name!r} answered "
                                    f"with HTTP {status} and {expected} bytes"
                                )
                            pending = self._recv_body(
                                sock, view[offset : offset + length], length, pending
                            )
                            break
                        except OSError as e:
                            if sock is not None:
                                sock.close()
                                sock = None
                            if attempt == self.retries:
                                raise
                            logger.warning(
                                "retrying range %d+%d of %s after %s",
                                offset,
                                length,
                                name,
                                e,
                            )
                            time.sleep(0.1 * (attempt + 1))
            except BaseException as e:  # noqa: BLE001 - re-raised by the caller
                stop.set()
                with lock:
                    errors.append(e)
            finally:
                if sock is not None:
                    sock.close()

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(max(1, min(self.connections, len(ranges))))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return out

    # -- BaseFileConnector --------------------------------------------------

    def shard_names(self) -> list[str]:
        if self._shards is not None:
            return self._shards
        index = self._get_object(_SAFETENSORS_INDEX)
        if index is not None:
            weight_map = json.loads(index)["weight_map"]
            self._shards = sorted(set(weight_map.values()))
        elif self._get_object(_SINGLE_SHARD) is not None:
            self._shards = [_SINGLE_SHARD]
        else:
            raise FileNotFoundError(
                f"neither {_SAFETENSORS_INDEX} nor {_SINGLE_SHARD} found under "
                f"{self.prefix!r}; only safetensors checkpoints are supported"
            )
        return self._shards

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        host, port = self.default_endpoint
        names = self.shard_names()
        if allow_pattern is not None:
            names = [
                name
                for name in names
                if any(fnmatch.fnmatch(name, pattern) for pattern in allow_pattern)
            ]
        return [f"{SCHEME}://{host}:{port}{self.prefix}/{name}" for name in names]

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        """Materialize metadata objects into the connector's local directory.

        Weight shards are never pulled: they are streamed by
        :meth:`weight_iterator`. Objects the store does not have are skipped, so
        the well-known metadata set can be probed without listing support.
        """
        for name in self.metadata_objects:
            if allow_pattern is not None and not any(
                fnmatch.fnmatch(name, pattern) for pattern in allow_pattern
            ):
                continue
            if ignore_pattern is not None and any(
                fnmatch.fnmatch(name, pattern) for pattern in ignore_pattern
            ):
                continue
            body = self._get_object(name)
            if body is None:
                continue
            destination = os.path.join(self.local_dir, name)
            os.makedirs(Path(destination).parent, exist_ok=True)
            with open(destination, "wb") as f:
                f.write(body)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        endpoint = self.endpoint_for_rank(rank)
        shards = self.shard_names()
        logger.info(
            "Streaming %d shards from %s over %d connections (rank %d, nic %s)",
            len(shards),
            f"[{endpoint[0]}]:{endpoint[1]}{self.prefix}",
            self.connections,
            rank,
            endpoint[2] or "routing table",
        )
        fetched = 0
        seconds = 0.0
        for name in shards:
            start = time.perf_counter()
            size = self._object_size(endpoint, name)
            buf = self._fetch_object(endpoint, name, size)
            seconds += time.perf_counter() - start
            fetched += size
            yield from iter_safetensors(buf)
            del buf
        logger.info(
            "Streamed %.2f GB in %.2f s (%.2f GB/s) for rank %d on nic %s",
            fetched / 1e9,
            seconds,
            fetched / max(seconds, 1e-9) / 1e9,
            rank,
            endpoint[2] or "routing table",
        )
