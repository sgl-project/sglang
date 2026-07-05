# SPDX-License-Identifier: Apache-2.0
"""Protocol definitions for the weight cache daemon.

Defines CacheConfig for validation and socket message protocol helpers.
"""

import hashlib
import json
import logging
import os
import pickle
import struct
from typing import Any, Dict, Optional

import msgspec

from sglang.srt.utils.common import safe_pickle_loads

logger = logging.getLogger(__name__)

# Socket path template for weight cache daemons (keyed by global rank
# = tp_size * pp_rank + tp_rank, so multi-node / multi-PP don't collide)
WEIGHT_CACHE_SOCKET_TEMPLATE = "/tmp/sglang_weight_cache_rank{global_rank}.sock"

# Ready file template — daemon writes this after loading completes
WEIGHT_CACHE_READY_TEMPLATE = "/tmp/sglang_weight_cache_rank{global_rank}.ready"


class CacheConfig(msgspec.Struct):
    """Fingerprint of the cached weights. Used to validate compatibility
    between a daemon's cached state and a requesting engine process.

    Any mismatch triggers a fallback to disk loading.
    """

    model_path: str
    model_arch: str
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    ep_size: int
    quant_method: str  # e.g. "fp8", "gptq_marlin", "" for unquantized
    quant_config_hash: str  # SHA-256 hash of quantization config
    dtype: str  # e.g. "torch.float16"

    def matches(self, other: "CacheConfig") -> bool:
        """Check if two configs are compatible for weight sharing."""
        return self == other

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__struct_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheConfig":
        return cls(**d)


def hash_quant_config(quant_config: Any) -> str:
    """Compute a stable hash of the quantization config.

    Avoids str()/repr() on arbitrary objects because those embed memory
    addresses (e.g. "at 0x7f..."), producing different hashes across
    processes and causing permanent config mismatch.
    """
    if quant_config is None:
        return ""
    try:
        if hasattr(quant_config, "to_dict"):
            config_str = json.dumps(quant_config.to_dict(), sort_keys=True)
        elif isinstance(quant_config, dict):
            config_str = json.dumps(quant_config, sort_keys=True)
        elif hasattr(quant_config, "__dict__"):
            config_str = (
                type(quant_config).__name__
                + ":"
                + json.dumps(
                    {
                        k: v
                        for k, v in sorted(quant_config.__dict__.items())
                        if not k.startswith("_")
                        and isinstance(
                            v, (str, int, float, bool, type(None), list, dict)
                        )
                    },
                    sort_keys=True,
                )
            )
        else:
            config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    except Exception:
        config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_quant_method_name(quant_config: Any) -> str:
    """Extract the quantization method name from config."""
    if quant_config is None:
        return ""
    if isinstance(quant_config, str):
        return quant_config
    if hasattr(quant_config, "get_name"):
        return quant_config.get_name()
    if hasattr(quant_config, "name"):
        return quant_config.name
    return type(quant_config).__name__


# ---------------------------------------------------------------------------
# Socket protocol helpers
# ---------------------------------------------------------------------------


MAX_MSG_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB sanity cap


def send_msg(sock, obj: Any) -> None:
    """Send a length-prefixed pickled message over a socket."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_msg(sock) -> Any:
    """Receive a length-prefixed pickled message from a socket."""
    header = _recv_exact(sock, 4)
    if header is None:
        raise ConnectionError("Connection closed while reading message header")
    length = struct.unpack("!I", header)[0]
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message size {length} exceeds {MAX_MSG_SIZE} byte cap")
    data = _recv_exact(sock, length)
    if data is None:
        raise ConnectionError("Connection closed while reading message body")
    return safe_pickle_loads(data)


def _recv_exact(sock, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def get_socket_path(global_rank: int) -> str:
    """Get the Unix socket path for a weight cache daemon.

    global_rank = tp_size * pp_rank + tp_rank
    """
    return WEIGHT_CACHE_SOCKET_TEMPLATE.format(global_rank=global_rank)


def get_ready_path(global_rank: int) -> str:
    """Get the ready-file path for a weight cache daemon.

    global_rank = tp_size * pp_rank + tp_rank
    """
    return WEIGHT_CACHE_READY_TEMPLATE.format(global_rank=global_rank)


def _read_ready_pid(ready_path: str) -> Optional[int]:
    """Read the daemon PID from a .ready file. Returns None if unreadable."""
    try:
        with open(ready_path) as f:
            for line in f:
                if line.startswith("pid="):
                    return int(line.strip().split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_stale_daemon_files(global_rank: int) -> None:
    """Validate and clean up .ready/.sock files for a daemon rank.

    If the .ready file exists and the recorded PID is still alive, the daemon
    is still running — raise RuntimeError so the caller doesn't clobber it.
    If the PID is dead (or unreadable), the files are stale leftovers from a
    crashed/killed daemon and are safe to remove.
    """
    ready_path = get_ready_path(global_rank)
    socket_path = get_socket_path(global_rank)

    if not os.path.exists(ready_path) and not os.path.exists(socket_path):
        return

    pid = _read_ready_pid(ready_path) if os.path.exists(ready_path) else None

    if pid is not None and _is_pid_alive(pid):
        raise RuntimeError(
            f"Weight cache daemon for rank {global_rank} is already running "
            f"(pid={pid}, ready={ready_path}). Stop the existing daemon before "
            f"launching a new one."
        )

    for path in (ready_path, socket_path):
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Removed stale daemon file: {path}")
