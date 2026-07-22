# SPDX-License-Identifier: Apache-2.0
"""Protocol definitions for the weight cache daemon.

Defines CacheConfig for validation and socket message protocol helpers.
"""

import hashlib
import json
import logging
import os
import pickle
import signal
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
    revision: str  # model revision the weights were loaded from ("" if unset)
    # Environment stamp: a daemon and a client that ran different post-processing
    # branches (different GPU compute capability or torch/kernel version) can
    # produce incompatible weights that would map cleanly yet serve garbage.
    # Comparing these turns that into a clean mismatch. See compute_env_stamp().
    device_capability: str  # local compute capability, e.g. "8.0" ("" if N/A)
    torch_version: str  # torch.__version__ of the process that built the weights

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
        return hashlib.sha256(config_str.encode()).hexdigest()
    except Exception:
        config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()


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
# IPC quantization-method allowlist
# ---------------------------------------------------------------------------
#
# CUDA IPC zero-copy sharing exports ONLY raw tensor data (state_dict + buffers).
# It is correct only when process_weights_after_loading's entire effect is
# captured by that tensor data. Several quant methods break this assumption:
#   - They stamp Python-side metadata on tensors that does NOT cross IPC
#     (e.g. block-FP8 sets `weight_scale_inv.format_ue8m0 = True`). The
#     meta-initialized client never runs post-processing, so the flag is
#     absent and the apply path silently selects the wrong kernel.
#   - They repack / transpose weights into shapes the meta-init client cannot
#     reproduce from create_weights alone (per-tensor FP8, Marlin, AWQ/GPTQ).
#
# Serving such weights over IPC yields silently-wrong numerics. Only methods
# verified to round-trip through pure tensor export are allowed here; every
# other method must hard-error. Extend the registry below only after a method
# has been verified end-to-end.


class UnsupportedQuantForIPCError(RuntimeError):
    """Raised when a quantization method is not on the verified allowlist for
    CUDA IPC zero-copy weight sharing."""


def _get_quant_field(quant_config: Any, key: str) -> Any:
    """Read a field from a quant config that may be a dict or an object."""
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config.get(key)
    return getattr(quant_config, key, None)


def _fp8_round_trips_via_ipc(quant_config: Any) -> bool:
    """Only block-wise FP8 is verified.

    Block-wise FP8 (weight_block_size set) preserves weight shape and the only
    post-load metadata it stamps is accounted for. Per-tensor FP8 transposes
    `layer.weight` during post-processing, a shape change the meta-init client
    cannot reproduce, so it is not supported.
    """
    return _get_quant_field(quant_config, "weight_block_size") is not None


# quant_method name -> predicate(quant_config) -> bool (True == verified safe).
# A method absent from this registry is unsupported and hard-errors.
IPC_QUANT_ALLOWLIST = {
    "": lambda _quant_config: True,  # unquantized
    "fp8": _fp8_round_trips_via_ipc,  # only block-wise FP8 verified
}


def is_ipc_quant_supported(quant_method: str, quant_config: Any) -> bool:
    """Return True if `quant_method` is verified safe for IPC zero-copy sharing."""
    predicate = IPC_QUANT_ALLOWLIST.get(quant_method)
    if predicate is None:
        return False
    try:
        return bool(predicate(quant_config))
    except Exception:
        return False


def check_ipc_quant_support(
    quant_method: str, quant_config: Any, *, where: str
) -> None:
    """Hard-error unless `quant_method` is verified safe for IPC zero-copy sharing.

    `where` is a short tag (e.g. "daemon"/"client") used only in the error
    message. Raises UnsupportedQuantForIPCError with an actionable message.
    """
    if is_ipc_quant_supported(quant_method, quant_config):
        return
    verified = ", ".join(
        (repr(m) if m else "'' (unquantized)") for m in IPC_QUANT_ALLOWLIST
    )
    raise UnsupportedQuantForIPCError(
        f"[weight_cache:{where}] quantization method {quant_method!r} is not "
        f"verified for CUDA IPC zero-copy weight sharing. Its "
        f"process_weights_after_loading may stamp Python-side metadata "
        f"(e.g. format_ue8m0) or repack/transpose weights into shapes the "
        f"meta-initialized client cannot reproduce, which would silently serve "
        f"wrong-numerics weights. Verified methods: {verified}. Note: FP8 is "
        f"only verified for block-wise configs (weight_block_size set), not "
        f"per-tensor FP8. Disable the weight cache (--weight-cache-mode off) "
        f"for this model."
    )


# ---------------------------------------------------------------------------
# Socket protocol helpers
# ---------------------------------------------------------------------------


# Sanity cap on a single socket message. IPC messages carry only metadata —
# base64 CUDA IPC handles plus per-tensor shape/dtype — never the weight bytes
# themselves (those stay in GPU memory and are shared zero-copy). Even a model
# with hundreds of thousands of tensors serializes to a few MiB here, so a
# generous 64 MiB cap still rejects a corrupt/hostile length prefix long before
# it can drive a multi-GiB allocation in recv_msg.
MAX_MSG_SIZE = 64 * 1024 * 1024  # 64 MiB


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


def compute_env_stamp() -> Dict[str, str]:
    """Local environment fingerprint for the IPC weight cache.

    Returns the device compute capability and torch version of the current
    process. A daemon and a connecting client that differ on either may have run
    different post-processing / kernel-selection branches, producing weights that
    map cleanly over IPC yet serve garbage; stamping these into CacheConfig turns
    that into a clean mismatch. Imported lazily so protocol.py stays cheap to
    import and usable on CPU-only hosts (both fields degrade to "").
    """
    device_capability = ""
    torch_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        pass
    try:
        from sglang.srt.platforms import current_platform

        cap = current_platform.get_device_capability()
        if cap is not None:
            device_capability = f"{cap.major}.{cap.minor}"
    except Exception:
        pass
    return {"device_capability": device_capability, "torch_version": torch_version}


def compute_global_rank(tp_size: int, pp_rank: int, tp_rank: int) -> int:
    """Single source of truth for the daemon rank formula.

    global_rank = tp_size * pp_rank + tp_rank, so each daemon gets a unique
    socket/ready path even across PP stages and nodes. Every call site (engine,
    loader, model_runner, daemon) must go through this so the copies can't drift.
    """
    return tp_size * pp_rank + tp_rank


def compute_local_gpu_id(
    pp_rank: int,
    tp_rank: int,
    pp_size_per_node: int,
    tp_size_per_node: int,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
) -> int:
    """Single source of truth for the local GPU id a daemon rank runs on.

    Mirrors the engine's device assignment so a daemon and the engine rank it
    serves always land on the same physical GPU (a prerequisite for CUDA IPC).
    ``base_gpu_id``/``gpu_id_step`` default to the identity mapping used by the
    standalone launcher; the engine passes its real ``--base-gpu-id`` /
    ``--gpu-id-step`` so every call site computes the id the same way instead of
    keeping three drifting copies of the formula.
    """
    return (
        base_gpu_id
        + (pp_rank % pp_size_per_node) * tp_size_per_node
        + (tp_rank % tp_size_per_node) * gpu_id_step
    )


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


def cleanup_stale_daemon_files(global_rank: int, *, force: bool = False) -> None:
    """Validate and clean up .ready/.sock files for a daemon rank.

    If the .ready file exists and the recorded PID is still alive, the daemon
    is still running — raise RuntimeError so the caller doesn't clobber it,
    unless ``force`` is set, in which case the running daemon is killed and its
    files are taken over (stale-takeover path for a wedged/orphaned daemon).
    If the PID is dead (or unreadable), the files are stale leftovers from a
    crashed/killed daemon and are safe to remove.
    """
    ready_path = get_ready_path(global_rank)
    socket_path = get_socket_path(global_rank)

    if not os.path.exists(ready_path) and not os.path.exists(socket_path):
        return

    pid = _read_ready_pid(ready_path) if os.path.exists(ready_path) else None

    if pid is not None and _is_pid_alive(pid):
        if not force:
            raise RuntimeError(
                f"Weight cache daemon for rank {global_rank} is already running "
                f"(pid={pid}, ready={ready_path}). Stop the existing daemon before "
                f"launching a new one, or pass force=True (--force) to kill it and "
                f"take over."
            )
        logger.warning(
            f"[weight_cache] force takeover: killing existing daemon pid={pid} "
            f"for rank {global_rank} and reclaiming its socket/ready files."
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for path in (ready_path, socket_path):
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Removed stale daemon file: {path}")
