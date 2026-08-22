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


def normalize_model_path_for_cache(model_path: str) -> str:
    """Canonicalize local paths while preserving model IDs and remote URIs."""
    return os.path.realpath(model_path) if os.path.exists(model_path) else model_path


class CacheConfig(msgspec.Struct):
    """Fingerprint of the cached weights. Used to validate compatibility
    between a daemon's cached state and a requesting engine process.

    A mismatch is never a cache hit. If the mismatched daemon already occupies
    the requested GPU, the client must fail rather than double-allocate weights.
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
    resolved_revision: str  # immutable HF commit hash when available
    # Environment stamp: a daemon and a client that ran different post-processing
    # branches (different GPU compute capability or torch/kernel version) can
    # produce incompatible weights that would map cleanly yet serve garbage.
    # Comparing these turns that into a clean mismatch. See compute_env_stamp().
    device_capability: str  # local compute capability, e.g. "8.0" ("" if N/A)
    torch_version: str  # torch.__version__ of the process that built the weights
    load_format: str  # disk/source loader whose output is cached
    model_loader_extra_config_hash: str  # full SHA-256 of canonical loader options
    trust_remote_code: bool  # remote model code can change loaded tensor layout

    def matches(self, other: "CacheConfig") -> bool:
        """Check if two configs are compatible for weight sharing."""
        return self == other

    def fingerprint(self) -> str:
        """Return the stable full SHA-256 identity of this cached shard config."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

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
                    },
                    sort_keys=True,
                )
            )
        else:
            raise TypeError(
                f"unsupported quantization config type: {type(quant_config).__name__}"
            )
        return hashlib.sha256(config_str.encode()).hexdigest()
    except Exception as exc:
        raise ValueError(
            "quantization config cannot be serialized canonically; refusing "
            "to create an unsafe weight-cache fingerprint"
        ) from exc


def hash_loader_extra_config(config: Any) -> str:
    """Hash loader options canonically, failing closed on unsupported values."""
    try:
        value = json.loads(config) if isinstance(config, str) else config
        encoded = json.dumps(value or {}, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "model loader extra config cannot be serialized canonically"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def get_resolved_model_revision(model_config: Any) -> str:
    """Return Transformers' resolved immutable model commit when available."""
    hf_config = getattr(model_config, "hf_config", None)
    return str(getattr(hf_config, "_commit_hash", "") or "")


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


def check_ipc_parallelism(dp_size: int, ep_size: int, *, where: str) -> None:
    """Reject DP/EP layouts until their cache identity semantics are defined."""
    if dp_size == 1 and ep_size == 1:
        return
    raise ValueError(
        f"[weight_cache:{where}] data/expert parallelism is not supported yet "
        f"(dp_size={dp_size}, ep_size={ep_size}). CacheConfig does not yet "
        "distinguish content-partitioning ranks from requester-only ranks, so "
        "enabling the cache here could select the wrong shard or defeat "
        "per-GPU sharing. Disable the weight cache for this configuration."
    )


# ---------------------------------------------------------------------------
# IPC quantization-method allowlist
# ---------------------------------------------------------------------------
#
# CUDA IPC zero-copy sharing exports ONLY raw tensor data, so it is correct only
# when process_weights_after_loading's entire effect is captured by that data.
# Methods that stamp Python-side metadata (e.g. block-FP8's format_ue8m0) or
# repack/transpose weights into shapes the meta-init client can't reproduce
# (per-tensor FP8, Marlin, AWQ/GPTQ) would serve silently-wrong numerics. Only
# methods verified to round-trip through pure tensor export are allowed; every
# other method hard-errors. Extend the registry below only after verifying a
# method end-to-end.


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
    return bool(predicate(quant_config))


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


MAX_MSG_SIZE = 256 * 1024 * 1024  # 256 MiB


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


def compute_env_stamp(device_id: int = 0) -> Dict[str, str]:
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

        cap = current_platform.get_device_capability(device_id)
        if cap is not None:
            device_capability = f"{cap.major}.{cap.minor}"
    except Exception:
        pass
    return {"device_capability": device_capability, "torch_version": torch_version}


def compute_global_rank(tp_size: int, pp_rank: int, tp_rank: int) -> int:
    """Return a rank inside one PP×TP distributed job.

    This rank is useful for distributed coordination and logging, but it is not
    a persistent cache or physical-device identity across independent jobs.
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
