# SPDX-License-Identifier: Apache-2.0
"""Protocol definitions for the weight cache daemon.

Defines CacheConfig for validation and socket message protocol helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import signal
import struct
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Type,
)

import msgspec

from sglang.srt.utils.common import safe_pickle_loads

if TYPE_CHECKING:
    import torch.nn as nn

    from sglang.srt.server_args import ServerArgs

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
    # Quantization backends
    fp4_gemm_backend: str
    fp8_gemm_backend: str
    moe_runner_backend: str

    def matches(self, other: CacheConfig) -> bool:
        """Check if two configs are compatible for weight sharing."""
        return self == other

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__struct_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CacheConfig:
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


def _nvfp4_round_trips_via_ipc(quant_config: Any) -> bool:
    """Only plain serialized NVFP4 is verified."""
    quant_algo = _get_quant_field(quant_config, "quant_algo")
    if quant_algo is None:
        quantization_section = _get_quant_field(quant_config, "quantization")
        if isinstance(quantization_section, dict):
            quant_algo = quantization_section.get("quant_algo", None)
    return quant_algo == "NVFP4"


_TRANSFERABLE_ATTR_TYPES = (int, float, bool, str, type(None))


def is_ipc_quant_supported(quant_method: str, quant_config: Any) -> bool:
    """Whether this method + config is verified for IPC zero-copy sharing."""
    registered = WeightCacheQuantStates._registry.get(quant_method)
    return registered is not None and bool(registered.is_supported(quant_config))


def check_ipc_quant_support(
    quant_method: str, quant_config: Any, *, where: str
) -> None:
    """Hard-error unless `quant_method` + `quant_config` is verified for IPC.

    Separate from constructing WeightCacheQuantStates because callers need to
    gate *before* they know a daemon exists -- and before they have the
    published server args that __init__ requires.

    `where` is a short tag ("daemon" / "client") used only in the message.
    """
    if is_ipc_quant_supported(quant_method, quant_config):
        return

    verified = ", ".join(
        (repr(m) if m else "'' (unquantized)") for m in WeightCacheQuantStates._registry
    )
    raise UnsupportedQuantForIPCError(
        f"[weight_cache:{where}] quantization method {quant_method!r} is not "
        f"verified for CUDA IPC zero-copy weight sharing. Its "
        f"process_weights_after_loading may stamp Python-side metadata "
        f"(e.g. format_ue8m0) or repack/transpose weights into shapes the "
        f"meta-initialized client cannot reproduce, which would silently serve "
        f"wrong-numerics weights. Verified methods: {verified}. Note: FP8 is "
        f"only verified for block-wise configs (weight_block_size set), not "
        f"per-tensor FP8; NVFP4 only for quant_algo=NVFP4, not NVFP4_AWQ. "
        f"Disable the weight cache (--weight-cache-mode off) for this model."
    )


class _RegisteredQuant(msgspec.Struct, frozen=True):
    """What varies per quant method.

    Held per registry entry rather than on the states class, because several
    methods may share one class (unquantized and block-FP8 both need nothing
    beyond the tensors) and class attributes would let the last registration
    silently overwrite the earlier one's predicate. frozen so an entry cannot be
    mutated after registration.

    Process-local, unlike CacheConfig above: the fields hold a class object and
    a callable, so this never crosses the socket.
    """

    states_cls: Type[WeightCacheQuantStates]
    is_supported: Callable[[Any], bool]
    capture_attrs: frozenset


class WeightCacheQuantStates:
    """Per-quant-method adaptation for CUDA IPC weight sharing.

    The daemon runs process_weights_after_loading and exports the resulting
    tensors; a client maps them and never runs post-processing itself. Whatever
    that pass leaves *outside* the tensors has to cross explicitly, and what
    that is depends on the quant method -- so each method registers a subclass
    here and constructing the base dispatches to it.
    """

    quant_config: Any
    quant_method: str
    module_attrs: Dict[str, Dict[str, Any]]

    # quant method name -> its registration; populated by register().
    _registry: Dict[str, _RegisteredQuant] = {}
    _capture_attrs: frozenset = frozenset()

    @classmethod
    def register(
        cls,
        quant_method: str,
        *,
        is_supported_func: Callable[[Any], bool],
        capture_attrs: Optional[Set[str]] = None,
    ):
        def decorator(quant_cls):
            cls._registry[quant_method] = _RegisteredQuant(
                states_cls=quant_cls,
                is_supported=is_supported_func,
                capture_attrs=frozenset(capture_attrs or ()),
            )
            return quant_cls

        return decorator

    def __new__(
        cls,
        quant_config: Any,
        quant_method: str,
        server_args: ServerArgs,
        *,
        where: str,
    ) -> WeightCacheQuantStates:
        assert cls is WeightCacheQuantStates
        check_ipc_quant_support(quant_method, quant_config, where=where)

        return object.__new__(cls._registry[quant_method].states_cls)

    def __init__(
        self,
        quant_config: Any,
        quant_method: str,
        server_args: ServerArgs,
        *,
        where: str,
    ) -> None:
        self.quant_config = quant_config
        self.quant_method = quant_method
        self.module_attrs = {}
        self._capture_attrs = self._registry[quant_method].capture_attrs

        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config(server_args)

        logger.info(
            f"[weight_cache:{where}] {type(self).__name__} for {quant_method!r}; "
            f"attrs to transfer: {sorted(self._capture_attrs) or 'none'}"
        )

    def capture_module_attrs(self, model: nn.Module) -> None:
        """Snapshot this method's layout attributes off the post-processed model.

        Matched by attribute name alone: the names are method-specific (only an
        NVFP4 linear layer has weights_padding_cols), the daemon holds exactly
        one states object for the model, and the value-type guard rejects
        anything that could not cross the wire anyway.
        """
        if not self._capture_attrs:
            return

        for name, module in model.named_modules():
            captured = {
                key: value
                for key, value in module.__dict__.items()
                if key in self._capture_attrs
                and isinstance(value, _TRANSFERABLE_ATTR_TYPES)
            }
            if captured:
                self.module_attrs[name] = captured

    def apply_module_attrs(
        self, model: nn.Module, module_attrs: Dict[str, Dict[str, Any]]
    ) -> int:
        """Stamp the daemon's captured attributes onto this process's modules.

        The client-side half of capture_module_attrs. Returns how many values
        actually changed. Raises if the daemon captured a module this process
        does not have: the two built structurally different models, and quietly
        skipping the stamp would leave the client reading its own
        pre-post-process layout values.
        """
        modules = dict(model.named_modules())
        missing = [name for name in module_attrs if name not in modules]
        if missing:
            raise RuntimeError(
                f"[weight_cache] The daemon captured attributes for "
                f"{len(missing)} module(s) that do not exist on this client, so "
                f"the two processes built structurally different models for "
                f"{self.quant_method!r}: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        changed = 0
        for module_name, captured in module_attrs.items():
            module = modules[module_name]
            for key, value in captured.items():
                previous = getattr(module, key, _ATTR_UNSET)
                # Only compare like with like: an absent attribute (sentinel) or
                # a client-side tensor under the same name must count as changed
                # rather than go through a tensor `!=`, which returns a tensor.
                if (
                    not isinstance(previous, _TRANSFERABLE_ATTR_TYPES)
                    or previous != value
                ):
                    changed += 1
                setattr(module, key, value)
        return changed

    def ipc_reshapes_weights(self) -> bool:
        """Whether post-processing changes the shape of exported tensors."""
        raise NotImplementedError

    def ipc_rebind_after_import(self, layer: nn.Module) -> None:
        """Re-establish state that depends on tensor identity, per module."""
        raise NotImplementedError

    @classmethod
    def compute_quant_stamp(cls) -> Dict[str, str]:
        from sglang.srt.layers.moe import get_moe_runner_backend

        moe_runner_backend = str(get_moe_runner_backend().value)

        return {
            "fp4_gemm_backend": "",
            "fp8_gemm_backend": "",
            "moe_runner_backend": moe_runner_backend,
        }

    @classmethod
    def quant_stamp_for(cls, quant_method: str) -> Dict[str, str]:
        """compute_quant_stamp() of the states class registered for a method.

        Lets a caller stamp without constructing an instance (which needs
        published server args); the per-class override still applies.
        """
        return cls._registry[quant_method].states_cls.compute_quant_stamp()


@WeightCacheQuantStates.register("", is_supported_func=lambda _quant_config: True)
@WeightCacheQuantStates.register("fp8", is_supported_func=_fp8_round_trips_via_ipc)
class TensorOnlyQuantStates(WeightCacheQuantStates):
    """Methods whose post-processing is fully captured by the exported tensors.

    Unquantized, and block-wise FP8 on CUDA: nothing is stamped outside the
    tensors, shapes are preserved, and no other object holds a reference to
    them.
    """

    def __init__(
        self,
        quant_config: Any,
        quant_method: str,
        server_args: ServerArgs,
        *,
        where: str,
    ) -> None:
        super().__init__(quant_config, quant_method, server_args, where=where)
        from sglang.srt.layers.quantization.fp8_utils import (
            initialize_fp8_gemm_config,
        )

        initialize_fp8_gemm_config(server_args)

    @classmethod
    def compute_quant_stamp(cls) -> Dict[str, str]:
        stamp = super().compute_quant_stamp()
        from sglang.srt.layers.quantization.fp8_utils import (
            get_fp8_gemm_runner_backend,
        )

        stamp["fp8_gemm_backend"] = str(get_fp8_gemm_runner_backend().value)
        return stamp

    def ipc_reshapes_weights(self) -> bool:
        return False

    def ipc_rebind_after_import(self, layer: nn.Module) -> None:
        return


@WeightCacheQuantStates.register(
    "modelopt_fp4",
    is_supported_func=_nvfp4_round_trips_via_ipc,
    capture_attrs={
        # ModelOptFp4LinearMethod: K padding for kernel alignment, read back by
        # apply() as getattr(layer, ..., 0) to pad the activation to match; and
        # the output size recomputed from the post-processed weight's shape.
        "weights_padding_cols",
        "output_size_per_partition",
        # ModelOptNvFp4FusedMoEMethod / align_fp4_moe_weights_for_flashinfer_trtllm:
        # padded MoE size read by the runners, and the guard marking w13 as
        # already deinterleaved.
        "intermediate_size_per_partition",
        "_w13_deinterleaved",
    },
)
class ModeloptFP4QuantStates(WeightCacheQuantStates):

    def __init__(
        self,
        quant_config: Any,
        quant_method: str,
        server_args: ServerArgs,
        *,
        where: str,
    ) -> None:
        super().__init__(quant_config, quant_method, server_args, where=where)
        from sglang.srt.layers.quantization.fp4_utils import (
            initialize_fp4_gemm_config,
        )

        initialize_fp4_gemm_config(server_args)

    @classmethod
    def compute_quant_stamp(cls) -> Dict[str, str]:
        stamp = super().compute_quant_stamp()
        from sglang.srt.layers.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )

        stamp["fp4_gemm_backend"] = str(get_fp4_gemm_runner_backend().value)
        return stamp

    def ipc_reshapes_weights(self) -> bool:
        # Weights are padded for kernel alignment and their block scales
        # swizzled, so exported shapes differ from what create_weights made.
        return True

    def ipc_rebind_after_import(self, layer: nn.Module) -> None:
        """Hand the MoE token dispatcher this layer's input global scale.

        The scale is a tensor reference, so a client -- which maps the daemon's
        tensors but not its Python objects -- has to redo the wiring that
        ModelOptNvFp4FusedMoEMethod.process_weights_after_loading did against
        its own copy. Kept in sync with the dispatcher call there.
        """
        from sglang.srt.layers.quantization.modelopt_quant import (
            MOE_NVFP4_DISPATCH,
            ModelOptNvFp4FusedMoEMethod,
            should_use_flashinfer_cutlass_moe_fp4_allgather,
        )

        # Dense NVFP4 layers have nothing to rebind; only the MoE ones own a
        # dispatcher holding a tensor reference.
        if not isinstance(
            getattr(layer, "quant_method", None), ModelOptNvFp4FusedMoEMethod
        ):
            return

        layer.dispatcher.set_quant_config(
            {
                "input_global_scale": (
                    layer.w13_input_scale_quant
                    if MOE_NVFP4_DISPATCH
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else None
                )
            }
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


def compute_env_stamp() -> Dict[str, str]:
    """Local environment fingerprint for the IPC weight cache.

    Returns the device compute capability and torch version of the current
    process; the quantization half of the fingerprint comes from
    WeightCacheQuantStates.compute_quant_stamp(). A daemon and a connecting
    client that differ on any of these may have run different post-processing /
    kernel-selection branches, producing weights that map cleanly over IPC yet
    serve garbage; stamping them into CacheConfig turns that into a clean
    mismatch. Imported lazily so protocol.py stays cheap to import and usable on
    CPU-only hosts (all fields degrade to "").
    """
    device_capability = ""
    torch_version = ""

    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        logger.warning(
            "[weight_cache] failed to get torch version during compute_env_stamp"
        )
    try:
        from sglang.srt.platforms import current_platform

        cap = current_platform.get_device_capability()
        if cap is not None:
            device_capability = f"{cap.major}.{cap.minor}"
    except Exception:
        logger.warning(
            "[weight_cache] failed to get device capability during compute_env_stamp"
        )

    return {
        "device_capability": device_capability,
        "torch_version": torch_version,
    }


_ATTR_UNSET = object()


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
