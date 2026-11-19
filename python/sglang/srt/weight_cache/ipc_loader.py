# SPDX-License-Identifier: Apache-2.0
"""Weight cache model loader — loads model weights from a Weight Cache Daemon.

Zero-copy: the engine builds the model on the meta device and installs the
daemon's load-diff snapshot (weight_cache/load_diff.py): every tensor the
loaded model held, as a view of transport-mapped GPU memory, and every plain
value that loading changed. The daemon picks the backend
(--weight-cache-transport) and names it in its response. With torch_ipc the
engine depends on the daemon staying alive; with vmm_fd the engine's own
mappings keep the weights valid.
"""

import hashlib
import json
import logging
import os
import signal
import stat
import threading
import time
from typing import List, Optional

import torch
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.weight_cache import load_diff
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta

from .protocol import (
    PROTOCOL_VERSION,
    CacheConfig,
    ProtocolVersionError,
    check_ipc_quant_support,
    check_protocol_version,
    compute_env_stamp,
    get_quant_method_name,
    get_socket_path,
    handshake_with_daemon,
    hash_quant_config,
    recv_msg,
    send_msg,
)
from .transport import (
    AUTO_BACKEND,
    TORCH_IPC_BACKEND,
    WeightCacheTransportBackend,
    check_served_transport,
    get_client_transport_backend,
)

logger = logging.getLogger(__name__)

# How often the client polls the serving daemon's PID for liveness.
_DAEMON_LIVENESS_POLL_INTERVAL = 5.0


def _daemon_alive(pid: int) -> bool:
    """Whether the daemon process ``pid`` is still running.

    A zombie counts as exited. Where nothing reaps exited processes (e.g. a
    container whose PID 1 does not), a killed daemon stays a zombie, and
    signal 0 still succeeds on a zombie.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            proc_stat = f.read()
    except FileNotFoundError:
        # Exited since the signal check, unless there is no procfs at all.
        return not os.path.exists("/proc/self/stat")
    except OSError:
        return True  # unreadable: the signal check decides
    # The command name is in parentheses and may contain spaces and
    # parentheses itself, so the state is the first field after the last ")".
    fields = proc_stat[proc_stat.rfind(b")") + 1 :].split()
    return not fields or fields[0] not in (b"Z", b"X")


# Transport backends of loaded models, kept until the process exits. For
# vmm_fd, a mapping keeps the daemon's memory alive, but the imported tensors
# sit on non-owning storages and a parameter or view can outlive the model
# object, so a loaded model's mappings must never be unmapped while this process
# runs. Keeping the backends here keeps the owner of those mappings reachable;
# nothing unmaps them on garbage collection, and the driver releases this
# process's mappings and address reservation when it exits.
_LOADED_MODEL_TRANSPORTS: List[WeightCacheTransportBackend] = []


class WeightCacheModelLoader(BaseModelLoader):
    """Load model weights from a Weight Cache Daemon through the transport the
    daemon serves: CUDA IPC handles (torch_ipc) or CUDA VMM allocations passed
    as file descriptors (vmm_fd).

    In daemon mode (weight_cache_mode="daemon"), the engine and daemon share
    the same GPU. Falling back to disk loading would cause OOM because both
    processes would hold weights on the same GPU. Therefore, daemon mode
    raises an error if the daemon is unavailable instead of falling back.

    In client mode, disk fallback is allowed ONLY when the daemon is genuinely
    absent (its Unix socket file does not exist). Every other failure is a hard
    error rather than a silent fallback, so a broken IPC path never masquerades
    as a healthy (but slow, disk-loaded) server:

    - socket file missing            -> fall back to disk load
    - connection refused             -> raise (daemon crashed after binding)
    - protocol version mismatch      -> raise (another SGLang version)
    - CacheConfig mismatch           -> raise (do NOT disk-load on a shared GPU
                                        holding a different config's weights;
                                        also surfaces fingerprint drift bugs)
    - any protocol / transfer error  -> raise

    See _fetch_from_cache for the authoritative fallback-vs-raise contract.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: Optional[str] = None,
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str = "auto",
        weight_cache_transport: str = AUTO_BACKEND,
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self.weight_cache_transport = weight_cache_transport
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format
        self.preloaded_weights_bytes = 0
        self._transport_backend = get_client_transport_backend(TORCH_IPC_BACKEND)

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> nn.Module:
        """Load model weights from the weight cache daemon.

        In daemon mode, raises RuntimeError if the daemon is unavailable
        (fallback to disk loading would cause OOM on shared GPUs).
        In client mode, falls back to DefaultModelLoader.
        """
        self.preloaded_weights_bytes = 0
        tic = time.perf_counter()

        # Hard-gate unsupported quant methods before touching the daemon, so an
        # unsupported model fails explicitly instead of silently disk-loading
        # (client mode) or serving wrong-numerics IPC weights. Checked here so
        # it applies regardless of whether the daemon is reachable.
        quant_method, engine_quant_config = self._resolve_engine_quant(model_config)
        check_ipc_quant_support(quant_method, engine_quant_config, where="client")

        # Try to fetch state from daemon
        cache_data = self._fetch_from_cache(model_config, device_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                raise RuntimeError(
                    f"[WeightCacheModelLoader] Weight cache daemon not available at "
                    f"{self.socket_path}. In daemon mode, fallback to disk "
                    f"loading is disabled because the daemon process already "
                    f"holds weights on the same GPU — loading from disk would "
                    f"cause OOM. Please ensure the weight cache daemon is "
                    f"running and the config matches."
                )
            logger.warning(
                "[WeightCacheModelLoader] Weight cache not available or config "
                "mismatch, falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

        try:
            entries = cache_data["entries"]
            # Older daemons omit this field; missing metadata means no correction.
            preloaded_weights_bytes = cache_data.get("preloaded_weights_bytes", 0)
            if preloaded_weights_bytes is None:
                preloaded_weights_bytes = 0
            if (
                isinstance(preloaded_weights_bytes, bool)
                or not isinstance(preloaded_weights_bytes, int)
                or preloaded_weights_bytes < 0
            ):
                raise RuntimeError(
                    "[WeightCacheModelLoader] Daemon returned invalid weight-memory "
                    f"metadata: {preloaded_weights_bytes=}"
                )
            logger.info(
                f"[WeightCacheModelLoader] Fetched {len(entries)} storages from daemon "
                f"(transport={self._transport_backend.name}) "
                f"in {time.perf_counter() - tic:.2f}s"
            )

            from sglang.srt.model_loader.loader import (
                _get_quantization_config,
            )

            quant_config = _get_quantization_config(model_config, self.load_config)

            model = self._install(model_config, device_config, cache_data, quant_config)
        except BaseException:
            # Until load_diff.install runs, nothing refers to what the
            # transport mapped: drop it. From then on _install keeps it.
            if self._transport_backend not in _LOADED_MODEL_TRANSPORTS:
                self._transport_backend.close()
            raise
        self.preloaded_weights_bytes = preloaded_weights_bytes

        # Skip _post_load_weights: the daemon already ran
        # process_weights_after_loading on the weights before exporting
        # IPC handles. Running it again would double-process (e.g.,
        # re-quantize already-quantized weights), corrupting tensor data.

        # With torch_ipc the model points into the daemon's GPU memory via CUDA
        # IPC. If the daemon dies, those pointers dangle, so watch it and fail
        # loud. vmm_fd mappings keep the memory alive, so a dead daemon is
        # only logged there.
        self._start_daemon_liveness_watchdog(
            cache_data.get("pid"),
            fatal=self._transport_backend.name == TORCH_IPC_BACKEND,
        )

        logger.info(
            f"[WeightCacheModelLoader] Loaded model via IPC "
            f"(mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _start_daemon_liveness_watchdog(
        self, daemon_pid: Optional[int], *, fatal: bool = True
    ) -> None:
        """Fail loud if the serving daemon dies while we hold its weights.

        With torch_ipc (``fatal=True``), in both client and (engine-spawned)
        daemon mode, the model's param.data points into the daemon's GPU memory
        via CUDA IPC, and CUDA graphs may capture those addresses. If the daemon
        exits, the pointers dangle:
        forward passes would read freed GPU memory -> illegal-address crashes or
        silent garbage. There is no safe in-place recovery, so a background
        thread polls the daemon PID and, on death, SIGKILLs this process with a
        clear message instead of letting it serve corrupt results.

        With ``fatal=False`` (vmm_fd, whose mappings keep the weights valid)
        the thread only logs that the daemon exited and stops.
        """
        if not daemon_pid or daemon_pid <= 0:
            logger.warning(
                "[WeightCacheModelLoader] Daemon did not report a PID; skipping the "
                "daemon-liveness watchdog. A daemon crash will not be detected."
            )
            return

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not _daemon_alive(daemon_pid):
                    if not fatal:
                        logger.warning(
                            f"[WeightCacheModelLoader] Weight cache daemon "
                            f"(pid={daemon_pid}) exited. This engine's own VMM "
                            f"mappings keep its weights valid; serving continues."
                        )
                        return
                    logger.critical(
                        f"[WeightCacheModelLoader] Weight cache daemon "
                        f"(pid={daemon_pid}) died while this engine holds its "
                        f"weights via CUDA IPC. The mapped weight pointers are "
                        f"now dangling; continuing would read freed GPU memory. "
                        f"Terminating this process."
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            f"[WeightCacheModelLoader] Started daemon-liveness watchdog for "
            f"pid={daemon_pid}"
        )

    def _resolve_engine_quant(self, model_config):
        """Return (quant_method, quant_config) matching the daemon's fingerprint.

        Shared by the IPC allowlist gate and the CacheConfig fingerprint so the
        two can never drift apart. ModelConfig always exposes
        hf_config/quantization directly; quantization_config is the only
        genuinely-optional attribute.
        """
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(model_config.quantization)
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)
        return quant_method, quant_config

    def _install(
        self, model_config, device_config, cache_data, quant_config
    ) -> nn.Module:
        """Build the model on the meta device and install the daemon's
        load-diff snapshot in it: every tensor the daemon's model held, as a
        view of a mapped storage, and every plain value that loading changed,
        then check the result (load_diff.install). Nothing is post-processed
        again: the daemon ran process_weights_after_loading before exporting,
        and running it twice would corrupt the weights."""
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        tic = time.perf_counter()
        # Initialize the model on the meta device: no memory is allocated.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )
        built_s = time.perf_counter() - tic

        backend = self._transport_backend
        manifest = cache_data["manifest"]
        storages = {
            name: backend.import_tensor(entry)
            for name, entry in cache_data["entries"].items()
        }
        storages.update(
            (name, torch.frombuffer(bytearray(data), dtype=torch.uint8))
            for name, data in manifest["host"].items()
        )
        # Equal on every rank of one load, as SnapshotManifest asks; the
        # publish vote mints it once the snapshot loader serves the cache.
        shared = {
            k: v for k, v in cache_data["config"].items() if not k.endswith("_rank")
        }
        digest = hashlib.sha256(
            json.dumps(shared, sort_keys=True, default=str).encode()
        )
        snapshot = SnapshotManifest(
            version=f"weight-cache-{digest.hexdigest()[:16]}",
            storages={
                name: StorageMeta(nbytes=nbytes, device=device)
                for name, (nbytes, device) in manifest["storages"].items()
            },
            load_diff=manifest["load_diff"],
            attestation=b"",
            source={},
        )
        # Keep the transport backend alive for as long as the process: from
        # here on the model, or a cache it shares, may refer to the mapped
        # storages, even when the install fails (nothing is put back). VMM
        # mappings are owned by the backend; a tensor can outlive the model.
        _LOADED_MODEL_TRANSPORTS.append(backend)
        model = load_diff.install(skeleton=model, manifest=snapshot, storages=storages)
        # Keep the imported storages referenced from the model. With torch_ipc
        # each one owns its IPC mapping, which GC would close; vmm_fd mappings
        # are owned by the transport backend kept above.
        model._ipc_imported_tensors = list(storages.values())
        model._weight_cache_transport_backend = backend
        logger.info(
            f"[WeightCacheModelLoader] Installed the daemon's snapshot "
            f"(build_s={built_s:.3f}, install_s="
            f"{time.perf_counter() - tic - built_s:.3f})"
        )
        return model

    @staticmethod
    def _check_response(result, engine_config) -> None:
        """Refuse a fetch_state response before any handle is opened or fd
        mapped: another protocol version, a config mismatch, or storages that
        do not add up (a transport entry for every GPU storage, each a whole
        uint8 storage of its size, and the bytes of every CPU storage). The
        load_diff bytes themselves are checked by load_diff.install."""
        if not isinstance(result, dict):
            raise RuntimeError("[WeightCacheModelLoader] Invalid daemon response")
        if result.get("status") == "protocol_mismatch":
            raise ProtocolVersionError(result.get("message", result))
        check_protocol_version(result.get("protocol_version"))
        if result.get("status") != "ok" or result.get("config") != (
            engine_config.to_dict()
        ):
            daemon_config = result.get("daemon_config", result.get("config"))
            raise RuntimeError(
                f"[WeightCacheModelLoader] Daemon config mismatch!\n"
                f"  Engine config: {engine_config.to_dict()}\n"
                f"  Daemon config: {daemon_config}"
            )
        manifest, entries = result.get("manifest"), result.get("entries")
        mismatch = "the daemon's response does not match its manifest"
        if not (
            isinstance(manifest, dict)
            and type(manifest.get("load_diff")) is bytes
            and isinstance(manifest.get("storages"), dict)
            and isinstance(manifest.get("host"), dict)
            and isinstance(entries, dict)
            and all(
                type(name) is str
                and type(meta) is list
                and len(meta) == 2
                and type(meta[0]) is int
                and meta[0] > 0
                and type(meta[1]) is str
                for name, meta in manifest["storages"].items()
            )
        ):
            raise load_diff.SnapshotError(mismatch, ["malformed"])
        host = manifest["host"]
        expected = {
            name: nbytes
            for name, (nbytes, _) in manifest["storages"].items()
            if name not in host
        }
        # The transport moves the storages on the daemon's device only.
        devices = {manifest["storages"][name][1] for name in expected}
        if (
            set(entries) != set(expected)
            or not set(host) <= set(manifest["storages"])
            or len(devices) > 1
        ):
            raise load_diff.SnapshotError(
                mismatch,
                [
                    f"{len(entries)} transport entries and {len(host)} byte strings "
                    f"for {len(manifest['storages'])} storages"
                ],
            )
        for name, data in host.items():
            nbytes, device = manifest["storages"][name]
            if type(data) is not bytes or len(data) != nbytes or device != "cpu":
                raise load_diff.SnapshotError(
                    mismatch, [f"storage {name}: not {nbytes} bytes on the CPU"]
                )
        for name, nbytes in expected.items():
            entry = entries[name]
            if not (
                isinstance(entry, dict)
                and entry.get("shape") == [nbytes]
                and entry.get("dtype") == "uint8"
                # vmm_fd describes where in its allocation: the whole storage
                and entry.get("storage_offset", 0) == 0
                and entry.get("stride", [1]) == [1]
                and entry.get("storage_nbytes", nbytes) == nbytes
            ):
                raise load_diff.SnapshotError(
                    "the daemon's response does not match its manifest",
                    [f"storage {name}: not {nbytes} bytes"],
                )

    def _fetch_from_cache(self, model_config, device_config) -> Optional[dict]:
        """Connect to daemon, validate config, fetch IPC handles.

        Returns the daemon response dict on success, None if the daemon is
        genuinely absent (socket file doesn't exist). Raises on all other
        failures so they are never silently swallowed as a disk-load fallback.
        """
        import socket as socket_mod

        if self.socket_path is None:
            device_uuid = current_platform.get_device_uuid(int(device_config.gpu_id))
            self.socket_path = get_socket_path(device_uuid)

        # Only connect to a real socket node owned by us: reject a symlink, a
        # plain file, or another user's socket planted at this /tmp path. An
        # absent socket means no daemon -> fall back to disk (return None).
        try:
            st = os.lstat(self.socket_path)
        except FileNotFoundError:
            logger.info(
                f"[WeightCacheModelLoader] Daemon socket not found at "
                f"{self.socket_path}."
            )
            return None
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(
                f"[WeightCacheModelLoader] Refusing to connect: "
                f"{self.socket_path} is not a socket owned by this user."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)
            sock.connect(self.socket_path)
        except FileNotFoundError:
            # Raced: socket removed between lstat and connect -> treat as absent.
            sock.close()
            return None
        except ConnectionRefusedError:
            sock.close()
            raise RuntimeError(
                f"[WeightCacheModelLoader] Daemon socket exists at "
                f"{self.socket_path} but refused the connection. The daemon may "
                f"have crashed after creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[WeightCacheModelLoader] Failed to connect to daemon at "
                f"{self.socket_path}: {e}"
            ) from e

        try:
            # A daemon of another version refuses here, before it reads a
            # config or exports any handle or fd.
            handshake_with_daemon(sock)

            # Build engine's config fingerprint
            parallel = get_parallel()
            tp_size = parallel.tp_size
            tp_rank = parallel.tp_rank

            pp_size = parallel.pp_size
            pp_rank = parallel.pp_rank

            ep_size = parallel.moe_ep_size
            # The configured width, as the daemon states it: the MoE
            # communicator is wider when it aliases attention CP.
            moe_dp_size = parallel.moe_dp_size
            moe_dp_rank = parallel.moe_dp_rank
            moe_ep_rank = parallel.moe_ep_rank

            quant_method, quant_config = self._resolve_engine_quant(model_config)

            engine_config = CacheConfig(
                model_path=model_config.model_path,
                model_arch=(
                    model_config.hf_config.architectures[0]
                    if model_config.hf_config.architectures
                    else ""
                ),
                tp_size=tp_size,
                tp_rank=tp_rank,
                pp_size=pp_size,
                pp_rank=pp_rank,
                dp_size=parallel.dp_size,
                ep_size=ep_size,
                moe_dp_size=moe_dp_size,
                moe_dp_rank=moe_dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_dp_size=parallel.attn_dp_size,
                enable_dp_lm_head=parallel.enable_dp_lm_head,
                attn_cp_size=parallel.attn_cp_size,
                moe_dense_tp_size=parallel.moe_dense_tp_size,
                moe_a2a_backend=get_exec().moe.moe_a2a_backend,
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
                revision=model_config.revision or "",
                **compute_env_stamp(),
            )

            logger.info(
                f"[WeightCacheModelLoader] Requesting weights from daemon at "
                f"{self.socket_path} with config: "
                f"model={engine_config.model_path}, "
                f"arch={engine_config.model_arch}, "
                f"tp={engine_config.tp_size}/{engine_config.tp_rank}, "
                f"quant={engine_config.quant_method}, "
                f"dtype={engine_config.dtype}"
            )

            send_msg(
                sock,
                {
                    "type": "fetch_state",
                    "protocol_version": PROTOCOL_VERSION,
                    "config": engine_config.to_dict(),
                },
            )
            result = recv_msg(sock)
            self._check_response(result, engine_config)
            backend_name = result.get("transport_backend", TORCH_IPC_BACKEND)
            check_served_transport(self.weight_cache_transport, backend_name)
            self._transport_backend = get_client_transport_backend(backend_name)
            result = self._transport_backend.recv_fetch_state_response(sock, result)
            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[WeightCacheModelLoader] Error communicating with daemon at "
                f"{self.socket_path}: {e}"
            ) from e
        finally:
            sock.close()

    def _fallback_load(self, model_config, device_config) -> nn.Module:
        """Fall back to DefaultModelLoader for disk-based loading."""
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.loader import DefaultModelLoader

        fallback_config = LoadConfig(
            load_format=self._fallback_load_format,
            download_dir=self.load_config.download_dir,
            model_loader_extra_config=self.load_config.model_loader_extra_config,
            tp_rank=self.load_config.tp_rank,
        )
        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(fallback_config)
        return fallback.load_model(
            model_config=model_config, device_config=device_config
        )

    def download_model(self, model_config) -> None:
        """No-op: daemon handles its own model downloading."""
        pass
