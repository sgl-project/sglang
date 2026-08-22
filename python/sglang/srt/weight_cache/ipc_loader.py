# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Zero-copy mode: param.data points directly to IPC-mapped GPU memory. Engine
depends on daemon staying alive.
"""

import logging
import os
import signal
import stat
import threading
import time
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
)
from sglang.srt.utils import MultiprocessingSerializer

from .protocol import (
    CacheConfig,
    check_ipc_parallelism,
    check_ipc_quant_support,
    compute_env_stamp,
    get_quant_method_name,
    get_resolved_model_revision,
    hash_loader_extra_config,
    hash_quant_config,
    normalize_model_path_for_cache,
    recv_msg,
    send_msg,
)
from .registry import (
    DaemonRegistration,
    FileWeightCacheRegistry,
    process_identity_is_alive,
)

logger = logging.getLogger(__name__)

# How often the client polls the serving daemon's PID for liveness.
_DAEMON_LIVENESS_POLL_INTERVAL = 5.0


class IpcModelLoader(BaseModelLoader):
    """Load model weights from a Weight Cache Daemon via CUDA IPC handles.

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
    - CacheConfig mismatch           -> raise (do NOT disk-load on a shared GPU
                                        holding a different config's weights;
                                        also surfaces fingerprint drift bugs)
    - any protocol / transfer error  -> raise

    See _fetch_from_cache for the authoritative fallback-vs-raise contract.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: Optional[str],
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str = "auto",
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format
        self._runtime_dir = load_config.weight_cache_runtime_dir
        self._namespace = load_config.weight_cache_namespace

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
                    "[IpcModelLoader] No matching weight cache daemon is "
                    "registered. In daemon mode, fallback to disk "
                    "loading is disabled because the daemon process already "
                    "holds weights on the same GPU — loading from disk would "
                    "cause OOM. Please ensure the weight cache daemon is "
                    "running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] No weight cache is registered for this GPU and "
                "config; falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

        # Start monitoring before constructing the meta model or importing any
        # tensor handles. The synchronous liveness check closes the post-
        # handshake/pre-import window as far as PID identity probing allows;
        # the thread then protects the lifetime of the mapped model.
        daemon_metadata = cache_data["daemon"]
        self._start_daemon_liveness_watchdog(
            daemon_metadata["pid"], daemon_metadata["process_start_time"]
        )

        entries = cache_data["entries"]
        logger.info(
            f"[IpcModelLoader] Fetched {len(entries)} IPC handles from daemon "
            f"in {time.perf_counter() - tic:.2f}s"
        )

        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        model = self._load_zero_copy_mode(
            model_config,
            device_config,
            entries,
            quant_config,
        )

        # Skip _post_load_weights: the daemon already ran
        # process_weights_after_loading on the weights before exporting
        # IPC handles. Running it again would double-process (e.g.,
        # re-quantize already-quantized weights), corrupting tensor data.

        # Rebuild stale tensor views. Some modules store tensor views as
        # plain attributes (not parameters/buffers) during __init__. When
        # the model is initialized on meta device and then weights are
        # replaced via IPC mapping, these views still point to the old
        # meta storage. We must recreate them from the now-valid tensors.
        self._rebuild_stale_views(model)

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _start_daemon_liveness_watchdog(
        self,
        daemon_pid: int,
        process_start_time: float,
    ) -> None:
        """Terminate if the producer dies while this process holds its tensors."""

        if not process_identity_is_alive(daemon_pid, process_start_time):
            raise RuntimeError(
                f"[IpcModelLoader] Weight cache daemon pid={daemon_pid} died "
                "after the handshake and before tensor import"
            )

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not process_identity_is_alive(daemon_pid, process_start_time):
                    logger.critical(
                        f"[IpcModelLoader] Weight cache daemon (pid={daemon_pid}) "
                        f"died while this engine holds its weights via CUDA IPC. "
                        f"The current transport requires a live producer; "
                        f"terminating rather than serving from mappings whose "
                        f"post-exit lifetime is not supported."
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            f"[IpcModelLoader] Started daemon-liveness watchdog for pid={daemon_pid} "
            f"start_time={process_start_time}"
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

    @staticmethod
    def _rebuild_stale_views(model):
        """Rebuild tensor views that went stale after IPC weight replacement.

        RadixLinearAttention.conv_weights is a view of conv1d.weight created
        during __init__. After IPC mapping replaces conv1d.weight with a new
        tensor, the old view still points to meta-device storage. Recreate
        it from the now-valid parameter.
        """
        try:
            from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        except ImportError:
            return

        count = 0
        for _, module in model.named_modules():
            conv1d = getattr(module, "conv1d", None)
            attn = getattr(module, "attn", None)
            if conv1d is not None and isinstance(attn, RadixLinearAttention):
                if hasattr(conv1d, "weight") and conv1d.weight is not None:
                    attn.conv_weights = conv1d.weight.view(
                        conv1d.weight.size(0), conv1d.weight.size(2)
                    )
                    if hasattr(conv1d, "bias") and conv1d.bias is not None:
                        attn.bias = conv1d.bias
                    count += 1

        if count > 0:
            logger.info(f"[IpcModelLoader] Rebuilt {count} stale conv_weights views")

    @staticmethod
    def _set_module_tensor(model, name, tensor, is_param=True):
        """Replace or register a parameter/buffer in the model by its full dotted name.

        This is necessary because setting param.data on a meta-device tensor
        raises a type mismatch error (meta and CUDA tensors have incompatible
        dispatch keys). Instead, we walk the module tree and use setattr to
        replace the entire parameter/buffer object.

        If the attribute already exists as a parameter/buffer, it is replaced.
        If it doesn't exist (e.g. post-quantization params like weight_scale),
        it is registered as a new parameter or buffer.
        """
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf_name = parts[-1]
        if is_param:
            # requires_grad=False: the IPC memory is shared/read-only and SGLang
            # is inference-only, so autograd must never write into it.
            new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(obj, leaf_name, new_param)
        else:
            # register_buffer raises KeyError if the name already exists as a
            # parameter or plain attribute (not a buffer). This happens when
            # process_weights_after_loading converts a parameter to a buffer
            # (e.g. Mamba's A_log). Remove the old attribute first.
            if leaf_name in obj._parameters:
                del obj._parameters[leaf_name]
            elif hasattr(obj, leaf_name) and leaf_name not in obj._buffers:
                delattr(obj, leaf_name)
            obj.register_buffer(leaf_name, tensor)

    def _load_zero_copy_mode(
        self,
        model_config,
        device_config,
        entries,
        quant_config,
    ) -> nn.Module:
        """Zero-copy load: map IPC tensors directly as param.data.

        The model is initialized on the meta device (no memory allocation),
        then each parameter's data is replaced with the IPC-mapped GPU tensor.
        The engine and daemon share the same physical GPU memory via CUDA IPC.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        # Initialize model on meta device to avoid any GPU/CPU memory allocation.
        # This creates the model structure with the correct parameter shapes/dtypes
        # but without allocating actual storage.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

        # Build lookup dicts of existing parameter/buffer names in the
        # meta-device model. Post-quantization parameters (e.g. weight_scale
        # from FP8) are created by process_weights_after_loading, which the
        # daemon already ran. These params exist in the daemon's entries but
        # NOT in the meta-device model — we must register them as new attrs.
        # Use dicts (not sets) so we can do O(1) shape/dtype validation
        # without re-traversing the model tree on every lookup.
        # remove_duplicate=False mirrors the daemon's export (which keys tied
        # weights under every name) so a tied parameter is recognized under all
        # of its names here too.
        existing_params = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        mismatched = []
        new_params_count = 0
        map_tic = time.perf_counter()

        # Iterate over ALL daemon entries (not just model params/buffers).
        # This ensures post-quantization parameters (weight_scale, etc.)
        # that were created by process_weights_after_loading are also mapped.
        for name, entry in entries.items():
            imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
            is_param = entry.get("is_param", True)

            if name in existing_names:
                # Existing parameter/buffer — validate shape/dtype
                if name in existing_params:
                    ref_param = existing_params[name]
                else:
                    ref_param = existing_buffers[name]
                if (
                    imported_tensor.shape != ref_param.shape
                    or imported_tensor.dtype != ref_param.dtype
                ):
                    mismatched.append(
                        f"  {name}: IPC={imported_tensor.shape}/"
                        f"{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    continue

            # Replace or register the tensor in the model
            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        if mismatched:
            raise RuntimeError(
                f"[IpcModelLoader] {len(mismatched)} tensor(s) have shape/dtype "
                f"mismatch between the IPC daemon and the meta-initialized model. "
                f"The quantization method passed the IPC allowlist gate "
                f"(check_ipc_quant_support), so this is NOT an unsupported-quant "
                f"case — it indicates the daemon's weight fingerprint is "
                f"incomplete or the daemon/client configs drifted (a bug to fix), "
                f"not merely uninitialized weights:\n" + "\n".join(mismatched)
            )

        # After mapping every daemon entry, any tensor still on the meta device
        # is one the daemon did NOT provide. Filling it with torch.empty() would
        # hand the model uninitialized GPU memory — silently producing wrong
        # output, the worst failure mode for a load path. Hard-error and list the
        # offenders instead.
        #
        # The daemon exports the full state_dict AND non-persistent buffers
        # (e.g. rotary embedding cos_sin_cache), so a correct setup leaves nothing
        # on meta here. A non-empty list means the daemon's export is incomplete,
        # or the model has a genuinely-recomputable buffer that must be recomputed
        # explicitly (not filled with garbage) — add that handling here if needed.
        still_on_meta_params = [
            name
            for name, param in model.named_parameters()
            if param.device.type == "meta"
        ]
        still_on_meta_buffers = [
            name for name, buf in model.named_buffers() if buf.device.type == "meta"
        ]

        if still_on_meta_params or still_on_meta_buffers:
            raise RuntimeError(
                f"[IpcModelLoader] After IPC mapping, "
                f"{len(still_on_meta_params)} parameter(s) and "
                f"{len(still_on_meta_buffers)} buffer(s) remain on the meta device "
                f"— the daemon did not export them. Refusing to fill them with "
                f"uninitialized memory, which would silently produce wrong output. "
                f"This means the daemon's export is incomplete, or a recomputable "
                f"buffer needs explicit recompute logic here.\n"
                f"  params: {still_on_meta_params[:10]}"
                f"{'...' if len(still_on_meta_params) > 10 else ''}\n"
                f"  buffers: {still_on_meta_buffers[:10]}"
                f"{'...' if len(still_on_meta_buffers) > 10 else ''}"
            )

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory)
        if imported_refs:
            model._ipc_imported_tensors = imported_refs

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors "
            f"({new_params_count} new post-quant), time={map_elapsed:.3f}s"
        )

        return model

    def _build_engine_config(self, model_config, device_id: int) -> CacheConfig:
        from sglang.srt.runtime_context import get_parallel

        ps = get_parallel()
        check_ipc_parallelism(ps.dp_size, ps.moe_ep_size, where="client")
        quant_method, quant_config = self._resolve_engine_quant(model_config)
        load_format = getattr(
            self._fallback_load_format, "value", self._fallback_load_format
        )
        return CacheConfig(
            model_path=normalize_model_path_for_cache(
                getattr(model_config, "model_weights", model_config.model_path)
            ),
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            pp_size=ps.pp_size,
            pp_rank=ps.pp_rank,
            dp_size=ps.dp_size,
            ep_size=ps.moe_ep_size,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=model_config.revision or "",
            resolved_revision=get_resolved_model_revision(model_config),
            load_format=str(load_format),
            model_loader_extra_config_hash=hash_loader_extra_config(
                self.load_config.model_loader_extra_config
            ),
            trust_remote_code=self.load_config.weight_cache_trust_remote_code,
            **compute_env_stamp(device_id),
        )

    @staticmethod
    def _verify_response_identity(
        result: dict,
        *,
        engine_config: CacheConfig,
        device_uuid: str,
        registration: Optional[DaemonRegistration],
    ) -> None:
        try:
            returned_config = CacheConfig.from_dict(result["config"])
            daemon = result["daemon"]
            daemon_id = str(daemon["daemon_id"])
            returned_device_uuid = str(daemon["device_uuid"])
            fingerprint = str(daemon["config_fingerprint"])
            pid = int(daemon["pid"])
            process_start_time = float(daemon["process_start_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("daemon returned incomplete identity metadata") from exc

        if not returned_config.matches(engine_config):
            raise RuntimeError(
                "daemon returned a CacheConfig different from the requested config"
            )
        if fingerprint != engine_config.fingerprint():
            raise RuntimeError("daemon returned the wrong CacheConfig fingerprint")
        if returned_device_uuid != device_uuid:
            raise RuntimeError(
                "daemon is attached to a different physical GPU: "
                f"expected {device_uuid}, got {returned_device_uuid}"
            )
        if pid <= 0 or process_start_time <= 0:
            raise RuntimeError("daemon returned invalid process identity metadata")

        if registration is not None and (
            daemon_id != registration.daemon_id
            or pid != registration.pid
            or abs(process_start_time - registration.process_start_time) >= 1e-3
            or returned_device_uuid != registration.identity.device_uuid
            or fingerprint != registration.identity.config_fingerprint
        ):
            raise RuntimeError(
                "daemon response identity does not match the discovered registration"
            )

    def _fetch_from_cache(self, model_config, device_config) -> Optional[dict]:
        """Discover/connect, bind the handshake to that daemon, then fetch."""
        import socket as socket_mod

        from sglang.srt.platforms import current_platform

        socket_path = self.socket_path

        # Preserve the explicit-override fast miss: it bypasses both registry
        # access and parallel/config inspection when the named socket is absent.
        if socket_path is not None:
            try:
                explicit_info = os.lstat(socket_path)
            except FileNotFoundError:
                logger.info(
                    f"[IpcModelLoader] Daemon socket not found at {socket_path}."
                )
                return None
            if (
                not stat.S_ISSOCK(explicit_info.st_mode)
                or explicit_info.st_uid != os.getuid()
            ):
                raise RuntimeError(
                    f"[IpcModelLoader] Refusing to connect: {socket_path} is not "
                    f"a socket owned by this user."
                )

        # DeviceConfig is required for every real load path. Do not silently
        # substitute GPU 0: physical-device identity is the cache key.
        device_id = int(device_config.gpu_id)
        engine_config = self._build_engine_config(model_config, device_id)
        device_uuid = current_platform.get_device_uuid(device_id)
        registration = None

        # An explicit socket is an intentional escape hatch and never touches
        # the registry. Otherwise discovery is an exact CacheConfig + GPU match.
        if socket_path is None:
            registry = FileWeightCacheRegistry(
                self._runtime_dir, namespace=self._namespace
            )
            registration = registry.discover(engine_config, device_uuid=device_uuid)
            if registration is None:
                logger.info(
                    "[IpcModelLoader] No registered daemon matches config=%s "
                    "device_uuid=%s namespace=%s",
                    engine_config.fingerprint(),
                    device_uuid,
                    self._namespace,
                )
                return None
            socket_path = registration.socket_path

        # Only connect to a real socket node owned by us: reject a symlink, a
        # plain file, or another user's socket planted at this /tmp path. An
        # absent socket means no daemon -> fall back to disk (return None).
        try:
            st = os.lstat(socket_path)
        except FileNotFoundError:
            if registration is not None and process_identity_is_alive(
                registration.pid, registration.process_start_time
            ):
                raise RuntimeError(
                    "[IpcModelLoader] A discovered weight cache socket vanished "
                    "while its daemon is still live or its liveness is "
                    "indeterminate; refusing disk fallback during teardown"
                )
            logger.info(f"[IpcModelLoader] Daemon socket not found at {socket_path}.")
            return None
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(
                f"[IpcModelLoader] Refusing to connect: {socket_path} is not "
                f"a socket owned by this user."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)
            sock.connect(socket_path)
        except FileNotFoundError:
            sock.close()
            if registration is not None and process_identity_is_alive(
                registration.pid, registration.process_start_time
            ):
                raise RuntimeError(
                    "[IpcModelLoader] A discovered weight cache socket vanished "
                    "during connect while its daemon is still live or its "
                    "liveness is indeterminate; refusing disk fallback"
                )
            return None
        except ConnectionRefusedError:
            sock.close()
            if registration is not None and not process_identity_is_alive(
                registration.pid, registration.process_start_time
            ):
                logger.info(
                    "[IpcModelLoader] Discovered daemon died before connect; "
                    "treating the refused socket as a cache miss."
                )
                return None
            raise RuntimeError(
                f"[IpcModelLoader] Daemon socket exists at {socket_path} but "
                f"refused the connection. The daemon may have crashed after "
                f"creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Failed to connect to daemon at {socket_path}: {e}"
            ) from e

        try:
            logger.info(
                f"[IpcModelLoader] Requesting weights from daemon at "
                f"{socket_path} with config: "
                f"model={engine_config.model_path}, "
                f"arch={engine_config.model_arch}, "
                f"tp={engine_config.tp_size}/{engine_config.tp_rank}, "
                f"quant={engine_config.quant_method}, "
                f"dtype={engine_config.dtype}"
            )

            send_msg(sock, {"type": "fetch_state", "config": engine_config.to_dict()})
            result = recv_msg(sock)

            if result.get("status") != "ok":
                daemon_config = result.get("daemon_config", {})
                raise RuntimeError(
                    f"[IpcModelLoader] Daemon config mismatch!\n"
                    f"  Engine config: {engine_config.to_dict()}\n"
                    f"  Daemon config: {daemon_config}"
                )

            # Validate the returned config and physical/process identity before
            # importing any tensor mappings.
            self._verify_response_identity(
                result,
                engine_config=engine_config,
                device_uuid=device_uuid,
                registration=registration,
            )

            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[IpcModelLoader] Error communicating with daemon at "
                f"{socket_path}: {e}"
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
