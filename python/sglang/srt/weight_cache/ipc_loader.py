# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Zero-copy mode: param.data points directly to IPC-mapped GPU memory. Only 1x GPU
memory needed — engine and daemon share the same physical GPU memory via CUDA IPC.
Engine depends on daemon staying alive.
"""

import logging
import os
import signal
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
    check_ipc_quant_support,
    compute_env_stamp,
    get_quant_method_name,
    hash_quant_config,
    recv_msg,
    send_msg,
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
        socket_path: str,
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str = "auto",
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format

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
        cache_data = self._fetch_from_cache(model_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                raise RuntimeError(
                    f"[IpcModelLoader] Weight cache daemon not available at "
                    f"{self.socket_path}. In daemon mode, fallback to disk "
                    f"loading is disabled because the daemon process already "
                    f"holds weights on the same GPU — loading from disk would "
                    f"cause OOM. Please ensure the weight cache daemon is "
                    f"running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] Weight cache not available or config mismatch, "
                "falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

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

        # The model now points into the daemon's GPU memory via CUDA IPC. If the
        # daemon dies, those pointers dangle, so watch it and fail loud.
        self._start_daemon_liveness_watchdog(cache_data.get("pid"))

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _start_daemon_liveness_watchdog(self, daemon_pid: Optional[int]) -> None:
        """Fail loud if the serving daemon dies while we hold its weights.

        In both client and (engine-spawned) daemon mode, the model's param.data
        points into the daemon's GPU memory via CUDA IPC, and CUDA graphs may
        capture those addresses. If the daemon exits, the pointers dangle:
        forward passes would read freed GPU memory -> illegal-address crashes or
        silent garbage. There is no safe in-place recovery, so a background
        thread polls the daemon PID and, on death, SIGKILLs this process with a
        clear message instead of letting it serve corrupt results.
        """
        if not daemon_pid or daemon_pid <= 0:
            logger.warning(
                "[IpcModelLoader] Daemon did not report a PID; skipping the "
                "daemon-liveness watchdog. A daemon crash will not be detected."
            )
            return

        def _daemon_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # exists but owned by another user
            return True

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not _daemon_alive(daemon_pid):
                    logger.critical(
                        f"[IpcModelLoader] Weight cache daemon (pid={daemon_pid}) "
                        f"died while this engine holds its weights via CUDA IPC. "
                        f"The mapped weight pointers are now dangling; continuing "
                        f"would read freed GPU memory. Terminating this process."
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            f"[IpcModelLoader] Started daemon-liveness watchdog for pid={daemon_pid}"
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
            # IPC-mapped tensors are read-only shared memory owned by the daemon
            # (its master copy is shared with every peer). SGLang is inference-only
            # and runs forward under torch.inference_mode(), so requires_grad has no
            # functional effect anyway. Force it False to make the intent explicit
            # and guarantee autograd can never write into the shared IPC memory.
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
                        f"  {name}: IPC={imported_tensor.shape}/{imported_tensor.dtype} "
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

        # Verify model is on the expected device after IPC mapping
        try:
            first_param_device = next(model.parameters()).device
            logger.info(
                f"[IpcModelLoader] Model device after zero-copy: {first_param_device}"
            )
        except StopIteration:
            pass

        return model

    def _fetch_from_cache(self, model_config) -> Optional[dict]:
        """Connect to daemon, validate config, fetch IPC handles.

        Returns the daemon response dict on success, None if the daemon is
        genuinely absent (socket file doesn't exist). Raises on all other
        failures so they are never silently swallowed as a disk-load fallback.
        """
        import socket as socket_mod

        # Harden the well-known /tmp socket path: refuse to connect through a
        # symlink so a link planted at the path cannot silently redirect us to a
        # rogue daemon. A genuine daemon socket is a real AF_UNIX node created by
        # bind(); the CacheConfig fingerprint check below still guards against a
        # stale/foreign daemon that happens to bind here.
        if os.path.islink(self.socket_path):
            raise RuntimeError(
                f"[IpcModelLoader] Refusing to connect: weight cache socket path "
                f"{self.socket_path} is a symlink, not a daemon socket."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)  # 30s timeout for large state dicts
            sock.connect(self.socket_path)
        except FileNotFoundError:
            sock.close()
            logger.info(
                f"[IpcModelLoader] Daemon socket not found at {self.socket_path}. "
                f"The daemon may not be running."
            )
            return None
        except ConnectionRefusedError:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Daemon socket exists at {self.socket_path} but "
                f"refused the connection. The daemon may have crashed after "
                f"creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Failed to connect to daemon at "
                f"{self.socket_path}: {e}"
            ) from e

        try:
            # Build engine's config fingerprint
            from sglang.srt.runtime_context import get_parallel

            ps = get_parallel()
            tp_size = ps.tp_size
            tp_rank = ps.tp_rank

            pp_size = ps.pp_size
            pp_rank = ps.pp_rank

            ep_size = ps.moe_ep_size

            from sglang.srt.runtime_context import get_server_args

            dp_size = get_server_args().dp_size

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
                dp_size=dp_size,
                ep_size=ep_size,
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
                revision=model_config.revision or "",
                **compute_env_stamp(),
            )

            logger.info(
                f"[IpcModelLoader] Requesting weights from daemon at "
                f"{self.socket_path} with config: "
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

            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[IpcModelLoader] Error communicating with daemon at "
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
