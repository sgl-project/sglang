# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Two modes:
- Copy mode: imports IPC handle, copies to own allocation, releases handle.
  Engine is fully independent after loading. Requires 2x GPU memory during load.
- Zero-copy mode (default for daemon): param.data points directly to IPC-mapped
  GPU memory. Only 1x GPU memory needed — engine and daemon share the same
  physical GPU memory via CUDA IPC. Engine depends on daemon staying alive.
"""

import logging
import time
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
    _post_load_weights,
)
from sglang.srt.utils import MultiprocessingSerializer

from .protocol import (
    CacheConfig,
    get_quant_method_name,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)

logger = logging.getLogger(__name__)


class IpcModelLoader(BaseModelLoader):
    """Load model weights from a Weight Cache Daemon via CUDA IPC handles.

    If the daemon is unavailable or config mismatches, falls back to
    DefaultModelLoader (disk load).
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: str,
        copy_mode: bool = True,
        fallback_loader_cls=None,
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.copy_mode = copy_mode
        self._fallback_loader_cls = fallback_loader_cls

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> nn.Module:
        """Load model weights from the weight cache daemon.

        Falls back to DefaultModelLoader if daemon is unavailable or config
        mismatches.
        """
        tic = time.perf_counter()

        # Try to fetch state from daemon
        cache_data = self._fetch_from_cache(model_config)

        if cache_data is None:
            logger.info(
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
            device_loading_context,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        if self.copy_mode:
            model = self._load_copy_mode(
                model_config, device_config, entries,
                target_device, quant_config,
            )
        else:
            model = self._load_zero_copy_mode(
                model_config, device_config, entries,
                target_device, quant_config,
            )

        # Post-load hooks (e.g., model-specific finalization)
        _post_load_weights(model)

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC ({self.copy_mode=}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _load_zero_copy_mode(
        self, model_config, device_config, entries, target_device, quant_config,
    ) -> nn.Module:
        """Zero-copy load: map IPC tensors directly as param.data.

        The model is initialized on the meta device (no memory allocation),
        then each parameter's data is replaced with the IPC-mapped GPU tensor.
        This avoids the 2x GPU memory overhead of copy mode — the engine and
        daemon share the same physical GPU memory via CUDA IPC.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        # Initialize model on meta device to avoid any GPU/CPU memory allocation.
        # This creates the model structure with the correct parameter shapes/dtypes
        # but without allocating actual storage.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config, self.load_config, quant_config,
                )

        # Replace parameter/buffer data with IPC-imported GPU tensors.
        # The daemon's state_dict already includes post-quantization parameters
        # (e.g. weight_scale from FP8), so we don't need to run
        # process_weights_after_loading again.
        imported_refs = []
        imported_count = 0
        skipped_count = 0
        map_tic = time.perf_counter()

        # Build a lookup for entries
        for name, param in model.named_parameters():
            if name in entries:
                entry = entries[name]
                imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
                if imported_tensor.shape != param.shape or imported_tensor.dtype != param.dtype:
                    logger.warning(
                        f"[IpcModelLoader] Shape/dtype mismatch for param {name}: "
                        f"IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={param.shape}/{param.dtype}"
                    )
                    del imported_tensor
                    skipped_count += 1
                    continue
                # Zero-copy: replace meta param.data with IPC-mapped GPU tensor
                param.data = imported_tensor
                imported_refs.append(imported_tensor)
                imported_count += 1
            else:
                # Parameter not in daemon entries — allocate on GPU
                param.data = torch.empty(
                    param.shape, dtype=param.dtype, device=target_device
                )

        for name, buf in model.named_buffers():
            if name in entries:
                entry = entries[name]
                imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
                if imported_tensor.shape != buf.shape or imported_tensor.dtype != buf.dtype:
                    logger.warning(
                        f"[IpcModelLoader] Shape/dtype mismatch for buffer {name}: "
                        f"IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={buf.shape}/{buf.dtype}"
                    )
                    del imported_tensor
                    skipped_count += 1
                    continue
                buf.data = imported_tensor
                imported_refs.append(imported_tensor)
                imported_count += 1
            else:
                # Buffer not in daemon entries — allocate on GPU
                buf.data = torch.empty(
                    buf.shape, dtype=buf.dtype, device=target_device
                )

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory)
        if imported_refs:
            model._ipc_imported_tensors = imported_refs

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors, "
            f"skipped {skipped_count}, time={map_elapsed:.3f}s"
        )

        return model

    def _load_copy_mode(
        self, model_config, device_config, entries, target_device, quant_config,
    ) -> nn.Module:
        """Copy mode: initialize model on GPU, then copy IPC weights over.

        This requires enough GPU memory for both the daemon's weights and
        the engine's model during the copy phase. After copying, the daemon
        can release its GPU memory.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config, self.load_config, quant_config,
                )
                # Run quant post-processing to create parameters like weight_scale
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)

        # Replace parameter data with IPC-imported tensors (copy)
        state_dict = model.state_dict()
        imported_count = 0
        skipped_count = 0
        copy_tic = time.perf_counter()

        for name, entry in entries.items():
            if name not in state_dict:
                skipped_count += 1
                continue

            imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
            state_dict[name].data.copy_(imported_tensor)
            del imported_tensor
            imported_count += 1

        copy_elapsed = time.perf_counter() - copy_tic

        # Tell the daemon to release GPU memory since we've copied all weights.
        self._notify_daemon_release()

        logger.info(
            f"[IpcModelLoader] Copy: imported {imported_count} tensors, "
            f"skipped {skipped_count}, time={copy_elapsed:.3f}s"
        )

        return model

    def _notify_daemon_release(self):
        """Tell the daemon to release its GPU memory after copy-mode load."""
        import socket as socket_mod

        try:
            sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect(self.socket_path)
            send_msg(sock, {"type": "release"})
            result = recv_msg(sock)
            sock.close()
            if result.get("status") == "ok":
                logger.info("[IpcModelLoader] Daemon released GPU memory after copy")
            else:
                logger.warning(f"[IpcModelLoader] Daemon release response: {result}")
        except Exception as e:
            logger.warning(f"[IpcModelLoader] Failed to notify daemon to release: {e}")

    def _fetch_from_cache(self, model_config) -> Optional[dict]:
        """Connect to daemon, validate config, fetch IPC handles."""
        import socket as socket_mod

        try:
            sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
            sock.settimeout(30)  # 30s timeout for large state dicts
            sock.connect(self.socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            logger.info(f"[IpcModelLoader] Daemon not available at {self.socket_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"[IpcModelLoader] Error connecting to daemon: {e}")
            return None

        try:
            # Build engine's config fingerprint
            from sglang.srt.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            try:
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                tp_size = 1
                tp_rank = 0

            quant_config = getattr(model_config, "hf_config", None)
            if quant_config is not None:
                quant_config = getattr(quant_config, "quantization_config", None)
            quant_method = get_quant_method_name(
                getattr(model_config, "quantization", None)
            )
            if not quant_method and quant_config is not None:
                quant_method = get_quant_method_name(quant_config)

            engine_config = CacheConfig(
                model_path=model_config.model_path,
                model_arch=model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else "",
                tp_size=tp_size,
                tp_rank=tp_rank,
                dp_size=1,  # TODO: get actual dp_size
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
            )

            send_msg(sock, {"type": "fetch_state", "config": engine_config.to_dict()})
            result = recv_msg(sock)

            if result.get("status") != "ok":
                logger.info(
                    f"[IpcModelLoader] Daemon config mismatch: "
                    f"daemon={result.get('daemon_config')}, engine={engine_config.to_dict()}"
                )
                return None

            return result

        except Exception as e:
            logger.warning(f"[IpcModelLoader] Error fetching from daemon: {e}", exc_info=True)
            return None
        finally:
            sock.close()

    def _fallback_load(self, model_config, device_config) -> nn.Module:
        """Fall back to DefaultModelLoader for disk-based loading."""
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader

        # Build a new LoadConfig with the original load_format (not IPC_CACHE),
        # since DefaultModelLoader doesn't know how to handle IPC_CACHE.
        fallback_config = LoadConfig(
            load_format=LoadFormat.AUTO,
            download_dir=self.load_config.download_dir,
            model_loader_extra_config=self.load_config.model_loader_extra_config,
            tp_rank=self.load_config.tp_rank,
        )
        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(fallback_config)
        return fallback.load_model(model_config=model_config, device_config=device_config)

    def download_model(self, model_config) -> None:
        """No-op: daemon handles its own model downloading."""
        pass
