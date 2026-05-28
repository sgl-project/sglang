# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Two modes:
- Copy mode (default): imports IPC handle, copies to own allocation, releases handle.
  Engine is fully independent after loading. ~0.3s for 70B FP16.
- Zero-copy mode: param.data points directly to IPC-mapped GPU memory.
  Fastest (<0.1s), but engine depends on daemon staying alive.
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

        # Build model on CUDA and run process_weights_after_loading
        # to establish the correct parameter structure (shapes, dtypes, new params).
        # The data will be overwritten by IPC imports, so correctness of the
        # intermediate values doesn't matter — only the structure matters.
        from sglang.srt.model_loader.loader import (
            DefaultModelLoader,
            _get_quantization_config,
            device_loading_context,
        )
        from sglang.srt.utils import set_default_torch_dtype

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
                # Run quant post-processing to create parameters like weight_scale
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)

        # Replace parameter data with IPC-imported tensors
        state_dict = model.state_dict()
        imported_refs = []  # Keep refs alive for zero-copy mode

        imported_count = 0
        skipped_count = 0
        copy_tic = time.perf_counter()

        for name, entry in entries.items():
            if name not in state_dict:
                skipped_count += 1
                continue

            imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])

            if self.copy_mode:
                # Copy to own allocation, then release IPC handle
                state_dict[name].data.copy_(imported_tensor)
                del imported_tensor
            else:
                # Zero-copy: replace param.data with IPC-mapped tensor
                state_dict[name].data = imported_tensor
                imported_refs.append(imported_tensor)

            imported_count += 1

        copy_elapsed = time.perf_counter() - copy_tic

        # For zero-copy mode, stash refs on the model to prevent GC
        if not self.copy_mode and imported_refs:
            model._ipc_imported_tensors = imported_refs

        # Post-load hooks (e.g., model-specific finalization)
        _post_load_weights(model)

        logger.info(
            f"[IpcModelLoader] Loaded {imported_count} tensors via IPC "
            f"({self.copy_mode=}), skipped {skipped_count}, "
            f"copy/mapping time={copy_elapsed:.3f}s, "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _fetch_from_cache(self, model_config) -> Optional[dict]:
        """Connect to daemon, validate config, fetch IPC handles."""
        import socket as socket_mod

        try:
            sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
            sock.settimeout(30)  # 30s timeout for large state dicts
            sock.connect(self.socket_path)
        except (socket_mod.FileNotFoundError, ConnectionRefusedError) as e:
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
                model_arch=model_config.architectures[0]
                if model_config.architectures
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
        from sglang.srt.model_loader.loader import DefaultModelLoader

        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(self.load_config)
        return fallback.load_model(model_config=model_config, device_config=device_config)

    def download_model(self, model_config) -> None:
        """No-op: daemon handles its own model downloading."""
        pass
