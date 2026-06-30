# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Zero-copy mode: param.data points directly to IPC-mapped GPU memory. Only 1x GPU
memory needed — engine and daemon share the same physical GPU memory via CUDA IPC.
Engine depends on daemon staying alive.
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
    hash_quant_config,
    recv_msg,
    send_msg,
)

logger = logging.getLogger(__name__)


class IpcModelLoader(BaseModelLoader):
    """Load model weights from a Weight Cache Daemon via CUDA IPC handles.

    In daemon mode (weight_cache_mode="daemon"), the engine and daemon share
    the same GPU. Falling back to disk loading would cause OOM because both
    processes would hold weights on the same GPU. Therefore, daemon mode
    raises an error if the daemon is unavailable instead of falling back.

    In client mode, fallback to DefaultModelLoader is allowed since the
    daemon is optional.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: str,
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls

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

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        model = self._load_zero_copy_mode(
            model_config,
            device_config,
            entries,
            target_device,
            quant_config,
        )

        # Post-load hooks (e.g., model-specific finalization)
        _post_load_weights(model)

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

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
            existing = getattr(obj, leaf_name, None)
            if isinstance(existing, nn.Parameter):
                new_param = nn.Parameter(tensor, requires_grad=existing.requires_grad)
            else:
                new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(obj, leaf_name, new_param)
        else:
            obj.register_buffer(leaf_name, tensor)

    def _load_zero_copy_mode(
        self,
        model_config,
        device_config,
        entries,
        target_device,
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
        existing_params = {name: param for name, param in model.named_parameters()}
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        skipped_count = 0
        new_params_count = 0
        missing_in_entries = []
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
                    logger.warning(
                        f"[IpcModelLoader] Shape/dtype mismatch for {name}: "
                        f"IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    skipped_count += 1
                    continue

            # Replace or register the tensor in the model
            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        # Handle model params/buffers that are NOT in daemon entries.
        # These are typically non-persistent buffers (e.g. rotary embedding
        # cos_sin_cache) that are computed during model init, not saved in
        # state_dict. We need to move these meta tensors to the target device.
        # We can't use model.to() or model.to_empty() because those would
        # also re-allocate IPC-mapped tensors, causing OOM. Instead, we only
        # move tensors that are still on the meta device.
        for name, param in list(model.named_parameters()):
            if name not in entries:
                missing_in_entries.append(name)

        for name, buf in list(model.named_buffers()):
            if name not in entries:
                missing_in_entries.append(name)

        if missing_in_entries:
            # Move only the meta tensors to the target device.
            # IPC-mapped tensors are already on GPU and must not be touched.
            materialized_count = 0
            for name, param in list(model.named_parameters()):
                if param.device.type == "meta":
                    gpu_tensor = torch.empty(
                        param.shape, dtype=param.dtype, device=target_device
                    )
                    self._set_module_tensor(model, name, gpu_tensor, is_param=True)
                    materialized_count += 1

            for name, buf in list(model.named_buffers()):
                if buf.device.type == "meta":
                    gpu_tensor = torch.empty(
                        buf.shape, dtype=buf.dtype, device=target_device
                    )
                    self._set_module_tensor(model, name, gpu_tensor, is_param=False)
                    materialized_count += 1

            logger.info(
                f"[IpcModelLoader] Materialized {materialized_count} meta tensors "
                f"on {target_device}"
            )
            missing_in_entries.clear()

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory)
        if imported_refs:
            model._ipc_imported_tensors = imported_refs

        if missing_in_entries:
            logger.warning(
                f"[IpcModelLoader] {len(missing_in_entries)} model params not in daemon entries, "
                f"allocated on GPU: {missing_in_entries[:5]}{'...' if len(missing_in_entries) > 5 else ''}"
            )

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors "
            f"({new_params_count} new post-quant), skipped {skipped_count}, "
            f"missing {len(missing_in_entries)}, time={map_elapsed:.3f}s"
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
        """Connect to daemon, validate config, fetch IPC handles."""
        import socket as socket_mod

        try:
            sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
            sock.settimeout(30)  # 30s timeout for large state dicts
            sock.connect(self.socket_path)
        except FileNotFoundError:
            logger.info(
                f"[IpcModelLoader] Daemon socket not found at {self.socket_path}. "
                f"The daemon may not be running."
            )
            return None
        except ConnectionRefusedError:
            logger.info(
                f"[IpcModelLoader] Daemon not accepting connections at "
                f"{self.socket_path}. It may still be loading."
            )
            return None
        except Exception as e:
            logger.warning(
                f"[IpcModelLoader] Error connecting to daemon at "
                f"{self.socket_path}: {e}"
            )
            return None

        try:
            # Build engine's config fingerprint
            from sglang.srt.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from sglang.srt.distributed.parallel_state import (
                get_moe_expert_parallel_world_size,
            )

            try:
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                tp_size = 1
                tp_rank = 0

            try:
                ep_size = get_moe_expert_parallel_world_size()
            except Exception:
                ep_size = 1

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
                model_arch=(
                    model_config.hf_config.architectures[0]
                    if model_config.hf_config.architectures
                    else ""
                ),
                tp_size=tp_size,
                tp_rank=tp_rank,
                dp_size=1,  # TODO: get actual dp_size
                ep_size=ep_size,
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
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
                logger.warning(
                    f"[IpcModelLoader] Daemon config mismatch!\n"
                    f"  Engine config: {engine_config.to_dict()}\n"
                    f"  Daemon config: {daemon_config}"
                )
                return None

            return result

        except Exception as e:
            logger.warning(
                f"[IpcModelLoader] Error fetching from daemon: {e}",
                exc_info=True,
            )
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
        return fallback.load_model(
            model_config=model_config, device_config=device_config
        )

    def download_model(self, model_config) -> None:
        """No-op: daemon handles its own model downloading."""
        pass
