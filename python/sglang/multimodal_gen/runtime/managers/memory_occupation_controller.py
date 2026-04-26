# SPDX-License-Identifier: Apache-2.0

import gc

import torch

from sglang.multimodal_gen.runtime.loader.weights_updater import get_updatable_modules
from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _get_module_device(module: torch.nn.Module) -> str:
    """Return best-effort device string for a module."""
    param = next(module.parameters(), None)
    if param is not None:
        return str(param.device)
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return str(buffer.device)

    for key, val in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(val, torch.Tensor):
            return str(val.device)

    return "cpu"


def _move_unregistered_tensors(module: torch.nn.Module, device: str) -> None:
    """Move tensor attributes that are not covered by `module.to(device)`."""

    def move_tensors(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move_tensors(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [move_tensors(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(move_tensors(v) for v in obj)
        return obj

    attrs = module.__dict__
    for attr_name, attr_value in list(attrs.items()):
        if attr_name.startswith("_"):
            continue
        if attr_name in {"_parameters", "_buffers", "_modules"}:
            continue

        moved_value = move_tensors(attr_value)
        if moved_value is not attr_value:
            attrs[attr_name] = moved_value


def _is_layerwise_offload_managed(module: torch.nn.Module) -> bool:
    if not isinstance(module, OffloadableDiTMixin):
        return False
    return any(manager.enabled for manager in module.layerwise_offload_managers)


class MemoryOccupationController:
    def __init__(
        self,
        pipeline: ComposedPipelineBase | None,
        rank: int,
        use_fsdp_inference: bool,
    ):
        self.pipeline = pipeline
        self.rank = rank
        self.use_fsdp_inference = use_fsdp_inference
        self._sleeping = False
        self._sleep_restore_map: dict[str, str] = {}

    def is_sleeping(self) -> bool:
        return self._sleeping

    def _memory_occupation_result(
        self, success: bool, message: str
    ) -> dict[str, bool | str]:
        return {
            "success": success,
            "sleeping": self._sleeping,
            "message": message,
        }

    @staticmethod
    def _clear_torch_device_cache() -> None:
        device = torch.get_device_module()
        device.synchronize()
        gc.collect()
        device.empty_cache()

    def _move_modules(self, names: list[str], device: str) -> None:
        """
        Move selected modules to device.

        This function has all-or-nothing semantics:
        - Stop on first failure (device query / move / sanitize).
        - Roll back modules already moved in this call.
        - Raise RuntimeError to caller after rollback.
        """
        modules = get_updatable_modules(self.pipeline)
        moved: list[str] = []
        src_device_map: dict[str, str] = {}

        try:
            for name in names:
                module = modules[name]
                src_device_map[name] = _get_module_device(module)
                module.to(device)
                moved.append(name)
                _move_unregistered_tensors(module, device)
        except Exception as e:
            logger.warning(
                f"[_move_modules] move failed, rollback started: target={device} moved={moved} error={e}",
            )
            for name in moved:
                module = modules.get(name)
                src_dev = src_device_map.get(name)
                module.to(src_dev)
                _move_unregistered_tensors(module, src_dev)
            raise RuntimeError(
                f"failed to move modules to {device}; rollback finished: error={e}"
            ) from e

    def _offload_active_modules_to_cpu(self) -> dict[str, str]:
        restore_map: dict[str, str] = {}
        for name, module in get_updatable_modules(self.pipeline).items():
            if _is_layerwise_offload_managed(module):
                continue
            device = _get_module_device(module)
            if not device.startswith("cpu"):
                restore_map[name] = device

        self._move_modules(list(restore_map.keys()), "cpu")
        self._clear_torch_device_cache()
        return restore_map

    def _restore_modules_to_original_devices(
        self, module_device_map: dict[str, str]
    ) -> None:
        grouped: dict[str, list[str]] = {}
        for name, device in module_device_map.items():
            grouped.setdefault(device, []).append(name)

        for device, names in grouped.items():
            self._move_modules(names, device)

    def release_memory_occupation(self) -> dict[str, bool | str]:
        logger.info(f"[SLEEP] release_memory_occupation rank={self.rank}")
        if self._sleeping:
            return self._memory_occupation_result(
                success=True,
                message="already sleeping",
            )
        if self.use_fsdp_inference:
            raise RuntimeError("sleep/wake does not support FSDP inference")
        if self.pipeline is None:
            return self._memory_occupation_result(
                success=False,
                message="pipeline not initialized",
            )

        self._sleep_restore_map = self._offload_active_modules_to_cpu()
        self._sleeping = True
        return self._memory_occupation_result(
            success=True,
            message="released GPU memory (moved active modules to CPU)",
        )

    def resume_memory_occupation(self) -> dict[str, bool | str]:
        logger.info(f"[WAKE] resume_memory_occupation rank={self.rank}")
        if not self._sleeping:
            return self._memory_occupation_result(
                success=True,
                message="already awake",
            )
        if self.pipeline is None:
            return self._memory_occupation_result(
                success=False,
                message="pipeline not initialized",
            )

        if not self._sleep_restore_map:
            self._sleeping = False
            return self._memory_occupation_result(
                success=True,
                message="no restore map; marked awake",
            )

        self._restore_modules_to_original_devices(self._sleep_restore_map)
        self._sleep_restore_map = {}
        self._sleeping = False
        return self._memory_occupation_result(
            success=True,
            message="resumed GPU memory (restored modules to original devices)",
        )
