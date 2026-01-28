"""
This file is a platform abstraction for MThreads (MUSA) GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import os
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any, TypeVar

import psutil
import pymtml

# isort: off
import torch
import torchada  # noqa: F401

# isort: on
from typing_extensions import ParamSpec

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def device_id_to_physical_device_id(device_id: int) -> int:
    if "MUSA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["MUSA_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            msg = (
                "MUSA_VISIBLE_DEVICES is set to empty string, which means"
                " GPU support is disabled. If you are using ray, please unset"
                " the environment variable `MUSA_VISIBLE_DEVICES` inside the"
                " worker/actor. "
                "Check https://github.com/vllm-project/vllm/issues/8402 for"
                " more information."
            )
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


def with_mtml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pymtml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pymtml.nvmlShutdown()

    return wrapper


class MusaPlatformBase(Platform):
    _enum = PlatformEnum.MUSA
    device_name: str = "musa"
    device_type: str = "musa"
    dispatch_key: str = "MUSA"
    device_control_env_var: str = "MUSA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable MUSA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def is_full_mtlink(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls) -> None:
        pass

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.cuda.empty_cache()

        device_props = torch.cuda.get_device_properties(device_id)
        if device_props.is_integrated:
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="musa")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        logger.info("Using Torch SDPA backend.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa


# MTML utils
# Note that MTML is not affected by `MUSA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using MTML is that it will not initialize MUSA
class MtmlMusaPlatform(MusaPlatformBase):
    @classmethod
    @lru_cache(maxsize=8)
    @with_mtml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = device_id_to_physical_device_id(device_id)
            handle = pymtml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pymtml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @lru_cache(maxsize=8)
    @with_mtml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return bool(super().has_device_capability(capability, device_id))
        except RuntimeError:
            return False

    @classmethod
    @lru_cache(maxsize=8)
    @with_mtml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @lru_cache(maxsize=8)
    @with_mtml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pymtml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return str(pymtml.nvmlDeviceGetUUID(handle))

    @classmethod
    @lru_cache(maxsize=8)
    @with_mtml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pymtml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pymtml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mtml_context
    def is_full_mtlink(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by mtlink (1 hop)
        """
        handles = [pymtml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pymtml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pymtml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pymtml.NVML_P2P_STATUS_OK:
                            return False
                    except pymtml.NVMLError:
                        logger.exception(
                            "MTLink detection failed. This is normal if"
                            " your machine has no MTLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pymtml.nvmlDeviceGetHandleByIndex(device_id)
        return str(pymtml.nvmlDeviceGetName(handle))

    @classmethod
    @with_mtml_context
    def log_warnings(cls) -> None:
        device_ids: int = pymtml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("MUSA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `MUSA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMtmlMusaPlatform(MusaPlatformBase):
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def is_full_mtlink(cls, physical_device_ids: list[int]) -> bool:
        logger.error(
            "MTLink detection not possible, as context support was"
            " not found. Assuming no MTLink available."
        )
        return False


# Autodetect either MTML-enabled or non-MTML platform
# based on whether MTML is available.
mtml_available = False

if "MUSA_DISABLE_MTML" not in os.environ:
    try:
        try:
            pymtml.nvmlInit()
            mtml_available = True
        except Exception:
            mtml_available = False
    finally:
        if mtml_available:
            pymtml.nvmlShutdown()

MusaPlatform = MtmlMusaPlatform if mtml_available else NonMtmlMusaPlatform

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pymtml, _MockModule):
        MusaPlatform.log_warnings()
except ModuleNotFoundError:
    MusaPlatform.log_warnings()

if __name__ == "__main__":
    print(MusaPlatform.__name__)
    print(MusaPlatform.get_device_name())
    print(MusaPlatform.get_device_capability())
    print(MusaPlatform.get_device_total_memory())
    print(MusaPlatform.is_full_mtlink([0, 1, 2, 3, 4, 5, 6, 7]))
