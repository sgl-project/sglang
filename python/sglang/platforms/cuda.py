# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA CUDA platform implementation.

This module provides the CudaPlatform class for NVIDIA GPU support.
It supports both NVML-based (preferred, no CUDA context initialization)
and fallback torch.cuda-based device queries.

Ops are registered as @cached_property with direct imports for:
- Lazy loading (import only on first access)
- IDE navigation (Ctrl+Click on import goes to implementation)
- Testability (property access validates import)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import cached_property, lru_cache, wraps
from typing import TYPE_CHECKING, Any, TypeVar

import psutil
import torch
from typing_extensions import ParamSpec

from sglang.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Attempt to import pynvml for NVML-based device queries
_nvml_available = False
_nvml_init_error: str | None = None

# Reuse the shared pynvml import helper (prefers bundled multimodal_gen copy,
# falls back to pip-installed pynvml, returns None if neither is available).
from sglang.platforms import _try_import_pynvml

_pynvml = _try_import_pynvml()

if _pynvml is not None:
    try:
        _pynvml.nvmlInit()
        _nvml_available = True
        _pynvml.nvmlShutdown()
    except Exception as e:
        # Store error for debugging and log appropriately
        _nvml_init_error = str(e)
        # Log at INFO level so users know why NVML isn't being used
        logger.info(
            "NVML initialization failed: %s. "
            "Falling back to torch.cuda for device queries. "
            "This may initialize CUDA context earlier than intended.",
            e,
        )


def device_id_to_physical_device_id(device_id: int) -> int:
    """Map logical device ID to physical device ID based on CUDA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            msg = (
                "CUDA_VISIBLE_DEVICES is set to empty string, which means "
                "GPU support is disabled. If you are using ray, please unset "
                "the environment variable `CUDA_VISIBLE_DEVICES` inside the "
                "worker/actor."
            )
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator to wrap function with NVML init/shutdown.

    The check for NVML availability is done at runtime (inside the wrapper),
    not at decoration time. This allows the module to be imported on systems
    without NVML, and the error is raised only when the decorated method
    is actually called.

    Raises:
        RuntimeError: If NVML is not available when the decorated function is called.
    """

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if _pynvml is None:
            raise RuntimeError(
                "NVML is not available. Cannot use NVML-based device queries. "
                f"NVML init error: {_nvml_init_error}"
            )
        _pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            _pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    """
    Base class for CUDA platform implementations.

    This class provides the common interface for CUDA platforms.
    Subclasses implement device queries using either NVML or torch.cuda.
    """

    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    def init_platform(self) -> None:
        """
        Initialize CUDA platform.

        Disables cudnn SDPA by default as some PyTorch versions use it
        and it can cause crashes on certain models.
        See: https://github.com/huggingface/diffusers/issues/9704
        """
        torch.backends.cuda.enable_cudnn_sdp(False)
        # Log warnings about device configuration (e.g., mixed GPU types)
        self.log_warnings()

    # =========================================================================
    # CUDA-specific capability checks (moved from base Platform class)
    # =========================================================================

    @classmethod
    @lru_cache(maxsize=8)
    def is_hopper(cls, device_id: int = 0) -> bool:
        """Check if running on NVIDIA Hopper (SM 9.0)."""
        return torch.cuda.get_device_capability(device_id) == (9, 0)

    @classmethod
    @lru_cache(maxsize=8)
    def is_blackwell(cls, device_id: int = 0) -> bool:
        """Check if running on NVIDIA Blackwell (SM 10.x)."""
        return torch.cuda.get_device_capability(device_id)[0] == 10

    @classmethod
    @lru_cache(maxsize=8)
    def is_sm120(cls, device_id: int = 0) -> bool:
        """Check if running on SM 12.x."""
        return torch.cuda.get_device_capability(device_id)[0] == 12

    # =========================================================================
    # Memory Management
    # =========================================================================

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Return peak memory usage in bytes."""
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
        """Return available memory in GiB."""
        if empty_cache:
            torch.cuda.empty_cache()

        device_props = torch.cuda.get_device_properties(device_id)
        if device_props.is_integrated:
            # Jetson and other integrated GPUs use system memory
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    # =========================================================================
    # Async Output Support
    # =========================================================================

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """CUDA supports async output when CUDA graphs are enabled."""
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since enforce-eager is enabled, async output "
                "processor cannot be used."
            )
            return False
        return True

    # =========================================================================
    # Backend Selection
    # =========================================================================

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """
        Get the attention backend class for CUDA.

        This method handles the complex logic for selecting the appropriate
        attention backend based on GPU capability, dtype, and user selection.
        """
        target_backend: AttentionBackendEnum | None = None

        # Handle specific backend selections
        if selected_backend == AttentionBackendEnum.SLIDING_TILE_ATTN:
            try:
                from st_attn import sliding_tile_attention  # noqa: F401

                logger.info("Using Sliding Tile Attention backend")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend"
            except ImportError as e:
                raise ImportError(
                    "Sliding Tile Attention backend is not installed."
                ) from e

        elif selected_backend == AttentionBackendEnum.SAGE_ATTN:
            try:
                from sageattention import sageattn  # noqa: F401

                logger.info("Using Sage Attention backend")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn.SageAttentionBackend"
            except ImportError:
                logger.warning(
                    "SageAttention backend requested but 'sageattention' package is not installed. "
                    "Falling back to FlashAttention. Install with: pip install sageattention"
                )
                target_backend = AttentionBackendEnum.FA

        elif selected_backend == AttentionBackendEnum.SAGE_ATTN_3:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3 import (  # noqa: F401
                    SageAttention3Backend,
                )

                logger.info("Using Sage Attention 3 backend")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3.SageAttention3Backend"
            except ImportError:
                logger.warning(
                    "SageAttention3 backend requested but not installed. "
                    "Falling back to Torch SDPA. See: "
                    "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell#installation"
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA

        elif selected_backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            try:
                from vsa import block_sparse_attn  # noqa: F401

                logger.info("Using Video Sparse Attention backend")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"
            except ImportError as e:
                raise ImportError(
                    "Video Sparse Attention backend is not installed."
                ) from e

        elif selected_backend == AttentionBackendEnum.VMOBA_ATTN:
            try:
                from kernel.attn.vmoba_attn.vmoba import moba_attn_varlen  # noqa: F401

                logger.info("Using Video MOBA Attention backend")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.vmoba.VMOBAAttentionBackend"
            except ImportError as e:
                raise ImportError(
                    "Video MoBA Attention backend is not installed."
                ) from e

        elif selected_backend == AttentionBackendEnum.AITER:
            logger.info("Using AITer backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend"

        elif selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        elif selected_backend == AttentionBackendEnum.SLA_ATTN:
            logger.info("Using Sparse Linear Attention backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn.SparseLinearAttentionBackend"

        elif selected_backend == AttentionBackendEnum.SAGE_SLA_ATTN:
            logger.info("Using Sage Sparse Linear Attention backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn.SageSparseLinearAttentionBackend"

        elif selected_backend == AttentionBackendEnum.FA2:
            logger.info("Using FlashAttention2 backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn_2.FlashAttention2Backend"

        elif selected_backend in [AttentionBackendEnum.FA]:
            if cls.is_sm120():
                logger.info(
                    "FlashAttention is not supported on SM12.x in this build; "
                    "falling back to Torch SDPA."
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA
            elif cls.is_blackwell():
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                    set_fa_ver,
                )

                set_fa_ver(4)
                target_backend = AttentionBackendEnum.FA
            else:
                target_backend = AttentionBackendEnum.FA

        elif selected_backend:
            raise ValueError(f"Invalid attention backend for {cls.device_name}")
        else:
            # Auto-select based on GPU capability
            if cls.is_sm120():
                logger.info("Defaulting to Torch SDPA backend on SM12.x")
                target_backend = AttentionBackendEnum.TORCH_SDPA
            elif cls.is_blackwell():
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                    set_fa_ver,
                )

                set_fa_ver(4)
                target_backend = AttentionBackendEnum.FA
            else:
                target_backend = AttentionBackendEnum.FA

        # Ensure we have a target backend
        if target_backend is None:
            target_backend = AttentionBackendEnum.FA

        # Set FA version for Blackwell
        if target_backend == AttentionBackendEnum.FA and cls.is_blackwell():
            from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                set_fa_ver,
            )

            set_fa_ver(4)

        # Validate FlashAttention compatibility
        if not cls.has_device_capability(80):
            logger.info(
                "Cannot use FlashAttention backend for Volta and Turing GPUs. "
                "Falling back to Torch SDPA."
            )
            target_backend = AttentionBackendEnum.TORCH_SDPA
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention backend for dtype %s. "
                "Only float16 and bfloat16 are supported. Falling back to Torch SDPA.",
                dtype,
            )
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FA:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                    FlashAttentionBackend,
                )

                supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention backend for head size %d. "
                        "Falling back to Torch SDPA.",
                        head_size,
                    )
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.warning(
                    "FlashAttention package not found. Falling back to Torch SDPA. "
                    "Make sure flash_attn is built and installed."
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        logger.info("Using FlashAttention backend (FA3 for Hopper, FA4 for Blackwell)")
        return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.FlashAttentionBackend"

    # =========================================================================
    # Distributed Communication
    # =========================================================================

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Return the CUDA device communicator class path."""
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"

    @cached_property
    def device_communicator(self) -> type:
        """
        Get CUDA device communicator (direct import).

        Note: Uses the multimodal_gen communicator path for compatibility.
        """
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        return CudaCommunicator

    # =========================================================================
    # Server Args Post-processing
    # =========================================================================

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply CUDA-specific defaults to server arguments."""
        if args.attention_backend is None:
            args.attention_backend = "flashinfer"
        if args.sampling_backend is None:
            args.sampling_backend = "flashinfer"

    # =========================================================================
    # Activation Ops - Lazy loaded via @cached_property
    # =========================================================================

    @cached_property
    def silu_and_mul(self):
        """
        SiLU activation fused with multiply (SwiGLU).

        Used by: LLaMA, Mistral, DeepSeek feed-forward layers
        """
        from sgl_kernel import silu_and_mul

        return silu_and_mul

    @cached_property
    def gelu_and_mul(self):
        """GELU activation fused with element-wise multiply."""
        from sgl_kernel import gelu_and_mul

        return gelu_and_mul

    @cached_property
    def gelu_tanh_and_mul(self):
        """GELU (tanh approximation) activation fused with multiply."""
        from sgl_kernel import gelu_tanh_and_mul

        return gelu_tanh_and_mul

    # =========================================================================
    # LayerNorm Ops - Lazy loaded via @cached_property
    # =========================================================================

    @cached_property
    def rmsnorm(self):
        """RMS Normalization kernel."""
        from sgl_kernel import rmsnorm

        return rmsnorm

    @cached_property
    def fused_add_rmsnorm(self):
        """Fused add + RMS Normalization kernel."""
        from sgl_kernel import fused_add_rmsnorm

        return fused_add_rmsnorm

    @cached_property
    def gemma_rmsnorm(self):
        """Gemma-style RMS Normalization kernel."""
        from sgl_kernel import gemma_rmsnorm

        return gemma_rmsnorm

    @cached_property
    def gemma_fused_add_rmsnorm(self):
        """Gemma-style fused add + RMS Normalization kernel."""
        from sgl_kernel import gemma_fused_add_rmsnorm

        return gemma_fused_add_rmsnorm

    @classmethod
    def log_warnings(cls) -> None:
        """Log any platform-specific warnings."""
        pass


class NvmlCudaPlatform(CudaPlatformBase):
    """
    CUDA platform using NVML for device queries.

    Preferred implementation - does not initialize CUDA context,
    allowing for better memory management and multi-process support.
    """

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = device_id_to_physical_device_id(device_id)
            handle = _pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = _pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError as e:
            logger.warning(
                "Failed to get device capability for device %d: %s. "
                "Check CUDA_VISIBLE_DEVICES and device visibility.",
                device_id,
                e,
            )
            return None

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return bool(super().has_device_capability(capability, device_id))
        except RuntimeError as e:
            logger.debug(
                "has_device_capability(%s, %d) failed: %s. Returning False.",
                capability,
                device_id,
                e,
            )
            return False

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = _pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return str(_pynvml.nvmlDeviceGetUUID(handle))

    @classmethod
    @lru_cache(maxsize=8)
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = _pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(_pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_full_nvlink(cls, physical_device_ids: list[int]) -> bool:
        """Query if the set of GPUs are fully connected by NVLink (1 hop)."""
        handles = [_pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = _pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            _pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != _pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except _pynvml.NVMLError as e:
                        logger.debug(
                            "NVLink detection failed for devices %d-%d: %s. "
                            "This is normal if your machine has no NVLink.",
                            physical_device_ids[i],
                            physical_device_ids[j],
                            e,
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return str(_pynvml.nvmlDeviceGetName(handle))

    @classmethod
    @with_nvml_context
    def log_warnings(cls) -> None:
        """Log warnings about device configuration."""
        device_count: int = _pynvml.nvmlDeviceGetCount()
        if device_count > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_count)
            ]
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please "
                    "make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):
    """
    CUDA platform using torch.cuda for device queries.

    Fallback implementation for systems without NVML (e.g., Jetson).
    Note: This will initialize the CUDA context on first device query.
    """

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """
        Get device UUID via torch.cuda.

        Note: torch.cuda doesn't provide UUID directly, so we construct a
        pseudo-UUID from available device properties. This is sufficient for
        identification but not guaranteed to match NVML UUIDs.
        """
        props = torch.cuda.get_device_properties(device_id)
        # Construct a pseudo-UUID from device properties
        return f"GPU-{device_id}-{props.name.replace(' ', '_')}-{props.total_memory}"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: list[int]) -> bool:
        """NVLink detection not possible without NVML."""
        logger.debug(
            "NVLink detection not possible without NVML. Assuming no NVLink available."
        )
        return False


# Select the appropriate CUDA platform implementation
CudaPlatform = NvmlCudaPlatform if _nvml_available else NonNvmlCudaPlatform

# Note: log_warnings() is now called lazily in init_platform() to avoid
# import-time side effects. This allows the module to be imported without
# triggering NVML calls or CUDA context initialization.
