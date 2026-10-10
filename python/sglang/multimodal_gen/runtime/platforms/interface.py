# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/interface.py
from __future__ import annotations

import enum
from collections.abc import Callable
from functools import lru_cache
from pkgutil import resolve_name
from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.platforms.device_mixin import (  # noqa: F401
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

logger = init_logger(__name__)


class AttentionBackendEnum(enum.Enum):
    FA2 = enum.auto()
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    TORCH_CUDNN_SDPA = enum.auto()
    DYNAMIC_CUDNN_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()
    SPARGE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN_H3 = enum.auto()
    HYBRID_WINDOW_ATTN_H3 = enum.auto()
    SPARSE_VIDEO_GEN_2_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    AITER_SAGE = enum.auto()
    SLA_ATTN = enum.auto()
    SAGE_SLA_ATTN = enum.auto()
    LASER_ATTN = enum.auto()
    BLOCK_SPARSE_ATTN = enum.auto()
    RAIN_FUSION_ATTN = enum.auto()
    SOL_ATTN = enum.auto()
    SUBBLOCK_SPARSE_ATTN = enum.auto()
    CUBE_SPARSE_ATTN = enum.auto()
    FP8_FA_SM120 = enum.auto()
    NO_ATTENTION = enum.auto()

    def __str__(self):
        return self.name.lower()

    @property
    def is_sparse(self) -> bool:
        return self in {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
            AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3,
            AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SLA_ATTN,
            AttentionBackendEnum.SAGE_SLA_ATTN,
            AttentionBackendEnum.SPARGE_ATTN,
            AttentionBackendEnum.LASER_ATTN,
            AttentionBackendEnum.BLOCK_SPARSE_ATTN,
            AttentionBackendEnum.RAIN_FUSION_ATTN,
            AttentionBackendEnum.SOL_ATTN,
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            AttentionBackendEnum.CUBE_SPARSE_ATTN,
        }


class MMPlatform(DeviceMixin):
    device: torch.device | None = None  # Dummy attribute for compatibility

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    dispatch_key: str = ""

    # The torch.compile backend for compiling simple and
    # standalone functions. The default value is "inductor" to keep
    # the same behavior as PyTorch.
    # NOTE: for the forward part of the model, vLLM has another separate
    # compilation strategy.
    simple_compile_backend: str = "inductor"

    supported_quantization: list[str] = []

    def init_backend(self) -> None:
        """One-time backend initialization, in each worker; raising aborts startup.

        Where out-of-tree platforms register their custom-op forwards.
        """
        pass

    def apply_server_args_defaults(self, server_args: ServerArgs) -> None:
        """Apply defaults before argument normalization and validation."""
        pass

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the backend used to compile diffusion modules."""
        return self.simple_compile_backend

    def get_compile_options(self, module: torch.nn.Module) -> dict[str, object] | None:
        """Return backend-specific options for a diffusion module."""
        return None

    def get_dispatch_key_name(self) -> str:
        """Return the behavioral dispatch key used by :class:`CustomOp`.

        This is intentionally separate from ``dispatch_key``, which names a
        PyTorch dispatcher key such as ``PrivateUse1``. An out-of-tree backend
        can return an existing key such as ``cuda`` to reuse compatible
        ``forward_cuda`` implementations, or a vendor key backed by registered
        forwards and ``forward_<key>`` methods.
        """
        return "native"

    def get_torch_library_dispatch_key(self) -> str:
        """Return the key used for direct ``torch.library`` registrations."""
        if self.is_out_of_tree():
            if not self.dispatch_key:
                raise NotImplementedError(
                    "Out-of-tree diffusion platforms must define dispatch_key"
                )
            return self.dispatch_key
        return "PrivateUse1" if self.is_npu() else "CUDA"

    @classmethod
    @lru_cache(maxsize=1)
    def is_blackwell(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 10

    @classmethod
    @lru_cache(maxsize=1)
    def is_hopper(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability() == (9, 0)

    @classmethod
    @lru_cache(maxsize=1)
    def is_sm120(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 12

    @classmethod
    def is_gfx1151(cls) -> bool:
        """True on the gfx1151 (Strix Halo) ROCm arch. Overridden on
        RocmPlatform; every other platform is False."""
        return False

    @classmethod
    def is_cuda_static(cls) -> bool:
        return cls._enum == PlatformEnum.CUDA

    @classmethod
    def is_rocm_static(cls) -> bool:
        return cls._enum == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_hpu(self) -> bool:
        return hasattr(torch, "hpu") and torch.hpu.is_available()

    def is_device_type(self, device_type: str | None) -> bool:
        """Return whether a device type belongs to this platform."""
        return device_type == self.device_type

    def is_hip(self) -> bool:
        return self.is_rocm()

    @classmethod
    @lru_cache(maxsize=1)
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return True

    @classmethod
    def get_modelopt_fp4_quantize_op(cls) -> Callable | None:
        return None

    @classmethod
    def get_modelopt_fp4_gemm_op(cls) -> tuple[Callable | None, str | None]:
        return None, None

    @classmethod
    def get_modelopt_flashinfer_fp4_backend(cls) -> str:
        return "auto"

    def get_local_torch_device(self) -> torch.device:
        return self.get_device(envs.LOCAL_RANK)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class of a device."""
        return ""

    def supports_distributed_device_id(self) -> bool:
        """Whether torch.distributed accepts this platform's device ID."""
        return not (
            self.is_out_of_tree()
            or self.is_mps()
            or self.is_musa()
            or self.is_npu()
            or self.is_cpu()
            or self.is_xpu()
        )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.
        """
        raise NotImplementedError

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.

        - This will raise an Error or Warning based on the model support on
        the current platform.
        - By default all models are considered supported.
        """
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in {cls.device_name}."
            )

    def get_communicator_class(self) -> type[DeviceCommunicatorBase]:
        """Return the platform's default device communicator class."""
        from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )

        return DeviceCommunicatorBase

    def get_all_to_all_communicator_class(self) -> type[DeviceCommunicatorBase]:
        """Return the communicator used by ``all_to_all_4D``."""
        if (
            self.is_out_of_tree()
            and type(self).get_communicator_class is MMPlatform.get_communicator_class
        ):
            raise NotImplementedError(
                "Out-of-tree diffusion platforms must implement "
                "get_all_to_all_communicator_class()"
            )
        return self.get_communicator_class()

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        """Whether automatic DiT layerwise offload is enabled on this platform."""
        return True

    @classmethod
    def device_shares_host_memory(cls) -> bool:
        """Whether the accelerator draws from the same physical pool as the host.

        On such a part (DGX Spark's GB10, Jetson) a device allocation is host
        memory the kernel no longer has, and a host copy of a mapped weight is
        a second copy of bytes the page cache already holds.
        """
        return False

    @classmethod
    def optimize_vae(cls, vae: torch.nn.Module) -> torch.nn.Module:
        """Apply platform-specific optimizations to VAE after loading."""
        return vae

    def get_attn_backend(self, *args, **kwargs) -> AttentionImpl:
        attention_cls_str = self.get_attn_backend_cls_str(*args, **kwargs)
        return resolve_name(attention_cls_str)

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        """Check if a tensor is on the current platform's device."""
        return t.is_cuda


Platform = MMPlatform
