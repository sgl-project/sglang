"""Cambricon MLU device operations for the SRT platform layer."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import KVPoolKind, PlatformCapabilities, SRTPlatform


class MluDeviceMixin(DeviceMixin):
    """MLU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.MLU
    device_name: str = "mlu"
    device_type: str = "mlu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.mlu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        return float(torch.mlu.max_memory_allocated(device))

    def get_device(self, local_rank: int = 0) -> torch.device:
        return torch.device("mlu", local_rank)

    def set_device(self, device: torch.device) -> None:
        torch.mlu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.mlu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        props = torch.mlu.get_device_properties(device_id)
        return str(props.uuid)

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        torch.mlu.empty_cache()

    def synchronize(self) -> None:
        torch.mlu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.mlu.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.mlu.manual_seed_all(seed)


class MluSRTPlatform(MluDeviceMixin, SRTPlatform):
    """In-tree Cambricon MLU SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=False,
        piecewise_graph=False,
        hicache_device_kernels=False,
        supports_pynccl=False,
    )

    def get_dispatch_key_name(self) -> str:
        return "mlu"

    def apply_server_args_defaults(self, server_args) -> None:
        if server_args.quantization is not None:
            raise NotImplementedError(
                "MLU inference currently requires unquantized weights."
            )
        if server_args.speculative_algorithm is not None:
            raise NotImplementedError("MLU speculative decoding is not supported yet.")
        if server_args.sampling_backend not in (None, "pytorch"):
            raise ValueError("MLU inference requires the pytorch sampling backend.")
        default_attention_backend = server_args.attention_backend or "mlu"
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            backend = getattr(server_args, field)
            if backend not in (None, "mlu", "triton"):
                raise ValueError(f"MLU inference requires {field}='mlu' or 'triton'.")
            if backend is None:
                setattr(server_args, field, default_attention_backend)
        if (
            server_args.prefill_attention_backend
            != server_args.decode_attention_backend
        ):
            raise ValueError(
                "MLU prefill and decode share a KV cache memory pool. "
                "Use the same attention backend to ensure compatible layouts."
            )
        use_hnd = server_args.prefill_attention_backend == "mlu"
        if not envs.SGLANG_USE_HND_KVCACHE.is_set():
            envs.SGLANG_USE_HND_KVCACHE.set(use_hnd)
        elif envs.SGLANG_USE_HND_KVCACHE.get() != use_hnd:
            from sglang.srt.arg_groups.model_override_base import use_mla_backend

            # MLA uses token-linear storage independently of the MHA layout.
            if not use_mla_backend(server_args):
                raise ValueError(
                    f"MLU MHA attention backend {server_args.prefill_attention_backend!r} "
                    f"requires SGLANG_USE_HND_KVCACHE={int(use_hnd)}."
                )
        if server_args.sampling_backend is None:
            server_args.sampling_backend = "pytorch"
        if server_args.page_size is None:
            server_args.page_size = 16
        server_args.disable_custom_all_reduce = True

    def get_default_attention_backend(self) -> str:
        return "mlu"

    def get_kv_pool_cls(self, *, kind: KVPoolKind) -> Optional[type]:
        if kind == "mha":
            if not envs.SGLANG_USE_HND_KVCACHE.get():
                return None
            from sglang.srt.hardware_backend.mlu.memory_pool import (
                MLUMHATokenToKVPool,
            )

            return MLUMHATokenToKVPool
        if kind == "mla":
            from sglang.srt.hardware_backend.mlu.memory_pool import (
                MLUMLATokenToKVPool,
            )

            return MLUMLATokenToKVPool
        return None
