import logging
from typing import Type

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

LORA_SUPPORTED_BACKENDS = {}


def register_lora_backend(name):
    def decorator(fn):
        LORA_SUPPORTED_BACKENDS[name] = fn
        return fn

    return decorator


@register_lora_backend("triton")
def create_triton_backend():
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend

    return TritonLoRABackend


@register_lora_backend("csgmv")
def create_triton_csgmv_backend():
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

    return ChunkedSgmvLoRABackend


@register_lora_backend("triton_v2")
def create_triton_v2_backend():
    from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend

    return TritonV2LoRABackend


@register_lora_backend("ascend")
def create_ascend_backend():
    from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend

    return AscendLoRABackend


@register_lora_backend("torch_native")
def create_torch_native_backend():
    from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend

    return TorchNativeLoRABackend


@register_lora_backend("uno_cublas")
def create_uno_cublas_backend():
    from sglang.srt.lora.backend.uno_cublas_backend import UnoCublasLoRABackend

    return UnoCublasLoRABackend


def routes_kv_b_lora_cached_rows(lora_backend: str, prefill_graph_backend: str) -> bool:
    """Whether MLA cached-prefix rows reach their own ``kv_b_proj`` adapters."""
    # Full prefill graphs capture the cached-prefix paths with static routing;
    # the ROCm paths (forward_mha_rocm, aiter) are not routed
    return (
        get_backend_from_name(lora_backend).supports_rows_routing
        and prefill_graph_backend != Backend.FULL
        and not is_hip()
    )


def get_backend_from_name(name: str) -> Type[BaseLoRABackend]:
    """
    Get corresponding backend class from backend's name
    """
    if name not in LORA_SUPPORTED_BACKENDS:
        raise ValueError(f"Invalid backend: {name}")
    lora_backend = LORA_SUPPORTED_BACKENDS[name]()
    return lora_backend
