import logging

from sglang.srt.lora.backend.base_backend import BaseLoRABackend

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


@register_lora_backend("ascend")
def create_ascend_backend():
    from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend

    return AscendLoRABackend


@register_lora_backend("flashinfer")
def create_flashinfer_backend():
    raise ValueError(
        "FlashInfer LoRA backend has been deprecated, please use `triton` instead."
    )


def get_backend_from_name(name: str) -> BaseLoRABackend:
    """
    Get corresponding backend class from backend's name
    """
    if name not in LORA_SUPPORTED_BACKENDS:
        raise ValueError(f"Invalid backend: {name}")
    lora_backend = LORA_SUPPORTED_BACKENDS[name]()
    return lora_backend
