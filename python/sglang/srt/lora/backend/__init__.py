from sglang.srt.lora.backend.base_backend import BaseLoRABackend


def get_backend_from_name(name: str) -> BaseLoRABackend:
    """
    Get corresponding backend class from backend's name
    """
    if name == "triton":
        from sglang.srt.lora.backend.triton_backend import TritonLoRABackend

        return TritonLoRABackend
    elif name == "flashinfer":
        from sglang.srt.lora.backend.flashinfer_backend import FlashInferLoRABackend

        return FlashInferLoRABackend
    else:
        raise ValueError(f"Invalid backend: {name}")


__all__ = [
    "BaseLoRABackend",
    "FlashInferLoRABackend",
    "TritonLoRABackend",
    "get_backend_from_name",
]
