from .base_backend import BaseLoRABackend
from .flashinfer_backend import FlashInferLoRABackend
from .triton_backend import TritonLoRABackend


def get_backend_from_name(name: str) -> BaseLoRABackend:
    """
    Get corresponding backend class from backend's name
    """
    backend_mapping = {
        "triton": TritonLoRABackend,
        "flashinfer": FlashInferLoRABackend,
    }

    if name in backend_mapping:
        return backend_mapping[name]

    raise Exception(
        f"No supported lora backend called {name}. It should be one of {list(backend_mapping.keys())}"
    )


__all__ = [
    "BaseLoRABackend",
    "FlashInferLoRABackend",
    "TritonLoRABackend",
    "get_backend_from_name",
]
