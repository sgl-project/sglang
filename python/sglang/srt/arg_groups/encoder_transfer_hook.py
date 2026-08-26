from __future__ import annotations


def resolve_encoder_transfer_backend(
    backend: str, model_arch: str, tp_size: int
) -> str:
    if backend != "auto":
        return backend
    if model_arch == "KimiK3ForConditionalGeneration" and tp_size > 1:
        return "zmq_to_tokenizer"
    return "zmq_to_scheduler"
