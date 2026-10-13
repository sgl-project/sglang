"""Backend capabilities for pooled DSA tail tokens."""

KPOOL_TAIL_BACKENDS = ("fa3", "tilelang", "trtllm")


def kpool_tail_backends(phase: str, *, bf16_kv: bool) -> tuple[str, ...]:
    if phase == "prefill" and bf16_kv:
        return (*KPOOL_TAIL_BACKENDS, "flashmla_sparse_q8")
    return KPOOL_TAIL_BACKENDS


def resolve_kpool_tail_backend(backend: str, device_sm_major: int) -> str:
    if backend == "flashmla_sparse":
        if device_sm_major >= 10:
            return "trtllm"
        if device_sm_major == 9:
            return "fa3"
    return backend
