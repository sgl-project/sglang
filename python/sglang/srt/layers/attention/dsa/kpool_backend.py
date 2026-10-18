"""Backend capabilities for pooled DSA tail tokens."""

KPOOL_TAIL_BACKENDS = ("fa3", "tilelang", "trtllm")


def kpool_tail_backends(phase: str, *, bf16_kv: bool) -> tuple[str, ...]:
    if phase == "prefill" and bf16_kv:
        return (*KPOOL_TAIL_BACKENDS, "flashmla_sparse_q8")
    return KPOOL_TAIL_BACKENDS


def resolve_kpool_tail_backend(
    backend: str,
    device_sm_major: int,
    *,
    phase: str,
    bf16_kv: bool,
) -> str:
    if backend == "flashmla_auto" and phase == "prefill" and bf16_kv:
        backend = "flashmla_sparse"
    if backend == "flashmla_sparse":
        if device_sm_major >= 10:
            return "trtllm"
        if device_sm_major == 9:
            return "fa3"
    return backend


def supported_kpool_tail_backends(
    phase: str, *, bf16_kv: bool, device_sm_major: int
) -> tuple[str, ...]:
    """Return canonical tail backends and aliases that resolve to them."""
    backends = kpool_tail_backends(phase, bf16_kv=bf16_kv)
    aliases = ("flashmla_sparse", "flashmla_auto")
    return (
        *backends,
        *(
            alias
            for alias in aliases
            if resolve_kpool_tail_backend(
                alias,
                device_sm_major,
                phase=phase,
                bf16_kv=bf16_kv,
            )
            in backends
        ),
    )
