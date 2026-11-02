"""Backend capabilities for pooled DSA tail tokens."""


def kpool_tail_backends(
    phase: str, *, bf16_kv: bool, device_sm_major: int
) -> tuple[str, ...]:
    """Return the backends that read a pooled tail on this KV dtype and GPU.

    FA3 needs BF16 rows and does not run on Blackwell. Q8 prefill reads the tail
    columns as ragged ids and runs only on SM90. Packed FP8 rows through those
    columns are untested. FlashInfer's trtllm-gen MLA runs only on SM10x.
    """
    if device_sm_major == 10:
        return ("tilelang", "trtllm")
    if not bf16_kv or device_sm_major > 10:
        return ("tilelang",)
    if phase == "prefill" and device_sm_major == 9:
        return ("tilelang", "fa3", "flashmla_sparse_q8")
    return ("tilelang", "fa3")


def resolve_kpool_tail_backend(
    backend: str, device_sm_major: int, *, packed_fp8: bool
) -> str:
    """Map flashmla_sparse to the kernel that reads the pooled tail tokens.

    Neither target reads packed FP8 rows, so the name stays as it is there.
    """
    if backend != "flashmla_sparse" or packed_fp8:
        return backend
    if device_sm_major >= 10:
        return "trtllm"
    if device_sm_major == 9:
        return "fa3"
    return backend
