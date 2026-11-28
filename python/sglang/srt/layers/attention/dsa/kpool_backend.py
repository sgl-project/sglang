"""Backend capabilities for pooled DSA tail tokens."""

from typing import Optional

_TRTLLM_MLA_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


def uses_raw_fp8_dsa_kv_layout(backend: Optional[str]) -> bool:
    # The raw FP8 MLA cache layout: 576 B/token, no fp8_ds_mla tile scales.
    return backend in ("trtllm", "prims_ts")


def kpool_tail_backends(
    phase: str,
    *,
    bf16_kv: bool,
    device_capability: tuple[int, int],
    raw_fp8_nope: bool = False,
) -> tuple[str, ...]:
    """Return the backends that read a pooled tail on this KV dtype and GPU.

    FA3's MLA query-value input requires SM90. Q8 prefill can read the tail
    columns as ragged ids on SM90 over BF16 KV or a raw 512-dim NoPE FP8 pool
    paired with TileLang decode. TRT-LLM uses the architectures that FlashInfer
    builds for its MLA runner.
    """
    if device_capability in _TRTLLM_MLA_CAPABILITIES:
        return ("tilelang", "trtllm")
    if device_capability != (9, 0):
        return ("tilelang",)
    if not bf16_kv:
        if phase == "prefill" and raw_fp8_nope:
            return ("tilelang", "flashmla_sparse_q8")
        return ("tilelang",)
    if phase == "prefill":
        return ("tilelang", "fa3", "flashmla_sparse_q8")
    return ("tilelang", "fa3")


def resolve_kpool_tail_backend(
    backend: str, device_capability: tuple[int, int], *, packed_fp8: bool
) -> str:
    """Map flashmla_sparse to the kernel that reads the pooled tail tokens.

    Neither target reads packed FP8 rows, so the name stays as it is there.
    """
    if backend != "flashmla_sparse" or packed_fp8:
        return backend
    if device_capability in _TRTLLM_MLA_CAPABILITIES:
        return "trtllm"
    if device_capability == (9, 0):
        return "fa3"
    return backend
