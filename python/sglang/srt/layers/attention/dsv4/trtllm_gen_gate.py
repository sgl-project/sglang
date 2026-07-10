from __future__ import annotations

import functools

from sglang.srt.environ import envs

_VALID_BACKENDS = ("auto", "flashmla", "trtllm_gen")


@functools.lru_cache(maxsize=1)
def use_trtllm_gen_dsv4_decode() -> bool:
    """Internal-format gate for the DSv4 uniform-FP8 / trtllm-gen decode path.

    Mirrors the ``is_unified_kv_triton()`` pattern: the KV pool layout and the
    decode attention dispatch must agree, so both read this single gate.

    ``auto`` currently resolves to the packed-FP8 FlashMLA path; the opt-in
    ``trtllm_gen`` value requires SM100/SM103 (B200).
    """
    backend = envs.SGLANG_DSV4_ATTN_DECODE_BACKEND.get()
    if backend in ("auto", "flashmla"):
        return False
    if backend != "trtllm_gen":
        raise ValueError(
            f"Invalid SGLANG_DSV4_ATTN_DECODE_BACKEND={backend!r}; "
            f"expected one of {_VALID_BACKENDS}"
        )

    from sglang.srt.utils.common import is_sm100_supported

    if not is_sm100_supported():
        raise ValueError(
            "SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen requires an SM100/SM103 "
            "(B200-class) GPU; use 'auto' or 'flashmla' on this device."
        )
    return True
