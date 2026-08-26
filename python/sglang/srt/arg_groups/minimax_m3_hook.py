from __future__ import annotations

from typing import Any


def m3_fp8_attn_gemm_enabled(args: Any) -> bool:
    """Whether MiniMax-M3 attention GEMMs run in fp8 (no opt-in flag; active
    whenever possible): fp8_e4m3 main + index KV caches, fp8-cast q, fp8
    sparse/MSA kernels, with dense layers on trtllm_mha's fp8-q path. Needs
    kv_cache_dtype fp8_e4m3 (e5m2 would silently mis-dispatch fmha_sm100's
    e4m3 kernel), the trtllm_mha backend (the only dense backend with fp8-q
    GEMMs), and SM100 (MSA fp8 variants and trtllm-gen fp8 dense kernels are
    sm100-only). SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 is the kill switch:
    it forces the pre-fp8 numerics (bf16 indexer + widening sparse path,
    bf16 q) without having to move off trtllm_mha.
    """
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import is_sm100_supported

    return (
        args.kv_cache_dtype == "fp8_e4m3"
        and args.attention_backend == "trtllm_mha"
        and is_sm100_supported()
        and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
    )
