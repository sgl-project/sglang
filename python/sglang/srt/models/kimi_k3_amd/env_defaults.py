"""AITER and SGLang env defaults Kimi-K3.1 is served with on gfx950."""

import os

K31_BF16_TUNED_GEMM = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kimi_k31_bf16_tuned_gemm.csv"
)


def apply_kimi_k31_rocm_env_defaults() -> None:
    """AITER and SGLang env defaults Kimi-K3.1 is served with on gfx950, unless the user set them."""
    # AITER's opus MoE stage 2 otherwise stores per-slot down outputs in fp8 before the
    # top-k sum: 11x the error of the bf16 partials the reference keeps.
    os.environ.setdefault("AITER_OPUS_STAGE2_FP8", "0")
    # the same experts as AITER's biased top-k, in under half its time
    os.environ.setdefault("SGLANG_K3_RADIX4_TOPK", "1")
    # every rank loads the TreeCore the gfx950 image bundles instead of building its own,
    # so no two ranks run different implementations and diverge on HiCache reclaim
    os.environ.setdefault("SGLANG_RUST_BUILD_MODE", "never")
    if "AITER_CONFIG_GEMM_BF16" not in os.environ:
        from sglang.srt.utils.aiter_tuned_tables import GEMM_BF16, install_intree_table

        # the K3.1 rows add shapes aiter does not tune; a shape it tunes keeps aiter's row
        install_intree_table(GEMM_BF16, (K31_BF16_TUNED_GEMM,))
