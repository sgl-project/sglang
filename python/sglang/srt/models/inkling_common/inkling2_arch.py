"""Architecture gates shared by the inkling2 model family."""

from sglang.srt.runtime_context import get_platform


def supports_fa4() -> bool:
    """Whether this device can run FA4 and the CUTE/CuTe-DSL kernels around it.

    Inkling-2's FA4-only paths (compact decode, padded GQA decode, the fused
    prologue, ``qk_l2_clip`` and ``attn_res`` CuTe kernels) all assume Hopper or
    newer. Pre-Hopper devices (e.g. A100/SM80) fall back to Triton and torch.
    """
    major = get_platform().device_capability[0]
    return major is not None and major >= 9


def prefer_triton_fallbacks(platform=None) -> bool:
    """Whether to use the Triton/torch fallbacks for the CUTE-only kernels.

    True on pre-Hopper CUDA devices only (e.g. A100/SM80). ROCm also reports a
    gfx major through ``device_sm``, so the CUDA check is what keeps ROCm on
    the existing kernels.
    """
    if platform is None:
        platform = get_platform()
    return bool(platform.is_cuda and platform.device_sm < 90)
