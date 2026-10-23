"""ROCm glue of rotary_embedding.base: the cos/sin cache of the fallback rope kernel,
bound there under _is_hip as _hip."""

import torch


def fallback_cos_sin_cache(rope, query: torch.Tensor) -> torch.Tensor:
    """A copy of rope.cos_sin_cache in the query dtype for sgl_kernel.rotary_embedding,
    while the fused HIP kernels keep the module's cache in fp32 (the DSpark draft's K
    rope is the one HIP caller of this path).

    Kept as a plain attribute (not a buffer) and rebuilt only when the fp32
    cache or the requested dtype / device changes, so the cast is paid once.
    """
    # Workaround until the ROCm sgl_kernel ships the fp32 cos/sin dispatch of sgl-project#42494.
    src = rope.cos_sin_cache
    if src.dtype == query.dtype:
        return src
    cached = rope._hip_fallback_cache
    key = (query.dtype, query.device, src.data_ptr(), tuple(src.shape))
    if cached is None or cached[0] != key:
        cached = (key, src.to(dtype=query.dtype))
        rope._hip_fallback_cache = cached
    return cached[1]
