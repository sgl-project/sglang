import torch


def set_kv_buffer_kernel(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fallback: bool = False,
):
    try:
        if fallback:
            raise RuntimeError("Fallback to torch implementation")
        max_tokens = k_cache.shape[0]
        num_tokens = loc.shape[0]
        k_cache_2d = k_cache.view(max_tokens, -1)
        v_cache_2d = v_cache.view(max_tokens, -1)
        k_2d = k.view(num_tokens, -1)
        v_2d = v.view(num_tokens, -1)
        torch.ops.sgl_kernel.store_kv_cache(  # type: ignore
            k_cache_2d, v_cache_2d, loc, k_2d, v_2d
        )
    except RuntimeError:  # ok, fallback to torch implementation
        k_cache[loc] = k
        v_cache[loc] = v
