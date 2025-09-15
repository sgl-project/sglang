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
        torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, loc, k, v)
    except RuntimeError:  # ok, fallback to torch implementation
        k_cache[loc] = k
        v_cache[loc] = v
