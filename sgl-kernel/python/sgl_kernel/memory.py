import torch

_jit_store = None
try:
    import sglang.jit_kernel.store as _jit_store
except Exception:
    pass


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
        if _jit_store is not None:
            _jit_store.store_kv_cache(k_cache, v_cache, loc, k, v)
        else:
            torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, loc, k, v)
    except RuntimeError:  # ok, fallback to torch implementation
        k_cache[loc] = k
        v_cache[loc] = v


def weak_ref_tensor(tensor):
    return (
        torch.ops.sgl_kernel.weak_ref_tensor(tensor)
        if isinstance(tensor, torch.Tensor)
        else tensor
    )
