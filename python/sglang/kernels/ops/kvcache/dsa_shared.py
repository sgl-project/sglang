import torch
import triton
import triton.language as tl


@triton.jit
def set_mla_kv_buffer_owner_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    owner_rank: tl.constexpr,
    owner_size: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    page = loc // page_size
    page_offset = loc - page * page_size
    owned = (loc >= 0) & ((page % owner_size) == owner_rank)
    local_loc = (page // owner_size) * page_size + page_offset
    mask = (offs < total_dim) & owned
    dst_ptr = kv_buffer_ptr + local_loc * buffer_stride + offs

    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs,
        mask=mask & is_nope,
        other=0,
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0,
    )
    src = tl.where(is_nope, src_nope, src_rope)
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_owner_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    owner_rank: int,
    owner_size: int,
    page_size: int,
) -> None:
    if loc.numel() == 0:
        return
    total_dim = cache_k_nope.shape[-1] + cache_k_rope.shape[-1]
    set_mla_kv_buffer_owner_kernel[(loc.numel(),)](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        cache_k_nope.shape[-1],
        cache_k_rope.shape[-1],
        owner_rank,
        owner_size,
        page_size,
        BLOCK=triton.next_power_of_2(total_dim),
    )
