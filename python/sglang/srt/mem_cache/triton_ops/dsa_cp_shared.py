import torch
import triton
import triton.language as tl


@triton.jit
def set_mla_kv_buffer_cp_shared_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    cp_rank: tl.constexpr,
    cp_size: tl.constexpr,
    slots_per_page: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    page = loc // slots_per_page
    page_offset = loc - page * slots_per_page
    owned = (loc >= 0) & ((page % cp_size) == cp_rank)
    local_page = page // cp_size
    local_loc = local_page * slots_per_page + page_offset
    mask = (offs < total_dim) & owned
    dst_ptr = kv_buffer_ptr + local_loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask)
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(cache_k_rope_ptr + pid_loc * rope_stride + offs_rope, mask=mask)
    else:
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < total_dim)
        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & is_rope,
            other=0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton_cp_shared(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    slots_per_page: int,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    block = triton.next_power_of_2(total_dim)
    set_mla_kv_buffer_cp_shared_kernel[(loc.numel(), 1)](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        cp_rank,
        cp_size,
        slots_per_page,
        BLOCK=block,
    )
