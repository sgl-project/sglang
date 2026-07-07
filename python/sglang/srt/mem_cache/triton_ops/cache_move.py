import torch
import triton
import triton.language as tl


@triton.jit
def set_kv_buffer_prefix_valid_tiled(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_BYTES: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < ROW_BYTES
    tl.multiple_of(byte_off, 16)

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_ptr = tl.cast(src_k_ptr, tl.pointer_type(tl.uint8))
    src_v_ptr = tl.cast(src_v_ptr, tl.pointer_type(tl.uint8))
    dst_k_ptr = tl.cast(dst_k_ptr, tl.pointer_type(tl.uint8))
    dst_v_ptr = tl.cast(dst_v_ptr, tl.pointer_type(tl.uint8))

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + byte_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + byte_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + byte_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + byte_off

    k_val = tl.load(src_k_row_ptr, mask=mask_byte, other=0)
    v_val = tl.load(src_v_row_ptr, mask=mask_byte, other=0)
    tl.store(dst_k_row_ptr, k_val, mask=mask_byte)
    tl.store(dst_v_row_ptr, v_val, mask=mask_byte)


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)


# ---------------------------------------------------------------------------
# store_cache_4d — single-launch Triton write into the 4-D page-major envelope
# K/V views. At `PAGE_SIZE = 1` the kernel constexpr-folds to byte-identical
# addresses as the slot-major envelope view; at `PAGE_SIZE > 1` it uses the
# same `(page_id, tok_in_p)` split the attention read kernels use.
# ---------------------------------------------------------------------------


@triton.jit
def store_cache_4d_kernel(
    k_view_ptr,
    v_view_ptr,
    cache_k_ptr,
    cache_v_ptr,
    loc_ptr,
    # Strides in ELEMENTS (not bytes); wrapper passes view.stride(D)
    # directly. K and V may have different head_dim → different per-token
    # strides, so we carry both.
    stride_k_page,
    stride_k_tok,
    stride_v_page,
    stride_v_tok,
    stride_src_k_row,
    stride_src_v_row,
    K_ROW_DIM: tl.constexpr,  # head_num * head_dim
    V_ROW_DIM: tl.constexpr,  # head_num * v_head_dim
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Token-parallel Triton write into a 4-D envelope-strided K/V view.

    Grid: ``(N, ceil(max(K_ROW_DIM, V_ROW_DIM) / BLOCK), 2)`` where:
      - axis 0 → one program per token (loc[i])
      - axis 1 → blocks within one slot's K (or V) row
      - axis 2 → 0 = K, 1 = V (two-tensor write fused into one launch)

    For each token i, the kernel writes:
        page_id  = loc[i] // PAGE_SIZE
        tok_in_p = loc[i] %  PAGE_SIZE
        k_view[page_id, tok_in_p, :, :] = cache_k[i, :, :]
        v_view[page_id, tok_in_p, :, :] = cache_v[i, :, :]

    Cuda-graph safe: no Python branching on tensor values, no `.item()`,
    all shapes/strides known at launch time.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_kv = tl.program_id(2)

    # 1. Resolve destination slot in the 4-D view.
    loc = tl.load(loc_ptr + pid_n).to(tl.int64)
    if PAGE_SIZE == 1:
        page_id = loc
        tok_in_p = tl.zeros([], dtype=tl.int64)
    else:
        page_id = loc // PAGE_SIZE
        tok_in_p = loc % PAGE_SIZE

    # 2. Compute per-tensor source/dest pointers.
    base_off = pid_b * BLOCK + tl.arange(0, BLOCK)

    if pid_kv == 0:
        mask = base_off < K_ROW_DIM
        # The trailing (head_num, head_dim) axes of `k_view` are
        # contiguous: stride[-1]==1, stride[-2]==head_dim. So we can
        # treat them as a flat K_ROW_DIM dimension addressed by `base_off`.
        # The wrapper asserts this invariant.
        src_ptr = cache_k_ptr + pid_n * stride_src_k_row + base_off
        dst_ptr = (
            k_view_ptr + page_id * stride_k_page + tok_in_p * stride_k_tok + base_off
        )
        src = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, src, mask=mask)
    else:
        mask = base_off < V_ROW_DIM
        src_ptr = cache_v_ptr + pid_n * stride_src_v_row + base_off
        dst_ptr = (
            v_view_ptr + page_id * stride_v_page + tok_in_p * stride_v_tok + base_off
        )
        src = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, src, mask=mask)


def store_cache_4d(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
) -> None:
    """One-launch Triton write into the 4-D page-major envelope K/V views.

    Writes ``cache_k[i]`` and ``cache_v[i]`` to
    ``k_view[loc[i]//ps, loc[i]%ps, :, :]`` (and analogously for V) for
    ``i in [0, N)``.

    Contract:
        - ``k_view``, ``v_view``: 4-D ``(num_pages, page_size, head_num,
          head_dim*)``, contiguous in the trailing ``(head_num, head_dim)``
          dims (i.e., ``stride[-1] == 1`` and ``stride[-2] == head_dim``).
        - ``cache_k``, ``cache_v``: 3-D ``(N, head_num, head_dim*)``,
          contiguous in the trailing ``(head_num, head_dim)`` dims.
        - ``loc``: 1-D int64 or int32, N elements, values in
          ``[0, num_pages * page_size)``. The caller is responsible for
          clamping any negative entries to ≥ 0.
        - At ``page_size == 1`` the kernel produces byte-identical output
          to the legacy advanced-indexing path.

    Returns nothing; writes in place.
    """
    if loc.numel() == 0:
        return
    assert k_view.is_cuda and v_view.is_cuda, "store_cache_4d: CUDA only"
    assert k_view.ndim == 4 and v_view.ndim == 4, (
        f"store_cache_4d: k_view/v_view must be 4-D, "
        f"got {k_view.ndim}/{v_view.ndim}"
    )
    assert cache_k.ndim == 3 and cache_v.ndim == 3, (
        f"store_cache_4d: cache_k/cache_v must be 3-D, "
        f"got {cache_k.ndim}/{cache_v.ndim}"
    )
    assert cache_k.shape[0] == cache_v.shape[0] == loc.numel(), (
        "store_cache_4d: cache_k/cache_v/loc batch dim mismatch: "
        f"{cache_k.shape[0]}, {cache_v.shape[0]}, {loc.numel()}"
    )
    assert k_view.dtype == v_view.dtype == cache_k.dtype == cache_v.dtype, (
        "store_cache_4d: dtype mismatch: "
        f"k_view={k_view.dtype}, v_view={v_view.dtype}, "
        f"cache_k={cache_k.dtype}, cache_v={cache_v.dtype}"
    )
    # Stride invariants — the kernel addresses (head_num, head_dim) as one
    # flat ROW_DIM dimension; this requires the trailing two dims to be
    # contiguous. This holds for the page-major envelope views
    # (k_stride = (page_bytes/itemsize, k_row_bytes/itemsize, head_dim, 1)) and
    # for cache_k/cache_v produced by the model forward.
    assert k_view.stride(-1) == 1 and k_view.stride(-2) == k_view.shape[-1], (
        f"store_cache_4d: k_view trailing dims must be contiguous; "
        f"got stride={k_view.stride()}, shape={tuple(k_view.shape)}"
    )
    assert v_view.stride(-1) == 1 and v_view.stride(-2) == v_view.shape[-1], (
        f"store_cache_4d: v_view trailing dims must be contiguous; "
        f"got stride={v_view.stride()}, shape={tuple(v_view.shape)}"
    )
    assert cache_k.stride(-1) == 1 and cache_k.stride(-2) == cache_k.shape[-1], (
        f"store_cache_4d: cache_k trailing dims must be contiguous; "
        f"got stride={cache_k.stride()}, shape={tuple(cache_k.shape)}"
    )
    assert cache_v.stride(-1) == 1 and cache_v.stride(-2) == cache_v.shape[-1], (
        f"store_cache_4d: cache_v trailing dims must be contiguous; "
        f"got stride={cache_v.stride()}, shape={tuple(cache_v.shape)}"
    )

    head_num = k_view.shape[2]
    head_dim = k_view.shape[3]
    v_head_dim = v_view.shape[3]
    K_ROW_DIM = head_num * head_dim
    V_ROW_DIM = head_num * v_head_dim
    BLOCK = 128
    N = loc.numel()
    row_dim_max = max(K_ROW_DIM, V_ROW_DIM)
    grid = (N, triton.cdiv(row_dim_max, BLOCK), 2)

    store_cache_4d_kernel[grid](
        k_view,
        v_view,
        cache_k,
        cache_v,
        loc,
        k_view.stride(0),
        k_view.stride(1),
        v_view.stride(0),
        v_view.stride(1),
        cache_k.stride(0),
        cache_v.stride(0),
        K_ROW_DIM=K_ROW_DIM,
        V_ROW_DIM=V_ROW_DIM,
        PAGE_SIZE=page_size,
        BLOCK=BLOCK,
        num_warps=4,
    )
