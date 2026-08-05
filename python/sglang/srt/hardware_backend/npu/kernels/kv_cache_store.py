import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_cache_prefix_valid_npu_kernel(
    dst_k_ptr,
    dst_v_ptr,
    src_k_ptr,
    src_v_ptr,
    loc_ptr,
    commit_lens_ptr,
    dst_k_row_stride,
    dst_v_row_stride,
    src_k_row_stride,
    src_v_row_stride,
    K_ROW_DIM: tl.constexpr,
    V_ROW_DIM: tl.constexpr,
    PREFIX_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    src_row = tl.program_id(0)
    block_id = tl.program_id(1)
    kv_id = tl.program_id(2)

    batch_id = src_row // PREFIX_WIDTH
    row_in_batch = src_row % PREFIX_WIDTH
    active = row_in_batch < tl.load(commit_lens_ptr + batch_id)
    loc = tl.load(loc_ptr + src_row, mask=active, other=0).to(tl.int64)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)

    if kv_id == 0:
        mask = active & (offsets < K_ROW_DIM)
        values = tl.load(
            src_k_ptr + src_row * src_k_row_stride + offsets,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_k_ptr + loc * dst_k_row_stride + offsets,
            values,
            mask=mask,
        )
    else:
        mask = active & (offsets < V_ROW_DIM)
        values = tl.load(
            src_v_ptr + src_row * src_v_row_stride + offsets,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_v_ptr + loc * dst_v_row_stride + offsets,
            values,
            mask=mask,
        )


def store_kv_cache_prefix_valid_npu_triton(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Commit valid K/V row prefixes into an NPU KV cache with Triton-Ascend.

    ``dst_k`` and ``dst_v`` are flattened slot-major views of either the
    paged NPU layout or the FIA layout. ``loc_2d`` is ``[batch, width]``;
    only each batch row's ``commit_lens[batch]`` prefix is committed. This
    keeps DSpark commit graph-safe without ``nonzero`` or ``index_select``.
    """
    if loc_2d.ndim != 2:
        raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}")
    if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
        raise ValueError(
            "commit_lens must match loc_2d batch size: "
            f"commit_lens={tuple(commit_lens.shape)}, loc_2d={tuple(loc_2d.shape)}"
        )
    if dst_k.ndim != 3 or dst_v.ndim != 3:
        raise ValueError(
            "dst_k and dst_v must be flattened [slots, heads, dim] views, "
            f"got {tuple(dst_k.shape)} and {tuple(dst_v.shape)}"
        )
    if src_k.ndim != 3 or src_v.ndim != 3:
        raise ValueError(
            "src_k and src_v must be [rows, heads, dim], "
            f"got {tuple(src_k.shape)} and {tuple(src_v.shape)}"
        )

    num_rows = loc_2d.numel()
    if src_k.shape[0] != num_rows or src_v.shape[0] != num_rows:
        raise ValueError(
            "source KV rows must match loc size: "
            f"src_k={tuple(src_k.shape)}, src_v={tuple(src_v.shape)}, "
            f"loc_2d={tuple(loc_2d.shape)}"
        )
    if dst_k.shape[1:] != src_k.shape[1:]:
        raise ValueError(
            f"K row shape mismatch: dst={tuple(dst_k.shape)}, src={tuple(src_k.shape)}"
        )
    if dst_v.shape[1:] != src_v.shape[1:]:
        raise ValueError(
            f"V row shape mismatch: dst={tuple(dst_v.shape)}, src={tuple(src_v.shape)}"
        )
    if not (dst_k.dtype == dst_v.dtype == src_k.dtype == src_v.dtype):
        raise ValueError(
            "K/V dtypes must match: "
            f"dst_k={dst_k.dtype}, dst_v={dst_v.dtype}, "
            f"src_k={src_k.dtype}, src_v={src_v.dtype}"
        )
    if not (
        dst_k.device
        == dst_v.device
        == src_k.device
        == src_v.device
        == loc_2d.device
        == commit_lens.device
    ):
        raise ValueError(
            "all K/V, location, and commit-length tensors must be on the same device"
        )
    if any(
        tensor.stride(-1) != 1
        or tensor.stride(-2) != tensor.shape[-1]
        for tensor in (dst_k, dst_v, src_k, src_v)
    ):
        raise ValueError("K/V head and head-dim axes must be contiguous")
    if num_rows == 0:
        return

    commit_lens = commit_lens.contiguous()
    loc_2d = loc_2d.contiguous()
    prefix_width = loc_2d.shape[1]
    k_row_dim = src_k.shape[1] * src_k.shape[2]
    v_row_dim = src_v.shape[1] * src_v.shape[2]
    block = 128
    grid = (
        num_rows,
        triton.cdiv(max(k_row_dim, v_row_dim), block),
        2,
    )
    _store_kv_cache_prefix_valid_npu_kernel[grid](
        dst_k,
        dst_v,
        src_k,
        src_v,
        loc_2d,
        commit_lens,
        dst_k.stride(0),
        dst_v.stride(0),
        src_k.stride(0),
        src_v.stride(0),
        K_ROW_DIM=k_row_dim,
        V_ROW_DIM=v_row_dim,
        PREFIX_WIDTH=prefix_width,
        BLOCK=block,
    )
