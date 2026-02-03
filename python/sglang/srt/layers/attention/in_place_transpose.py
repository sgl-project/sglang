"""
In-place transpose kernel for SSM states.

This kernel performs in-place transpose of the last two dimensions of ssm_states
for selected indices. Used to convert between K-last and V-last memory layouts
without allocating intermediate buffers.

Requires: K == V (square matrix for in-place transpose)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _in_place_transpose_kernel_v5(
    # Pointers
    ssm_states_ptr,
    cache_indices_ptr,
    # Strides
    stride_seq,
    stride_head,
    stride_row,
    stride_col,
    # Dimensions
    num_heads: tl.constexpr,
    K: tl.constexpr,
    TILE_M: tl.constexpr,  # Tile size for rows
    TILE_N: tl.constexpr,  # Tile size for cols (should equal TILE_M for square)
):
    """
    Tile-based transpose with better cache locality.
    
    For in-place transpose of KxK matrix:
    - Diagonal tiles: swap upper triangle within tile
    - Off-diagonal tiles: load tile (i,j) and tile (j,i), swap and store back
    
    Grid: (batch_size, num_heads, num_tile_pairs)
    where num_tile_pairs = num_tiles * (num_tiles + 1) / 2
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_pair_idx = tl.program_id(2)
    
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    if seq_idx < 0:
        return
    
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Number of tiles
    num_tiles = (K + TILE_M - 1) // TILE_M
    
    # Convert tile_pair_idx to (tile_i, tile_j) using triangular number formula
    # tile_pair_idx = tile_j * (tile_j + 1) / 2 + tile_i, where tile_i <= tile_j
    tile_j_float = (-1.0 + tl.sqrt(1.0 + 8.0 * tile_pair_idx.to(tl.float32))) / 2.0
    tile_j = tile_j_float.to(tl.int32)
    
    # Boundary adjustment
    check = (tile_j + 1) * tile_j // 2
    tile_j = tl.where(check > tile_pair_idx, tile_j - 1, tile_j)
    tile_j = tl.maximum(tile_j, 0)
    
    tile_i = tile_pair_idx - tile_j * (tile_j + 1) // 2
    
    # Tile boundaries
    row_start = tile_i * TILE_M
    col_start = tile_j * TILE_N
    
    # Offsets within tile
    row_offs = tl.arange(0, TILE_M)
    col_offs = tl.arange(0, TILE_N)
    
    # Global coordinates
    rows = row_start + row_offs[:, None]  # [TILE_M, 1]
    cols = col_start + col_offs[None, :]  # [1, TILE_N]
    
    # Masks for valid elements
    row_mask = rows < K
    col_mask = cols < K
    valid_mask = row_mask & col_mask
    
    if tile_i == tile_j:
        # Diagonal tile: swap upper triangle (cols > rows)
        upper_mask = valid_mask & (cols > rows)
        
        # Addresses for (rows, cols) and (cols, rows)
        addr_rc = base_ptr + rows * stride_row + cols * stride_col
        addr_cr = base_ptr + cols * stride_row + rows * stride_col
        
        # Load
        val_rc = tl.load(addr_rc, mask=upper_mask, other=0.0)
        val_cr = tl.load(addr_cr, mask=upper_mask, other=0.0)
        
        # Swap and store
        tl.store(addr_rc, val_cr, mask=upper_mask)
        tl.store(addr_cr, val_rc, mask=upper_mask)
    else:
        # Off-diagonal tiles: swap tile (i,j) with tile (j,i)
        # Tile at (tile_i, tile_j) swaps with tile at (tile_j, tile_i)
        # After transpose: (i,j)[r,c] <-> (j,i)[c,r]
        
        # Upper tile (tile_i, tile_j): (rows, cols)
        addr_upper = base_ptr + rows * stride_row + cols * stride_col
        
        # For the swap: (rows, cols) should get value from (cols, rows)
        # which is in the lower tile (tile_j, tile_i)
        addr_lower = base_ptr + cols * stride_row + rows * stride_col
        
        # Load both tiles
        val_upper = tl.load(addr_upper, mask=valid_mask, other=0.0)
        val_lower = tl.load(addr_lower, mask=valid_mask, other=0.0)
        
        # Swap
        tl.store(addr_upper, val_lower, mask=valid_mask)
        tl.store(addr_lower, val_upper, mask=valid_mask)


def in_place_transpose_indexed(
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> None:
    """
    In-place transpose the last two dimensions of ssm_states for selected indices.
    
    Args:
        ssm_states: Tensor of shape [total_seqs, num_heads, K, V] where K == V
        cache_indices: Tensor of shape [batch_size] with indices into ssm_states
    
    Requirements:
        - K must equal V (square matrix)
        - ssm_states must be contiguous
        - cache_indices values < 0 are treated as padding and skipped
    
    After this operation, ssm_states[cache_indices] will have its last two
    dimensions transposed in-place.
    """
    assert ssm_states.is_contiguous(), "ssm_states must be contiguous"
    assert ssm_states.shape[-1] == ssm_states.shape[-2], "K must equal V for in-place transpose"
    
    batch_size = cache_indices.shape[0]
    num_heads = ssm_states.shape[1]
    K = ssm_states.shape[2]
    
    if batch_size == 0:
        return
    
    # Get strides (in elements)
    stride_seq = ssm_states.stride(0)
    stride_head = ssm_states.stride(1)
    stride_row = ssm_states.stride(2)
    stride_col = ssm_states.stride(3)
    
    # Tile-based kernel (v5) - best performance
    TILE_M = 32
    TILE_N = 32
    num_tiles = (K + TILE_M - 1) // TILE_M
    num_tile_pairs = num_tiles * (num_tiles + 1) // 2
    grid = (batch_size, num_heads, num_tile_pairs)
    
    _in_place_transpose_kernel_v5[grid](
        ssm_states,
        cache_indices,
        stride_seq,
        stride_head,
        stride_row,
        stride_col,
        num_heads=num_heads,
        K=K,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
    )
