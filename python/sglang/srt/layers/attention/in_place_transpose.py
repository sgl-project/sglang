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


# ============================================================================
# Optimized v4: Row-wise vectorized transpose with better memory coalescing
# ============================================================================

@triton.jit
def _in_place_transpose_kernel_v4(
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
    BLOCK_SIZE: tl.constexpr,  # Number of rows to process per program
):
    """
    Optimized transpose kernel with better memory coalescing.
    
    Key optimizations:
    1. Each program processes multiple rows for better parallelism
    2. Vectorized column loads/stores for coalesced memory access
    3. Process entire row at once, swap with corresponding column
    
    Grid: (batch_size * num_heads * ceil(K/BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    
    # Decompose pid
    num_row_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    row_block_idx = pid % num_row_blocks
    temp = pid // num_row_blocks
    head_idx = temp % num_heads
    batch_idx = temp // num_heads
    
    # Load sequence index
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    if seq_idx < 0:
        return
    
    # Base pointer for this (seq, head)
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Row range for this program
    row_start = row_block_idx * BLOCK_SIZE
    
    # Column offsets for vectorized access
    col_offs = tl.arange(0, K)
    
    # Process each row in this block
    for local_row in range(BLOCK_SIZE):
        row_i = row_start + local_row
        
        # Only process valid rows (use mask instead of early return)
        row_valid = row_i < K
        
        # For row i, we need to swap elements (i, j) with (j, i) for j > i
        # Only swap upper triangle: j > row_i
        swap_mask = (col_offs > row_i) & row_valid
        
        # Row elements: (row_i, col_offs)
        row_addrs = base_ptr + row_i * stride_row + col_offs * stride_col
        # Column elements: (col_offs, row_i) 
        col_addrs = base_ptr + col_offs * stride_row + row_i * stride_col
        
        # Load with mask
        row_vals = tl.load(row_addrs, mask=swap_mask, other=0.0)
        col_vals = tl.load(col_addrs, mask=swap_mask, other=0.0)
        
        # Store swapped
        tl.store(row_addrs, col_vals, mask=swap_mask)
        tl.store(col_addrs, row_vals, mask=swap_mask)


# ============================================================================
# Optimized v5: Tile-based with shared memory for cache locality
# ============================================================================

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
        # Tile (tile_i, tile_j): elements at (row_start+r, col_start+c)
        # Tile (tile_j, tile_i): elements at (col_start+r, row_start+c)
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


@triton.jit
def _in_place_transpose_kernel(
    # Pointers
    ssm_states_ptr,      # [total_seqs, num_heads, K, K]
    cache_indices_ptr,   # [batch_size]
    # Strides
    stride_seq,          # stride for seq dimension
    stride_head,         # stride for head dimension  
    stride_row,          # stride for row dimension
    stride_col,          # stride for col dimension
    # Dimensions
    K: tl.constexpr,
    # Block size for tiling
    BLOCK_K: tl.constexpr,
):
    """
    In-place transpose kernel using tile-based approach.
    
    Grid: (batch_size, num_heads, num_tiles * (num_tiles + 1) // 2)
    Each program handles one tile or one tile-pair.
    
    For diagonal tiles (i, i): load, transpose in registers, store back
    For off-diagonal tile pairs (i, j) and (j, i): swap with transpose
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    
    # Load the actual sequence index from cache_indices
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    
    # Skip padding (seq_idx < 0)
    if seq_idx < 0:
        return
    
    # Calculate number of tiles per dimension
    num_tiles = (K + BLOCK_K - 1) // BLOCK_K
    
    # Convert linear tile_idx to (tile_row, tile_col) for upper triangle
    # tile_idx maps to upper triangle: (0,0), (0,1), (0,2), ..., (1,1), (1,2), ...
    # Use quadratic formula to find tile_row
    # tile_idx = tile_row * num_tiles - tile_row * (tile_row - 1) // 2 + (tile_col - tile_row)
    
    # Find tile_row using the formula
    tile_row = 0
    cumsum = 0
    for i in range(num_tiles):
        tiles_in_row = num_tiles - i
        if cumsum + tiles_in_row > tile_idx:
            tile_row = i
            tile_col = tile_row + (tile_idx - cumsum)
            break
        cumsum += tiles_in_row
    else:
        # Should not reach here if grid is sized correctly
        return
    
    # Calculate base pointer for this head
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Calculate tile start positions
    row_start = tile_row * BLOCK_K
    col_start = tile_col * BLOCK_K
    
    # Create offset ranges
    row_offsets = tl.arange(0, BLOCK_K)
    col_offsets = tl.arange(0, BLOCK_K)
    
    # Handle diagonal tile (i, i) - transpose in place
    if tile_row == tile_col:
        # For diagonal tile, only swap upper triangle elements
        # Load the tile
        for i in range(BLOCK_K):
            actual_row = row_start + i
            if actual_row >= K:
                break
            for j in range(i + 1, BLOCK_K):
                actual_col = col_start + j
                if actual_col >= K:
                    break
                
                # Calculate addresses for (i, j) and (j, i)
                addr_ij = base_ptr + actual_row * stride_row + actual_col * stride_col
                addr_ji = base_ptr + actual_col * stride_row + actual_row * stride_col
                
                # Load both values
                val_ij = tl.load(addr_ij)
                val_ji = tl.load(addr_ji)
                
                # Store swapped
                tl.store(addr_ij, val_ji)
                tl.store(addr_ji, val_ij)
    else:
        # For off-diagonal tiles (tile_row, tile_col) and (tile_col, tile_row)
        # Swap entire tiles with transpose
        for i in range(BLOCK_K):
            actual_row_upper = row_start + i
            actual_row_lower = col_start + i
            if actual_row_upper >= K or actual_row_lower >= K:
                break
            
            for j in range(BLOCK_K):
                actual_col_upper = col_start + j
                actual_col_lower = row_start + j
                if actual_col_upper >= K or actual_col_lower >= K:
                    continue
                
                # Upper tile (tile_row, tile_col): element at (actual_row_upper, actual_col_upper)
                # Lower tile (tile_col, tile_row): element at (actual_row_lower, actual_col_lower)
                # After transpose:
                #   Upper[i,j] should contain what was at Lower[j,i] = (col_start+j, row_start+i)
                #   Lower[i,j] should contain what was at Upper[j,i] = (row_start+j, col_start+i)
                
                # For transpose: A[i,j] <-> A[j,i]
                # Upper tile element (row_start+i, col_start+j) goes to (col_start+j, row_start+i)
                # which is in the lower tile
                
                addr_upper = base_ptr + actual_row_upper * stride_row + actual_col_upper * stride_col
                addr_lower = base_ptr + actual_col_upper * stride_row + actual_row_upper * stride_col
                
                val_upper = tl.load(addr_upper)
                val_lower = tl.load(addr_lower)
                
                tl.store(addr_upper, val_lower)
                tl.store(addr_lower, val_upper)


@triton.jit  
def _in_place_transpose_kernel_v2(
    # Pointers
    ssm_states_ptr,      # [total_seqs, num_heads, K, K]
    cache_indices_ptr,   # [batch_size]
    # Strides (in elements, not bytes)
    stride_seq,
    stride_head,
    stride_row,
    stride_col,
    # Dimensions
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized in-place transpose kernel.
    
    Grid: (batch_size, num_heads)
    Each program handles one complete K x K matrix.
    Uses vectorized loads/stores for better memory throughput.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Load sequence index
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    
    # Skip padding
    if seq_idx < 0:
        return
    
    # Base pointer for this (seq, head)
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Process upper triangle: swap (i, j) with (j, i) for all i < j
    # Total pairs: K * (K - 1) / 2
    # Distribute across threads
    
    num_pairs = K * (K - 1) // 2
    
    # Each iteration processes BLOCK_K pairs
    for pair_base in range(0, num_pairs, BLOCK_K):
        pair_offsets = pair_base + tl.arange(0, BLOCK_K)
        mask = pair_offsets < num_pairs
        
        # Convert linear pair index to (i, j) coordinates where i < j
        # pair_idx = j * (j - 1) / 2 + i
        # Solve for j: j = floor((1 + sqrt(1 + 8 * pair_idx)) / 2)
        pair_idx_float = pair_offsets.to(tl.float32)
        j_float = (1.0 + tl.sqrt(1.0 + 8.0 * pair_idx_float)) / 2.0
        j = j_float.to(tl.int32)
        
        # Handle boundary case where j is slightly off due to floating point
        # Adjust j if needed
        j_check = j * (j - 1) // 2
        j = tl.where(j_check > pair_offsets, j - 1, j)
        
        # Calculate i from j
        i = pair_offsets - j * (j - 1) // 2
        
        # Calculate memory addresses
        addr_ij = base_ptr + i * stride_row + j * stride_col
        addr_ji = base_ptr + j * stride_row + i * stride_col
        
        # Load values
        val_ij = tl.load(addr_ij, mask=mask)
        val_ji = tl.load(addr_ji, mask=mask)
        
        # Store swapped values
        tl.store(addr_ij, val_ji, mask=mask)
        tl.store(addr_ji, val_ij, mask=mask)


@triton.jit
def _in_place_transpose_kernel_tiled(
    # Pointers
    ssm_states_ptr,
    cache_indices_ptr,
    # Strides
    stride_seq,
    stride_head,
    stride_row,
    stride_col,
    # Dimensions
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    Tile-based in-place transpose for better cache locality.
    
    Grid: (batch_size, num_heads, num_tile_pairs)
    Each program handles one tile or tile-pair.
    
    Uses vectorized operations within each tile for better memory throughput.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_pair_idx = tl.program_id(2)
    
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    if seq_idx < 0:
        return
    
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Number of tiles per dimension
    num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE
    
    # Convert tile_pair_idx to (tile_i, tile_j) in upper triangle including diagonal
    tile_pair_float = tile_pair_idx
    tile_j_float = (-1.0 + tl.sqrt(1.0 + 8.0 * tile_pair_float)) / 2.0
    tile_j = tile_j_float.to(tl.int32)
    
    # Adjust for boundary
    check = (tile_j + 1) * tile_j // 2
    tile_j = tl.where(check > tile_pair_idx, tile_j - 1, tile_j)
    tile_j = tl.where(tile_j < 0, 0, tile_j)
    
    tile_i = tile_pair_idx - tile_j * (tile_j + 1) // 2
    
    # Tile boundaries
    row_start = tile_i * TILE_SIZE
    col_start = tile_j * TILE_SIZE
    
    # Create offsets for vectorized access
    row_offs = tl.arange(0, TILE_SIZE)
    col_offs = tl.arange(0, TILE_SIZE)
    
    is_diagonal = tile_i == tile_j
    
    # Process row by row with vectorized column access
    for local_i in range(TILE_SIZE):
        global_i = row_start + local_i
        
        # Skip if out of bounds
        if global_i < K:
            if is_diagonal:
                # Diagonal tile: only process upper triangle (local_j > local_i)
                # Create mask for upper triangle
                col_indices = col_start + col_offs
                # For diagonal tile: we swap (global_i, col) with (col, global_i)
                # Only do this when col > global_i
                upper_mask = (col_indices > global_i) & (col_indices < K)
                
                if upper_mask.to(tl.int32).sum() > 0:
                    # Load row elements: positions (global_i, col_start:col_start+TILE_SIZE)
                    addr_row = base_ptr + global_i * stride_row + col_indices * stride_col
                    # Load column elements: positions (col_start:col_start+TILE_SIZE, global_i)  
                    addr_col = base_ptr + col_indices * stride_row + global_i * stride_col
                    
                    val_row = tl.load(addr_row, mask=upper_mask, other=0.0)
                    val_col = tl.load(addr_col, mask=upper_mask, other=0.0)
                    
                    tl.store(addr_row, val_col, mask=upper_mask)
                    tl.store(addr_col, val_row, mask=upper_mask)
            else:
                # Off-diagonal tile: swap entire tile with its transpose
                # Tile at (tile_i, tile_j) swaps with tile at (tile_j, tile_i)
                # Element (row_start+local_i, col_start+col_offs) <-> (col_start+col_offs, row_start+local_i)
                
                col_indices = col_start + col_offs
                col_mask = col_indices < K
                
                # Upper tile element: (global_i, col_indices)
                addr_upper = base_ptr + global_i * stride_row + col_indices * stride_col
                # Lower tile element: (col_indices, global_i)
                addr_lower = base_ptr + col_indices * stride_row + global_i * stride_col
                
                val_upper = tl.load(addr_upper, mask=col_mask, other=0.0)
                val_lower = tl.load(addr_lower, mask=col_mask, other=0.0)
                
                tl.store(addr_upper, val_lower, mask=col_mask)
                tl.store(addr_lower, val_upper, mask=col_mask)


@triton.jit
def _in_place_transpose_kernel_v3(
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
    BLOCK_PAIRS: tl.constexpr,
):
    """
    Most optimized version: each SM handles multiple (batch, head) combinations.
    
    Grid: (batch_size * num_heads * num_pair_blocks,)
    Maximizes SM utilization by having fine-grained parallelism.
    """
    pid = tl.program_id(0)
    
    # Calculate number of pair blocks per head
    num_pairs = K * (K - 1) // 2
    num_pair_blocks = (num_pairs + BLOCK_PAIRS - 1) // BLOCK_PAIRS
    
    # Decompose pid into (batch_idx, head_idx, pair_block_idx)
    pair_block_idx = pid % num_pair_blocks
    temp = pid // num_pair_blocks
    head_idx = temp % num_heads
    batch_idx = temp // num_heads
    
    # Load sequence index
    seq_idx = tl.load(cache_indices_ptr + batch_idx)
    
    # Skip padding
    if seq_idx < 0:
        return
    
    # Base pointer
    base_ptr = ssm_states_ptr + seq_idx * stride_seq + head_idx * stride_head
    
    # Process this block of pairs
    pair_base = pair_block_idx * BLOCK_PAIRS
    pair_offsets = pair_base + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < num_pairs
    
    # Convert to (i, j) coordinates
    pair_idx_float = pair_offsets.to(tl.float32)
    j_float = (1.0 + tl.sqrt(1.0 + 8.0 * pair_idx_float)) / 2.0
    j = j_float.to(tl.int32)
    j_check = j * (j - 1) // 2
    j = tl.where(j_check > pair_offsets, j - 1, j)
    i = pair_offsets - j * (j - 1) // 2
    
    # Load, swap, store
    addr_ij = base_ptr + i * stride_row + j * stride_col
    addr_ji = base_ptr + j * stride_row + i * stride_col
    
    val_ij = tl.load(addr_ij, mask=mask)
    val_ji = tl.load(addr_ji, mask=mask)
    
    tl.store(addr_ij, val_ji, mask=mask)
    tl.store(addr_ji, val_ij, mask=mask)


def in_place_transpose_indexed(
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    version: str = "v5",  # v5 is fastest (4-5x faster than PyTorch)
) -> None:
    """
    In-place transpose the last two dimensions of ssm_states for selected indices.
    
    Args:
        ssm_states: Tensor of shape [total_seqs, num_heads, K, V] where K == V
        cache_indices: Tensor of shape [batch_size] with indices into ssm_states
        version: Kernel version to use ("v2", "v3", or "tiled")
    
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
    
    num_pairs = K * (K - 1) // 2
    
    if version == "v2":
        # Version 2: each block handles one (batch, head)
        BLOCK_K = min(256, triton.next_power_of_2(num_pairs))
        grid = (batch_size, num_heads)
        
        _in_place_transpose_kernel_v2[grid](
            ssm_states,
            cache_indices,
            stride_seq,
            stride_head,
            stride_row,
            stride_col,
            K=K,
            BLOCK_K=BLOCK_K,
        )
    elif version == "v3":
        # Version 3: fine-grained parallelism across all dimensions
        BLOCK_PAIRS = 256
        num_pair_blocks = (num_pairs + BLOCK_PAIRS - 1) // BLOCK_PAIRS
        grid = (batch_size * num_heads * num_pair_blocks,)
        
        _in_place_transpose_kernel_v3[grid](
            ssm_states,
            cache_indices,
            stride_seq,
            stride_head,
            stride_row,
            stride_col,
            num_heads=num_heads,
            K=K,
            BLOCK_PAIRS=BLOCK_PAIRS,
        )
    elif version == "tiled":
        # Tiled version for better cache locality
        TILE_SIZE = 32  # 32x32 tiles fit well in L1 cache
        num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE
        num_tile_pairs = num_tiles * (num_tiles + 1) // 2
        grid = (batch_size, num_heads, num_tile_pairs)
        
        _in_place_transpose_kernel_tiled[grid](
            ssm_states,
            cache_indices,
            stride_seq,
            stride_head,
            stride_row,
            stride_col,
            K=K,
            TILE_SIZE=TILE_SIZE,
        )
    elif version == "v4":
        # Version 4: Row-wise vectorized with better coalescing
        BLOCK_SIZE = 8  # Process 8 rows per program
        num_row_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (batch_size * num_heads * num_row_blocks,)
        
        _in_place_transpose_kernel_v4[grid](
            ssm_states,
            cache_indices,
            stride_seq,
            stride_head,
            stride_row,
            stride_col,
            num_heads=num_heads,
            K=K,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif version == "v5":
        # Version 5: Tile-based with 2D vectorized operations
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
    else:
        raise ValueError(f"Unknown version: {version}")


def in_place_transpose_indexed_ref(
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> None:
    """
    Reference implementation using PyTorch operations (for correctness testing).
    
    Note: This is NOT in-place - it creates intermediate tensors.
    """
    valid_mask = cache_indices >= 0
    valid_indices = cache_indices[valid_mask]
    
    if valid_indices.numel() > 0:
        # Read, transpose, write back
        data = ssm_states[valid_indices]  # [batch, heads, K, V]
        data_transposed = data.transpose(-1, -2).contiguous()
        ssm_states[valid_indices] = data_transposed
