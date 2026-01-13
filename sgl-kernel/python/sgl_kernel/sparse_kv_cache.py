"""
Sparse KV Cache Manager with Bitmap-based Residence Tracking

This module provides efficient GPU hot buffer management for sparse attention using:
1. Bitmap for O(1) residence checking (vs O(H) linear scan)
2. Token-to-GPU location mapping for O(1) location lookup
3. GPU-location-to-token reverse mapping for eviction
4. Triton JIT kernels for easy iteration and optimization

Data Structures:
- residence_bitmap: [max_tokens] - 1 if token is on GPU, 0 otherwise
- token_to_gpu_loc: [max_tokens] - GPU location for each token (-1 if not resident)
- gpu_loc_to_token: [hot_buffer_size] - Token at each GPU location (-1 if empty)
- gpu_loc_available: [hot_buffer_size] - 1 if GPU location is available for eviction

Usage:
    manager = SparseCacheManager(max_tokens=4096, hot_buffer_size=512, device="cuda")
    
    # Process top-k for a request
    top_k_gpu_locs, copy_info = manager.process_topk(top_k_indices)
    
    # Perform actual data copies
    manager.execute_copies(cpu_cache, gpu_cache, copy_info, item_size_bytes)
"""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Tuple, Optional


@triton.jit
def sparse_cache_check_hits_kernel(
    # Inputs
    top_k_indices_ptr,      # [K] tokens we need
    residence_bitmap_ptr,   # [N] 1 if token on GPU
    token_to_gpu_loc_ptr,   # [N] GPU location for each token
    # Outputs
    top_k_gpu_locs_ptr,     # [K] GPU locations for top_k (-1 for misses)
    hit_mask_ptr,           # [K] 1 if hit, 0 if miss
    num_misses_ptr,         # [1] atomic counter for misses
    miss_indices_ptr,       # [K] indices of misses in top_k (compacted)
    # Sizes
    top_k_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Check which top_k tokens are cache hits using bitmap.
    O(1) lookup per token instead of O(H) scan.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < top_k_size
    
    # Load top_k token indices
    token_indices = tl.load(top_k_indices_ptr + offsets, mask=mask, other=0)
    
    # O(1) bitmap lookup for residence check
    is_resident = tl.load(residence_bitmap_ptr + token_indices, mask=mask, other=0)
    
    # For hits, get GPU location
    gpu_locs = tl.load(token_to_gpu_loc_ptr + token_indices, mask=mask, other=-1)
    
    # Store results
    # For misses, gpu_loc will be -1 (set later after eviction assignment)
    result_locs = tl.where(is_resident == 1, gpu_locs, -1)
    tl.store(top_k_gpu_locs_ptr + offsets, result_locs, mask=mask)
    tl.store(hit_mask_ptr + offsets, is_resident, mask=mask)
    
    # Count and compact misses
    is_miss = (is_resident == 0) & mask
    # Use atomic to get position in miss list
    for i in range(BLOCK_SIZE):
        if tl.load(is_miss.to(tl.int1), i):  # Check if this position is a miss
            actual_offset = block_start + i
            if actual_offset < top_k_size:
                miss_pos = tl.atomic_add(num_misses_ptr, 1)
                tl.store(miss_indices_ptr + miss_pos, actual_offset)


@triton.jit
def sparse_cache_find_evictable_kernel(
    # Inputs
    gpu_loc_to_token_ptr,   # [H] token at each GPU location
    residence_bitmap_ptr,   # [N] for checking if token still needed
    top_k_indices_ptr,      # [K] tokens we need (to avoid evicting needed tokens)
    top_k_size: tl.constexpr,
    # Outputs  
    evictable_locs_ptr,     # [H] GPU locations that can be evicted (compacted)
    num_evictable_ptr,      # [1] count of evictable locations
    # Sizes
    hot_buffer_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Find GPU locations that can be evicted.
    A location is evictable if its token is not in top_k.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hot_buffer_size
    
    # Get token at each GPU location
    tokens = tl.load(gpu_loc_to_token_ptr + offsets, mask=mask, other=-1)
    
    # Check if token is in top_k (need to scan top_k for each token)
    # This is O(K) per GPU location, but K is typically small
    # Alternative: build a top_k bitmap, but for small K this is fine
    is_needed = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    for k in range(top_k_size):
        top_k_token = tl.load(top_k_indices_ptr + k)
        is_needed = tl.where(tokens == top_k_token, 1, is_needed)
    
    # Location is evictable if:
    # 1. It has a valid token (tokens >= 0)
    # 2. That token is not in top_k (is_needed == 0)
    is_evictable = (tokens >= 0) & (is_needed == 0) & mask
    
    # Compact evictable locations using atomic
    for i in range(BLOCK_SIZE):
        actual_offset = block_start + i
        if actual_offset < hot_buffer_size:
            token_val = tl.load(gpu_loc_to_token_ptr + actual_offset)
            needed = 0
            for k in range(top_k_size):
                if tl.load(top_k_indices_ptr + k) == token_val:
                    needed = 1
            if token_val >= 0 and needed == 0:
                pos = tl.atomic_add(num_evictable_ptr, 1)
                tl.store(evictable_locs_ptr + pos, actual_offset)


@triton.jit 
def sparse_cache_assign_and_update_kernel(
    # Inputs
    top_k_indices_ptr,      # [K] tokens we need
    miss_indices_ptr,       # [M] indices in top_k that are misses
    evictable_locs_ptr,     # [E] GPU locations available for eviction
    cache_cpu_locs_ptr,     # [N] CPU location for each token
    num_misses: tl.constexpr,
    # Outputs (modified in-place)
    top_k_gpu_locs_ptr,     # [K] GPU locations for top_k
    residence_bitmap_ptr,   # [N] update: set new tokens, clear evicted
    token_to_gpu_loc_ptr,   # [N] update: set new mappings
    gpu_loc_to_token_ptr,   # [H] update: set new tokens at locations
    copy_src_cpu_locs_ptr,  # [M] CPU locations to copy from
    copy_dst_gpu_locs_ptr,  # [M] GPU locations to copy to
    # Sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 3: Assign eviction slots to misses and update all mappings.
    Also prepares the copy list for CPU→GPU transfers.
    Updates bitmap in parallel - can overlap with data loading.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_misses
    
    # Each thread handles one miss
    miss_idx_in_topk = tl.load(miss_indices_ptr + offsets, mask=mask, other=0)
    
    # Get token for this miss
    token = tl.load(top_k_indices_ptr + miss_idx_in_topk, mask=mask, other=0)
    
    # Get assigned GPU location from evictable list (1:1 mapping)
    gpu_loc = tl.load(evictable_locs_ptr + offsets, mask=mask, other=-1)
    
    # Get CPU location for this token
    cpu_loc = tl.load(cache_cpu_locs_ptr + token, mask=mask, other=0)
    
    # Get old token at this GPU location (for clearing its bitmap entry)
    old_token = tl.load(gpu_loc_to_token_ptr + gpu_loc, mask=mask, other=-1)
    
    # === Update all mappings (can happen in parallel with data loading) ===
    
    # 1. Update top_k_gpu_locs output
    tl.store(top_k_gpu_locs_ptr + miss_idx_in_topk, gpu_loc, mask=mask)
    
    # 2. Clear old token from bitmap (if valid)
    old_token_valid = old_token >= 0
    tl.store(residence_bitmap_ptr + old_token, 0, mask=mask & old_token_valid)
    tl.store(token_to_gpu_loc_ptr + old_token, -1, mask=mask & old_token_valid)
    
    # 3. Set new token in bitmap
    tl.store(residence_bitmap_ptr + token, 1, mask=mask)
    tl.store(token_to_gpu_loc_ptr + token, gpu_loc, mask=mask)
    
    # 4. Update GPU location -> token mapping
    tl.store(gpu_loc_to_token_ptr + gpu_loc, token, mask=mask)
    
    # 5. Prepare copy list
    tl.store(copy_src_cpu_locs_ptr + offsets, cpu_loc, mask=mask)
    tl.store(copy_dst_gpu_locs_ptr + offsets, gpu_loc, mask=mask)


@triton.jit
def sparse_cache_fused_kernel(
    # Inputs
    top_k_indices_ptr,      # [K] tokens we need
    cache_cpu_locs_ptr,     # [N] CPU location for each token
    # State (modified in-place)
    residence_bitmap_ptr,   # [N] 1 if token on GPU
    token_to_gpu_loc_ptr,   # [N] GPU location for each token
    gpu_loc_to_token_ptr,   # [H] token at each GPU location
    # Outputs
    top_k_gpu_locs_ptr,     # [K] GPU locations for top_k
    copy_src_cpu_locs_ptr,  # [K] CPU locations to copy from
    copy_dst_gpu_locs_ptr,  # [K] GPU locations to copy to
    num_copies_ptr,         # [1] number of copies needed
    # Sizes
    top_k_size: tl.constexpr,
    hot_buffer_size: tl.constexpr,
    max_tokens: tl.constexpr,
):
    """
    Fused single-block kernel for small top_k sizes.
    Processes everything in one kernel launch for minimal overhead.
    
    Algorithm:
    1. Check hits using bitmap (O(1) per token)
    2. Find evictable GPU locations
    3. Assign evictions to misses
    4. Update all mappings and prepare copy list
    """
    # Use single thread for simplicity in fused version
    # (Multi-threaded version above for larger sizes)
    tid = tl.program_id(0)
    if tid != 0:
        return
    
    # Track misses and evictable locations
    num_misses = 0
    evict_ptr = 0
    
    # Phase 1: Check hits, collect misses
    for k in range(top_k_size):
        token = tl.load(top_k_indices_ptr + k)
        is_resident = tl.load(residence_bitmap_ptr + token)
        
        if is_resident == 1:
            # Hit - get GPU location
            gpu_loc = tl.load(token_to_gpu_loc_ptr + token)
            tl.store(top_k_gpu_locs_ptr + k, gpu_loc)
        else:
            # Miss - mark for later
            tl.store(top_k_gpu_locs_ptr + k, -1)  # Temporary marker
            num_misses += 1
    
    # Early exit if all hits
    if num_misses == 0:
        tl.store(num_copies_ptr, 0)
        return
    
    # Phase 2 & 3: Find evictable locations and assign to misses
    copy_idx = 0
    miss_idx = 0
    
    for k in range(top_k_size):
        gpu_loc = tl.load(top_k_gpu_locs_ptr + k)
        if gpu_loc == -1:  # This was a miss
            token = tl.load(top_k_indices_ptr + k)
            
            # Find an evictable GPU location
            found_evict = 0
            for h in range(hot_buffer_size):
                if found_evict == 0:
                    existing_token = tl.load(gpu_loc_to_token_ptr + h)
                    
                    # Check if this location's token is in top_k
                    is_needed = 0
                    for kk in range(top_k_size):
                        if tl.load(top_k_indices_ptr + kk) == existing_token:
                            is_needed = 1
                    
                    # Can evict if has valid token that's not needed, or is empty
                    can_evict = (existing_token == -1) | ((existing_token >= 0) & (is_needed == 0))
                    
                    if can_evict:
                        # Found evictable location
                        evict_loc = h
                        found_evict = 1
                        
                        # Clear old token's bitmap entry
                        if existing_token >= 0:
                            tl.store(residence_bitmap_ptr + existing_token, 0)
                            tl.store(token_to_gpu_loc_ptr + existing_token, -1)
                        
                        # Update mappings for new token
                        tl.store(residence_bitmap_ptr + token, 1)
                        tl.store(token_to_gpu_loc_ptr + token, evict_loc)
                        tl.store(gpu_loc_to_token_ptr + evict_loc, token)
                        
                        # Store result
                        tl.store(top_k_gpu_locs_ptr + k, evict_loc)
                        
                        # Add to copy list
                        cpu_loc = tl.load(cache_cpu_locs_ptr + token)
                        tl.store(copy_src_cpu_locs_ptr + copy_idx, cpu_loc)
                        tl.store(copy_dst_gpu_locs_ptr + copy_idx, evict_loc)
                        copy_idx += 1
    
    tl.store(num_copies_ptr, copy_idx)


@dataclass
class CopyInfo:
    """Information about copies needed after cache management."""
    src_cpu_locs: torch.Tensor  # [num_copies] CPU locations to copy from
    dst_gpu_locs: torch.Tensor  # [num_copies] GPU locations to copy to
    num_copies: int


class SparseCacheManager:
    """
    Manages GPU hot buffer for sparse attention with bitmap-based tracking.
    
    Provides O(1) residence checking instead of O(H) linear scan.
    """
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.int64,
    ):
        """
        Initialize the sparse cache manager.
        
        Args:
            max_tokens: Maximum number of tokens per request
            hot_buffer_size: Size of GPU hot buffer
            device: Device to use
            dtype: Data type for indices (int64)
        """
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.device = device
        self.dtype = dtype
        
        # Bitmap: 1 if token is on GPU, 0 otherwise
        self.residence_bitmap = torch.zeros(
            max_tokens, dtype=torch.int32, device=device
        )
        
        # Token -> GPU location mapping (-1 if not resident)
        self.token_to_gpu_loc = torch.full(
            (max_tokens,), -1, dtype=dtype, device=device
        )
        
        # GPU location -> Token mapping (-1 if empty)
        self.gpu_loc_to_token = torch.full(
            (hot_buffer_size,), -1, dtype=dtype, device=device
        )
        
        # Temporary buffers for kernel outputs
        self._copy_src = torch.zeros(hot_buffer_size, dtype=dtype, device=device)
        self._copy_dst = torch.zeros(hot_buffer_size, dtype=dtype, device=device)
        self._num_copies = torch.zeros(1, dtype=torch.int32, device=device)
    
    def reset(self):
        """Reset the cache manager state."""
        self.residence_bitmap.zero_()
        self.token_to_gpu_loc.fill_(-1)
        self.gpu_loc_to_token.fill_(-1)
    
    def process_topk(
        self,
        top_k_indices: torch.Tensor,
        cache_cpu_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, CopyInfo]:
        """
        Process top-k tokens for a request.
        
        Identifies hits/misses, assigns eviction slots, and prepares copy info.
        Updates internal state (bitmap, mappings) atomically.
        
        Args:
            top_k_indices: [K] Token indices that need to be in GPU
            cache_cpu_locations: [N] CPU location for each token
            
        Returns:
            top_k_gpu_locs: [K] GPU locations for all top_k tokens
            copy_info: Information about CPU→GPU copies needed
        """
        top_k_size = top_k_indices.numel()
        
        # Output tensor
        top_k_gpu_locs = torch.zeros(
            top_k_size, dtype=self.dtype, device=self.device
        )
        
        # Reset copy buffers
        self._num_copies.zero_()
        
        # Choose kernel based on size
        if top_k_size <= 64 and self.hot_buffer_size <= 256:
            # Use fused kernel for small sizes
            sparse_cache_fused_kernel[(1,)](
                top_k_indices,
                cache_cpu_locations,
                self.residence_bitmap,
                self.token_to_gpu_loc,
                self.gpu_loc_to_token,
                top_k_gpu_locs,
                self._copy_src,
                self._copy_dst,
                self._num_copies,
                top_k_size=top_k_size,
                hot_buffer_size=self.hot_buffer_size,
                max_tokens=self.max_tokens,
            )
        else:
            # Use multi-phase kernels for larger sizes
            self._process_topk_multiphase(
                top_k_indices, cache_cpu_locations, top_k_gpu_locs
            )
        
        num_copies = self._num_copies.item()
        
        return top_k_gpu_locs, CopyInfo(
            src_cpu_locs=self._copy_src[:num_copies].clone(),
            dst_gpu_locs=self._copy_dst[:num_copies].clone(),
            num_copies=num_copies,
        )
    
    def _process_topk_multiphase(
        self,
        top_k_indices: torch.Tensor,
        cache_cpu_locations: torch.Tensor,
        top_k_gpu_locs: torch.Tensor,
    ):
        """Multi-phase processing for larger sizes."""
        top_k_size = top_k_indices.numel()
        BLOCK_SIZE = 256
        
        # Temporary buffers
        hit_mask = torch.zeros(top_k_size, dtype=torch.int32, device=self.device)
        num_misses = torch.zeros(1, dtype=torch.int32, device=self.device)
        miss_indices = torch.zeros(top_k_size, dtype=torch.int64, device=self.device)
        
        # Phase 1: Check hits
        grid = ((top_k_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        sparse_cache_check_hits_kernel[grid](
            top_k_indices,
            self.residence_bitmap,
            self.token_to_gpu_loc,
            top_k_gpu_locs,
            hit_mask,
            num_misses,
            miss_indices,
            top_k_size=top_k_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        n_misses = num_misses.item()
        if n_misses == 0:
            return
        
        # Phase 2: Find evictable locations
        evictable_locs = torch.zeros(
            self.hot_buffer_size, dtype=torch.int64, device=self.device
        )
        num_evictable = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        grid = ((self.hot_buffer_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        sparse_cache_find_evictable_kernel[grid](
            self.gpu_loc_to_token,
            self.residence_bitmap,
            top_k_indices,
            top_k_size,
            evictable_locs,
            num_evictable,
            hot_buffer_size=self.hot_buffer_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Phase 3: Assign evictions and update mappings
        grid = ((n_misses + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        sparse_cache_assign_and_update_kernel[grid](
            top_k_indices,
            miss_indices,
            evictable_locs,
            cache_cpu_locations,
            n_misses,
            top_k_gpu_locs,
            self.residence_bitmap,
            self.token_to_gpu_loc,
            self.gpu_loc_to_token,
            self._copy_src,
            self._copy_dst,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        self._num_copies.fill_(n_misses)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        num_resident = self.residence_bitmap.sum().item()
        return {
            "num_resident": num_resident,
            "hot_buffer_size": self.hot_buffer_size,
            "occupancy": num_resident / self.hot_buffer_size if self.hot_buffer_size > 0 else 0,
        }


def execute_copies(
    cpu_cache: torch.Tensor,
    gpu_cache: torch.Tensor,
    copy_info: CopyInfo,
    item_size_elements: int,
):
    """
    Execute the CPU→GPU copies.
    
    Uses async copy for overlap with computation.
    
    Args:
        cpu_cache: CPU pinned memory buffer [N, item_size]
        gpu_cache: GPU memory buffer [H, item_size]
        copy_info: Copy information from process_topk
        item_size_elements: Number of elements per cache entry
    """
    if copy_info.num_copies == 0:
        return
    
    # Perform copies using advanced indexing
    # This leverages PyTorch's optimized copy kernels
    src_indices = copy_info.src_cpu_locs
    dst_indices = copy_info.dst_gpu_locs
    
    # Batch copy: gpu_cache[dst] = cpu_cache[src]
    gpu_cache[dst_indices] = cpu_cache[src_indices].to(gpu_cache.device, non_blocking=True)


# ============================================================================
# Alternative: Pure Python reference implementation for testing
# ============================================================================

def reference_sparse_cache_manager(
    top_k_indices: torch.Tensor,
    residence_bitmap: torch.Tensor,
    token_to_gpu_loc: torch.Tensor,
    gpu_loc_to_token: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Reference implementation for testing.
    
    Returns:
        top_k_gpu_locs, copy_src, copy_dst, num_copies
    """
    device = top_k_indices.device
    top_k_size = top_k_indices.numel()
    hot_buffer_size = gpu_loc_to_token.numel()
    
    # Convert to Python for reference implementation
    top_k = top_k_indices.cpu().tolist()
    bitmap = residence_bitmap.cpu().numpy().copy()
    tok_to_loc = token_to_gpu_loc.cpu().numpy().copy()
    loc_to_tok = gpu_loc_to_token.cpu().numpy().copy()
    cpu_locs = cache_cpu_locations.cpu().numpy()
    
    top_k_gpu_locs = []
    copy_src = []
    copy_dst = []
    
    # Build set of top_k for fast lookup
    top_k_set = set(top_k)
    
    for token in top_k:
        if bitmap[token] == 1:
            # Hit
            top_k_gpu_locs.append(tok_to_loc[token])
        else:
            # Miss - find evictable location
            evict_loc = -1
            for h in range(hot_buffer_size):
                existing_token = loc_to_tok[h]
                if existing_token == -1 or existing_token not in top_k_set:
                    evict_loc = h
                    break
            
            if evict_loc == -1:
                raise RuntimeError("No evictable location found")
            
            # Evict old token
            old_token = loc_to_tok[evict_loc]
            if old_token >= 0:
                bitmap[old_token] = 0
                tok_to_loc[old_token] = -1
            
            # Update mappings for new token
            bitmap[token] = 1
            tok_to_loc[token] = evict_loc
            loc_to_tok[evict_loc] = token
            
            top_k_gpu_locs.append(evict_loc)
            copy_src.append(cpu_locs[token])
            copy_dst.append(evict_loc)
    
    # Update state tensors
    residence_bitmap.copy_(torch.from_numpy(bitmap).to(device))
    token_to_gpu_loc.copy_(torch.from_numpy(tok_to_loc).to(device))
    gpu_loc_to_token.copy_(torch.from_numpy(loc_to_tok).to(device))
    
    return (
        torch.tensor(top_k_gpu_locs, dtype=torch.int64, device=device),
        torch.tensor(copy_src, dtype=torch.int64, device=device),
        torch.tensor(copy_dst, dtype=torch.int64, device=device),
        len(copy_src),
    )
