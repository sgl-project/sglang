"""
Paged Scatter KV Cache for Diff TP PD Disaggregation.

This module provides a paged scatter buffer implementation that mimics
the KV cache structure for efficient RDMA transfer.

Key Design:
1. Scatter buffer has the same layout as KV cache: List[Tensor] for each layer
2. Uses paged allocation like KV cache (free_pages list)
3. Prefill sends KV data using _send_kvcache_generic (same as same-TP transfer)
4. Scatter operation copies from scatter buffer pages to KV cache pages with head reordering

Memory Layout:
- k_buffer: List[Tensor], each [total_tokens, src_num_heads, head_dim]
- v_buffer: List[Tensor], each [total_tokens, src_num_heads, head_dim]
- Each prefill TP rank has its own set of buffers

Use case: When Prefill TP > Decode TP (e.g., TP4 -> TP2), we receive
KV data to scatter buffer and then scatter it to the correct head positions.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PagedScatterKVManager:

    def __init__(
        self,
        num_scatter_pages: int,
        page_size: int,
        src_num_heads: int,  # Prefill's heads per rank
        dst_num_heads: int,  # Decode's heads per rank
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype,
        device: str = "cuda",
        p_d_tp_size_ratio: int = 1,  # The Number of prefill ranks for one decode rank
    ):
        """
        Initialize the Paged Scatter KV Manager.

        Args:
            num_scatter_pages: Maximum number of pages in the scatter buffer pool.
            page_size: Number of tokens per page.
            src_num_heads: Number of KV heads per prefill TP rank.
            dst_num_heads: Number of KV heads per decode TP rank.
            head_dim: Dimension per attention head.
            num_layers: Number of transformer layers.
            dtype: Data type for KV cache.
            device: CUDA device string.
            p_d_tp_size_ratio: The Number of prefill ranks for one decode rank.
        """
        self.num_scatter_pages = num_scatter_pages
        self.max_pages = num_scatter_pages
        self.page_size = page_size
        self.src_num_heads = src_num_heads
        self.dst_num_heads = dst_num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device
        self.p_d_tp_size_ratio = p_d_tp_size_ratio

        # Handle fp8 types - store as uint8 like KV cache does
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        # Total tokens in scatter buffer
        self.total_tokens = num_scatter_pages * page_size

        # Calculate item length per page (for RDMA transfer)
        # This matches KV cache's kv_item_lens
        self.bytes_per_element = dtype.itemsize
        self.kv_item_len = page_size * src_num_heads * head_dim * self.bytes_per_element

        # Create buffers that match KV cache structure
        # Layout: [p_d_tp_size_ratio][num_layers] -> Tensor[total_tokens, src_num_heads, head_dim]
        self.k_buffers: List[List[torch.Tensor]] = []
        self.v_buffers: List[List[torch.Tensor]] = []

        # Create data pointer arrays for each TP rank (for RDMA registration)
        # Layout: [p_d_tp_size_ratio] -> [k_ptr_layer0, ..., k_ptr_layerN, v_ptr_layer0, ..., v_ptr_layerN]
        self.kv_data_ptrs_per_tp: List[List[int]] = []

        for tp_rank in range(p_d_tp_size_ratio):
            k_buffer_layers = []
            v_buffer_layers = []

            for layer_idx in range(num_layers):
                k_tensor = torch.zeros(
                    (self.total_tokens, src_num_heads, head_dim),
                    dtype=self.store_dtype,
                    device=device,
                )
                v_tensor = torch.zeros(
                    (self.total_tokens, src_num_heads, head_dim),
                    dtype=self.store_dtype,
                    device=device,
                )
                k_buffer_layers.append(k_tensor)
                v_buffer_layers.append(v_tensor)

            self.k_buffers.append(k_buffer_layers)
            self.v_buffers.append(v_buffer_layers)

            # Layout: [k0, k1, ..., kN, v0, v1, ..., vN]
            kv_ptrs = []
            for k in k_buffer_layers:
                kv_ptrs.append(k.data_ptr())
            for v in v_buffer_layers:
                kv_ptrs.append(v.data_ptr())
            self.kv_data_ptrs_per_tp.append(kv_ptrs)

        # Page allocation tracking (shared across all TP ranks)
        self._lock = threading.Lock()
        self._free_pages: List[int] = list(range(num_scatter_pages))

        # Request allocations: request_id -> RequestAllocation
        # RequestAllocation: (scatter_page_indices, kv_page_indices, num_pages)
        self._request_allocations: Dict[str, Tuple[List[int], torch.Tensor, int]] = {}

        # Create async stream for scatter operations
        self.scatter_stream = torch.cuda.Stream(device=device)

        # Calculate memory usage
        buffer_size_bytes = (
            p_d_tp_size_ratio
            * 2
            * num_layers
            * self.total_tokens
            * src_num_heads
            * head_dim
            * self.bytes_per_element
        )

        logger.info(
            f"PagedScatterKVManager initialized: "
            f"num_scatter_pages={num_scatter_pages}, page_size={page_size}, "
            f"total_tokens={self.total_tokens}, "
            f"src_heads={src_num_heads}, dst_heads={dst_num_heads}, "
            f"head_dim={head_dim}, layers={num_layers}, "
            f"p_d_tp_size_ratio={p_d_tp_size_ratio}, "
            f"kv_item_len={self.kv_item_len}, "
            f"buffer_size={buffer_size_bytes / 1024**3:.2f} GB"
        )

    def get_buffer_stats(self) -> Tuple[int, int, int]:
        """
        Get buffer usage statistics.

        Returns:
            Tuple of (allocated_pages, total_pages, free_pages)
        """
        with self._lock:
            free_count = len(self._free_pages)
            allocated = self.max_pages - free_count
            return (allocated, self.max_pages, free_count)

    def get_kv_data_ptrs(self, prefill_tp_rank: int) -> List[int]:
        """
        Get KV data pointers for a specific prefill TP rank.

        This is used by prefill to get dst_kv_ptrs for _send_kvcache_generic.

        Args:
            prefill_tp_rank: Local prefill TP rank (0, 1, ... p_d_tp_size_ratio-1)

        Returns:
            List of data pointers: [k0, k1, ..., kN, v0, v1, ..., vN]
        """
        if 0 <= prefill_tp_rank < self.p_d_tp_size_ratio:
            return self.kv_data_ptrs_per_tp[prefill_tp_rank]
        else:
            logger.warning(f"Invalid prefill_tp_rank: {prefill_tp_rank}")
            return []

    def get_kv_item_len(self) -> int:
        """Get the item length for RDMA transfer (bytes per page)."""
        return self.kv_item_len

    def try_allocate(
        self,
        request_id: str,
        num_pages: int,
        kv_page_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Try to allocate scatter buffer pages for a request.

        Args:
            request_id: Unique identifier for the request.
            num_pages: Number of pages needed.
            kv_page_indices: KV cache page indices (for later scatter).
                            Can be None initially and set via update_kv_indices.

        Returns:
            Tuple of (success, scatter_page_indices):
            - success: True if allocation succeeded
            - scatter_page_indices: List of allocated page indices in scatter buffer

        If success is False, the request should fallback to all-to-all mode.
        """
        if num_pages <= 0:
            return (False, None)

        with self._lock:
            if num_pages > len(self._free_pages):
                allocated_count = self.max_pages - len(self._free_pages)
                usage_percent = (allocated_count / self.max_pages) * 100

                # Find top holders
                holders = []
                for rid, (_, _, pages) in self._request_allocations.items():
                    holders.append((rid, pages))
                holders.sort(key=lambda x: x[1], reverse=True)
                top_holders = holders[:5]

                logger.warning(
                    f"Paged scatter ALLOC FAIL: request_id={request_id}, pages={num_pages}, "
                    f"available={len(self._free_pages)}, usage={usage_percent:.1f}% ({allocated_count}/{self.max_pages}). "
                    f"Top holders: {top_holders}. Fallback to all-to-all."
                )
                return (False, None)

            # Allocate pages from free list
            scatter_page_indices = self._free_pages[:num_pages]
            self._free_pages = self._free_pages[num_pages:]

            # Store allocation info
            self._request_allocations[request_id] = (
                scatter_page_indices,
                kv_page_indices,
                num_pages,
            )

            allocated_count = self.max_pages - len(self._free_pages)
            usage_percent = (allocated_count / self.max_pages) * 100
            logger.info(
                f"Paged scatter ALLOC OK: request_id={request_id}, "
                f"pages={num_pages}, usage={usage_percent:.1f}% ({allocated_count}/{self.max_pages})"
            )

            return (True, scatter_page_indices)

    def update_kv_indices(self, request_id: str, kv_page_indices: torch.Tensor) -> bool:
        """
        Update KV cache page indices for a request.

        This is called when actual KV indices are known (in init()).

        Args:
            request_id: Unique identifier for the request.
            kv_page_indices: KV cache page indices for scatter destination.

        Returns:
            True if update succeeded, False if request not found.
        """
        with self._lock:
            if request_id not in self._request_allocations:
                return False

            scatter_indices, _, num_pages = self._request_allocations[request_id]
            self._request_allocations[request_id] = (
                scatter_indices,
                kv_page_indices,
                num_pages,
            )
            return True

    def release(self, request_id: str) -> bool:
        """
        Release scatter buffer pages for a completed request.

        Args:
            request_id: Unique identifier for the request.

        Returns:
            True if release succeeded, False if request not found.
        """
        with self._lock:
            if request_id not in self._request_allocations:
                return False

            scatter_indices, _, num_pages = self._request_allocations.pop(request_id)

            # Return pages to free list
            self._free_pages.extend(scatter_indices)

            allocated_count = self.max_pages - len(self._free_pages)
            usage_percent = (allocated_count / self.max_pages) * 100

            logger.info(
                f"Paged scatter RELEASE: request_id={request_id}, pages={num_pages}, "
                f"usage={usage_percent:.1f}% ({allocated_count}/{self.max_pages})"
            )

            return True

    def get_request_info(
        self, request_id: str
    ) -> Optional[Tuple[List[int], torch.Tensor, int]]:
        """
        Get allocation info for a request.

        Args:
            request_id: Unique identifier for the request.

        Returns:
            Tuple of (scatter_page_indices, kv_page_indices, num_pages), or None.
        """
        with self._lock:
            return self._request_allocations.get(request_id)

    def get_packed_scatter_page_indices(self, request_id: str) -> Optional[bytes]:
        """
        Get packed scatter page indices for sending to prefill.

        Args:
            request_id: Unique identifier for the request.

        Returns:
            Packed bytes of scatter page indices (int32), or None if not found.
        """
        info = self.get_request_info(request_id)
        if info is None:
            return None

        scatter_indices, _, _ = info
        return np.array(scatter_indices, dtype=np.int32).tobytes()

    def scatter(
        self,
        request_id: str,
        dst_k_buffers: List[torch.Tensor],
        dst_v_buffers: List[torch.Tensor],
        prefill_tp_rank: int,
        sync: bool = True,
    ) -> bool:
        """
        Scatter received data from scatter buffer to KV cache.

        Args:
            request_id: Unique identifier for the request.
            dst_k_buffers: List of K cache tensors, one per layer.
            dst_v_buffers: List of V cache tensors, one per layer.
            prefill_tp_rank: Local prefill TP rank (0, 1, ...)
            sync: Whether to synchronize after scatter.

        Returns:
            True if scatter succeeded, False if request not found.
        """
        info = self.get_request_info(request_id)
        if info is None:
            logger.warning(f"Scatter: request {request_id} not found in allocations")
            return False

        scatter_page_indices, kv_page_indices, num_pages = info

        if kv_page_indices is None:
            logger.warning(f"Scatter: request {request_id} has no kv_page_indices")
            return False

        # Use actual kv_page_indices length as the number of pages to scatter
        # This handles cases where scatter_page_indices was pre-allocated with
        # an estimated size but actual kv_page_indices has fewer pages
        actual_num_pages = len(kv_page_indices)
        if actual_num_pages == 0:
            logger.debug(f"Scatter: actual_num_pages is 0 for request {request_id}")
            return True

        if actual_num_pages > len(scatter_page_indices):
            logger.error(
                f"Scatter: kv_page_indices ({actual_num_pages}) > scatter_page_indices ({len(scatter_page_indices)})"
            )
            return False

        # Truncate scatter_page_indices to match actual kv_page_indices length
        scatter_page_indices_to_use = scatter_page_indices[:actual_num_pages]

        # Calculate head offset for this TP rank
        dst_head_offset = prefill_tp_rank * self.src_num_heads
        head_end = dst_head_offset + self.src_num_heads

        # Validate head offset
        if head_end > self.dst_num_heads:
            logger.error(
                f"Scatter: head_end ({head_end}) > dst_num_heads ({self.dst_num_heads}). "
                f"prefill_tp_rank={prefill_tp_rank}, src_num_heads={self.src_num_heads}"
            )
            return False

        # Get source buffers for this TP rank
        src_k_buffers = self.k_buffers[prefill_tp_rank]
        src_v_buffers = self.v_buffers[prefill_tp_rank]

        # Ensure tensors are on correct device
        device = dst_k_buffers[0].device

        if not kv_page_indices.is_cuda:
            kv_page_indices = kv_page_indices.to(device)

        scatter_page_tensor = torch.tensor(
            scatter_page_indices_to_use, dtype=torch.int64, device=device
        )

        # Compute token indices
        # scatter_token_indices: which tokens to read from scatter buffer
        # dst_token_indices: which tokens to write to KV cache
        page_size = self.page_size

        scatter_page_expanded = scatter_page_tensor.unsqueeze(
            1
        )  # [actual_num_pages, 1]
        kv_page_expanded = kv_page_indices.unsqueeze(1).to(
            torch.int64
        )  # [actual_num_pages, 1]
        token_offsets = torch.arange(
            page_size, device=device, dtype=torch.int64
        ).unsqueeze(
            0
        )  # [1, page_size]

        scatter_token_indices = (
            scatter_page_expanded * page_size + token_offsets
        ).flatten()  # [num_tokens]
        dst_token_indices = (
            kv_page_expanded * page_size + token_offsets
        ).flatten()  # [num_tokens]

        # Validate indices to prevent out-of-bounds access
        # Note: .item() triggers synchronization, which may report previous async errors
        try:
            max_scatter_idx = scatter_token_indices.max().item()
            min_scatter_idx = scatter_token_indices.min().item()
            max_dst_idx = dst_token_indices.max().item()
            min_dst_idx = dst_token_indices.min().item()
        except Exception as e:
            logger.error(f"Scatter: failed to get min/max indices: {e}")
            return False

        if max_scatter_idx >= self.total_tokens or min_scatter_idx < 0:
            logger.error(
                f"Scatter index out of bounds: range=[{min_scatter_idx}, {max_scatter_idx}], "
                f"total_tokens={self.total_tokens}, scatter_page_indices={scatter_page_indices_to_use[:5]}"
            )
            return False

        if max_dst_idx >= dst_k_buffers[0].shape[0] or min_dst_idx < 0:
            logger.error(
                f"Dst index out of bounds: range=[{min_dst_idx}, {max_dst_idx}], "
                f"kv_cache_size={dst_k_buffers[0].shape[0]}, kv_page_indices={kv_page_indices[:5].tolist()}"
            )
            return False

        # Perform scatter for each layer
        stream_ctx = torch.cuda.stream(self.scatter_stream)

        # Note: Both scatter buffer and KV cache store fp8 as uint8
        # So we can copy directly without dtype conversion

        with stream_ctx:
            for layer_idx in range(self.num_layers):
                # Read from scatter buffer (stored as uint8 for fp8)
                src_k = src_k_buffers[layer_idx][
                    scatter_token_indices
                ]  # [num_tokens, src_heads, head_dim]
                src_v = src_v_buffers[layer_idx][scatter_token_indices]

                # Debug: log first layer info and check data
                if layer_idx == 0 and prefill_tp_rank == 0:
                    # Check if scatter buffer has non-zero data
                    src_k_sum = src_k.float().abs().sum().item()
                    logger.info(
                        f"Scatter debug: layer=0, tp_rank={prefill_tp_rank}, "
                        f"scatter_tokens={scatter_token_indices[:5].tolist()}, "
                        f"dst_tokens={dst_token_indices[:5].tolist()}, "
                        f"head_offset={dst_head_offset}, head_end={head_end}, "
                        f"src_k_shape={src_k.shape}, src_k_sum={src_k_sum:.4f}, "
                        f"dst_k_shape={dst_k_buffers[layer_idx].shape}"
                    )

                # Write to KV cache with head offset (also stored as uint8 for fp8)
                dst_k_buffers[layer_idx][
                    dst_token_indices, dst_head_offset:head_end, :
                ] = src_k
                dst_v_buffers[layer_idx][
                    dst_token_indices, dst_head_offset:head_end, :
                ] = src_v

        if sync:
            # Avoid blocking the main thread if possible, but for correctness we sync here
            # Ideally, we should use events or wait on the stream in the main loop
            self.scatter_stream.synchronize()

        logger.debug(
            f"Scatter completed: request={request_id}, tp_rank={prefill_tp_rank}, "
            f"actual_pages={actual_num_pages}, alloc_pages={num_pages}, head_offset={dst_head_offset}"
        )

        return True

    def scatter_all_tp_ranks(
        self,
        request_id: str,
        dst_k_buffers: List[torch.Tensor],
        dst_v_buffers: List[torch.Tensor],
        sync: bool = True,
    ) -> bool:
        """
        Scatter data from all prefill TP ranks to KV cache.

        Args:
            request_id: Unique identifier for the request.
            dst_k_buffers: List of K cache tensors, one per layer.
            dst_v_buffers: List of V cache tensors, one per layer.
            sync: Whether to synchronize after all scatters.

        Returns:
            True if all scatters succeeded.
        """
        for tp_rank in range(self.p_d_tp_size_ratio):
            # Only sync on last scatter
            do_sync = sync and (tp_rank == self.p_d_tp_size_ratio - 1)
            success = self.scatter(
                request_id=request_id,
                dst_k_buffers=dst_k_buffers,
                dst_v_buffers=dst_v_buffers,
                prefill_tp_rank=tp_rank,
                sync=do_sync,
            )
            if not success:
                return False

        return True
