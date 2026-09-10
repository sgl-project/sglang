"""Reuse metadata derived from identical MTP draft-step inputs."""

from __future__ import annotations

from sglang.srt.utils import is_hip

_is_hip = is_hip()


class DSAMetadataSiblingMixin:
    pass

    def _copy_base_replay_buffers(self, bs, metadata, precomputed, forward_mode):
        # Track whether fused kernel succeeded
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        if not _is_hip:
            try:
                from sglang.kernels.ops.attention.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.dsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.dsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.dsa_cache_seqlens_int32,
                    metadata.dsa_seqlens_expanded,
                    metadata.dsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print(
                    "Warning: Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print(
                    f"Warning: Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if the fused kernel is unavailable
        # or fails at runtime.
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            if forward_mode.is_decode_or_idle():
                # Decode mode
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)

            elif forward_mode.is_target_verify():
                # Target verify mode
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)

            # Copy DSA cu_seqlens
            size = precomputed.seqlens_expanded_size
            metadata.dsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.dsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)
