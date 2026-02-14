"""
Verification utilities for NSA backend fused metadata copy operations.

This module contains verification code to ensure that fused metadata copy kernels
produce the same results as individual copy operations.
"""

import torch


def verify_single_backend_fused_metadata_copy(
    metadata,
    precomputed,
    forward_mode,
    bs,
    flashmla_num_splits_src=None,
    flashmla_metadata_src=None,
    flashmla_num_splits_dst=None,
    flashmla_metadata_dst=None,
):
    """
    Verify that the fused metadata copy kernel produces the same results as individual copies.

    Args:
        metadata: The NSA metadata object containing destination tensors
        precomputed: The precomputed metadata containing source tensors
        forward_mode: The forward mode (decode, target_verify, or draft_extend)
        bs: Batch size
        flashmla_num_splits_src: Source FlashMLA num_splits tensor (optional)
        flashmla_metadata_src: Source FlashMLA metadata tensor (optional)
        flashmla_num_splits_dst: Destination FlashMLA num_splits tensor (optional)
        flashmla_metadata_dst: Destination FlashMLA metadata tensor (optional)

    Raises:
        RuntimeError: If verification fails (tensors don't match)
    """
    # Clone destination tensors to preserve fused kernel results
    fused_cache_seqlens = metadata.cache_seqlens_int32.clone()
    fused_cu_seqlens_k = metadata.cu_seqlens_k.clone()
    fused_page_table_1 = metadata.page_table_1.clone()
    fused_nsa_cache_seqlens = metadata.nsa_cache_seqlens_int32.clone()
    fused_nsa_seqlens_expanded = metadata.nsa_seqlens_expanded.clone()
    fused_nsa_cu_seqlens_k = metadata.nsa_cu_seqlens_k.clone()
    fused_real_page_table = (
        metadata.real_page_table.clone()
        if precomputed.real_page_table is not None
        else None
    )
    fused_flashmla_num_splits = None
    fused_flashmla_metadata = None
    if precomputed.flashmla_metadata is not None:
        fused_flashmla_num_splits = flashmla_num_splits_dst.clone()
        fused_flashmla_metadata = flashmla_metadata_dst.clone()

    # Create reference tensors (zeroed out)
    ref_cache_seqlens = torch.zeros_like(metadata.cache_seqlens_int32)
    ref_cu_seqlens_k = torch.zeros_like(metadata.cu_seqlens_k)
    ref_page_table_1 = torch.zeros_like(metadata.page_table_1)
    ref_nsa_cache_seqlens = torch.zeros_like(metadata.nsa_cache_seqlens_int32)
    ref_nsa_seqlens_expanded = torch.zeros_like(metadata.nsa_seqlens_expanded)
    ref_nsa_cu_seqlens_k = torch.zeros_like(metadata.nsa_cu_seqlens_k)
    ref_real_page_table = (
        torch.zeros_like(metadata.real_page_table)
        if precomputed.real_page_table is not None
        else None
    )
    ref_flashmla_num_splits = None
    ref_flashmla_metadata = None
    if precomputed.flashmla_metadata is not None:
        ref_flashmla_num_splits = torch.zeros_like(flashmla_num_splits_dst)
        ref_flashmla_metadata = torch.zeros_like(flashmla_metadata_dst)

    # Run individual copy operations (reference implementation)
    ref_cache_seqlens.copy_(precomputed.cache_seqlens)
    ref_cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

    if forward_mode.is_decode_or_idle():
        # Decode mode
        ref_page_table_1[:, : precomputed.max_len].copy_(precomputed.page_indices)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)
    elif forward_mode.is_target_verify():
        # Target verify mode
        ref_page_table_1[:, : precomputed.max_seqlen_k].copy_(precomputed.page_indices)
        ref_nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)
    elif forward_mode.is_draft_extend():
        # Draft extend mode
        rows = precomputed.page_indices.shape[0]
        cols = precomputed.max_seqlen_k
        ref_page_table_1[:rows, :cols].copy_(precomputed.page_indices)
        size = precomputed.seqlens_expanded_size
        ref_nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
        ref_nsa_cache_seqlens[:size].copy_(precomputed.nsa_cache_seqlens)

    # Copy NSA cu_seqlens
    size = precomputed.seqlens_expanded_size
    ref_nsa_cu_seqlens_k[1 : 1 + size].copy_(precomputed.nsa_cu_seqlens_k[1 : 1 + size])

    # Copy real page table
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        ref_real_page_table[:rows, :cols].copy_(precomputed.real_page_table)

    # Copy FlashMLA metadata
    if precomputed.flashmla_metadata is not None:
        size = precomputed.seqlens_expanded_size
        ref_flashmla_num_splits[: size + 1].copy_(flashmla_num_splits_src[: size + 1])
        ref_flashmla_metadata.copy_(flashmla_metadata_src)

    # Compare results and crash if inconsistent
    def check_tensor_equal(name, fused, ref):
        if not torch.equal(fused, ref):
            max_diff = (fused.float() - ref.float()).abs().max().item()
            mismatched_elements = (fused != ref).sum().item()
            total_elements = fused.numel()
            raise RuntimeError(
                f"FUSED METADATA COPY VERIFICATION FAILED!\n"
                f"Tensor: {name}\n"
                f"Max difference: {max_diff}\n"
                f"Mismatched elements: {mismatched_elements}/{total_elements}\n"
                f"Fused shape: {fused.shape}, Ref shape: {ref.shape}\n"
                f"Forward mode: {forward_mode}, bs={bs}\n"
                f"The fused kernel produces different results than individual copies.\n"
                f"This indicates a bug in the fused metadata copy kernel."
            )

    # Verify all tensors (only compare the slices that were actually updated)
    check_tensor_equal("cache_seqlens", fused_cache_seqlens, ref_cache_seqlens)
    check_tensor_equal("cu_seqlens_k", fused_cu_seqlens_k, ref_cu_seqlens_k)

    # Compare page_table_1 only for the region that was updated
    if forward_mode.is_decode_or_idle():
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:, : precomputed.max_len],
            ref_page_table_1[:, : precomputed.max_len],
        )
    elif forward_mode.is_target_verify():
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:, : precomputed.max_seqlen_k],
            ref_page_table_1[:, : precomputed.max_seqlen_k],
        )
    elif forward_mode.is_draft_extend():
        rows = precomputed.page_indices.shape[0]
        cols = precomputed.max_seqlen_k
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:rows, :cols],
            ref_page_table_1[:rows, :cols],
        )

    # Compare nsa_cache_seqlens only for the region that was updated
    if forward_mode.is_decode_or_idle():
        check_tensor_equal(
            "nsa_cache_seqlens",
            fused_nsa_cache_seqlens,
            ref_nsa_cache_seqlens,
        )
    else:  # TARGET_VERIFY or DRAFT_EXTEND
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "nsa_cache_seqlens",
            fused_nsa_cache_seqlens[:size],
            ref_nsa_cache_seqlens[:size],
        )

    # Compare nsa_seqlens_expanded only for TARGET_VERIFY and DRAFT_EXTEND
    if forward_mode.is_target_verify() or forward_mode.is_draft_extend():
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "nsa_seqlens_expanded",
            fused_nsa_seqlens_expanded[:size],
            ref_nsa_seqlens_expanded[:size],
        )

    # Compare nsa_cu_seqlens_k only for the region that was updated
    size = precomputed.seqlens_expanded_size
    check_tensor_equal(
        "nsa_cu_seqlens_k",
        fused_nsa_cu_seqlens_k[: 1 + size],
        ref_nsa_cu_seqlens_k[: 1 + size],
    )

    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        check_tensor_equal(
            "real_page_table",
            fused_real_page_table[:rows, :cols],
            ref_real_page_table[:rows, :cols],
        )

    if precomputed.flashmla_metadata is not None:
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "flashmla_num_splits",
            fused_flashmla_num_splits[: size + 1],
            ref_flashmla_num_splits[: size + 1],
        )
        check_tensor_equal(
            "flashmla_metadata",
            fused_flashmla_metadata,
            ref_flashmla_metadata,
        )


def verify_multi_backend_fused_metadata_copy(
    metadata0,
    metadata1,
    metadata2,
    precomputed,
    bs,
    flashmla_num_splits_src=None,
    flashmla_metadata_src=None,
):
    """
    Verify that the multi-backend fused metadata copy kernel produces the same results
    as individual copies for all three backends.

    Args:
        metadata0: The NSA metadata object for backend 0
        metadata1: The NSA metadata object for backend 1
        metadata2: The NSA metadata object for backend 2
        precomputed: The precomputed metadata containing source tensors
        bs: Batch size
        flashmla_num_splits_src: Source FlashMLA num_splits tensor (optional)
        flashmla_metadata_src: Source FlashMLA metadata tensor (optional)

    Raises:
        RuntimeError: If verification fails (tensors don't match)
    """
    # Clone destination tensors to preserve fused kernel results
    fused_results = []
    for idx, metadata in enumerate([metadata0, metadata1, metadata2]):
        fused_cache_seqlens = metadata.cache_seqlens_int32.clone()
        fused_cu_seqlens_k = metadata.cu_seqlens_k.clone()
        fused_page_table_1 = metadata.page_table_1.clone()
        fused_nsa_cache_seqlens = metadata.nsa_cache_seqlens_int32.clone()
        fused_nsa_cu_seqlens_k = metadata.nsa_cu_seqlens_k.clone()
        fused_real_page_table = (
            metadata.real_page_table.clone()
            if precomputed.real_page_table is not None
            else None
        )
        fused_flashmla_num_splits = None
        fused_flashmla_metadata = None
        if precomputed.flashmla_metadata is not None:
            fused_flashmla_num_splits = metadata.flashmla_metadata.num_splits.clone()
            fused_flashmla_metadata = (
                metadata.flashmla_metadata.flashmla_metadata.clone()
            )

        fused_results.append(
            {
                "cache_seqlens": fused_cache_seqlens,
                "cu_seqlens_k": fused_cu_seqlens_k,
                "page_table_1": fused_page_table_1,
                "nsa_cache_seqlens": fused_nsa_cache_seqlens,
                "nsa_cu_seqlens_k": fused_nsa_cu_seqlens_k,
                "real_page_table": fused_real_page_table,
                "flashmla_num_splits": fused_flashmla_num_splits,
                "flashmla_metadata": fused_flashmla_metadata,
            }
        )

    # Run individual copy operations for each backend (reference implementation)
    ref_results = []
    for idx in range(3):
        metadata = [metadata0, metadata1, metadata2][idx]

        # Create reference tensors (zeroed out)
        ref_cache_seqlens = torch.zeros_like(metadata.cache_seqlens_int32)
        ref_cu_seqlens_k = torch.zeros_like(metadata.cu_seqlens_k)
        ref_page_table_1 = torch.zeros_like(metadata.page_table_1)
        ref_nsa_cache_seqlens = torch.zeros_like(metadata.nsa_cache_seqlens_int32)
        ref_nsa_cu_seqlens_k = torch.zeros_like(metadata.nsa_cu_seqlens_k)
        ref_real_page_table = (
            torch.zeros_like(metadata.real_page_table)
            if precomputed.real_page_table is not None
            else None
        )
        ref_flashmla_num_splits = None
        ref_flashmla_metadata = None
        if precomputed.flashmla_metadata is not None:
            ref_flashmla_num_splits = torch.zeros_like(
                metadata.flashmla_metadata.num_splits
            )
            ref_flashmla_metadata = torch.zeros_like(
                metadata.flashmla_metadata.flashmla_metadata
            )

        # Copy operations (decode mode)
        ref_cache_seqlens.copy_(precomputed.cache_seqlens)
        ref_cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])
        ref_page_table_1[:, : precomputed.max_len].copy_(precomputed.page_indices)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)

        # Copy NSA cu_seqlens
        size = precomputed.seqlens_expanded_size
        ref_nsa_cu_seqlens_k[1 : 1 + size].copy_(
            precomputed.nsa_cu_seqlens_k[1 : 1 + size]
        )

        # Copy real page table
        if precomputed.real_page_table is not None:
            rows, cols = precomputed.real_page_table.shape
            ref_real_page_table[:rows, :cols].copy_(precomputed.real_page_table)

        # Copy FlashMLA metadata
        if precomputed.flashmla_metadata is not None:
            ref_flashmla_num_splits[: size + 1].copy_(
                flashmla_num_splits_src[: size + 1]
            )
            ref_flashmla_metadata.copy_(flashmla_metadata_src)

        ref_results.append(
            {
                "cache_seqlens": ref_cache_seqlens,
                "cu_seqlens_k": ref_cu_seqlens_k,
                "page_table_1": ref_page_table_1,
                "nsa_cache_seqlens": ref_nsa_cache_seqlens,
                "nsa_cu_seqlens_k": ref_nsa_cu_seqlens_k,
                "real_page_table": ref_real_page_table,
                "flashmla_num_splits": ref_flashmla_num_splits,
                "flashmla_metadata": ref_flashmla_metadata,
            }
        )

    # Compare results for all 3 backends
    def check_tensor_equal(backend_idx, name, fused, ref):
        if not torch.equal(fused, ref):
            max_diff = (fused.float() - ref.float()).abs().max().item()
            mismatched_elements = (fused != ref).sum().item()
            total_elements = fused.numel()
            raise RuntimeError(
                f"MULTI-BACKEND FUSED METADATA COPY VERIFICATION FAILED!\n"
                f"Backend: {backend_idx}\n"
                f"Tensor: {name}\n"
                f"Max difference: {max_diff}\n"
                f"Mismatched elements: {mismatched_elements}/{total_elements}\n"
                f"Fused shape: {fused.shape}, Ref shape: {ref.shape}\n"
                f"Batch size: {bs}\n"
                f"The multi-backend fused kernel produces different results than individual copies.\n"
                f"This indicates a bug in the fused metadata copy kernel."
            )

    # Verify all tensors for all 3 backends (multi-backend is DECODE mode only)
    for idx in range(3):
        fused = fused_results[idx]
        ref = ref_results[idx]

        check_tensor_equal(
            idx,
            "cache_seqlens",
            fused["cache_seqlens"],
            ref["cache_seqlens"],
        )
        check_tensor_equal(
            idx,
            "cu_seqlens_k",
            fused["cu_seqlens_k"],
            ref["cu_seqlens_k"],
        )
        # Multi-backend is DECODE mode only, so compare only [:, :max_len]
        check_tensor_equal(
            idx,
            "page_table_1",
            fused["page_table_1"][:, : precomputed.max_len],
            ref["page_table_1"][:, : precomputed.max_len],
        )
        check_tensor_equal(
            idx,
            "nsa_cache_seqlens",
            fused["nsa_cache_seqlens"],
            ref["nsa_cache_seqlens"],
        )
        # DECODE mode uses bs for nsa_cu_seqlens_k size
        check_tensor_equal(
            idx,
            "nsa_cu_seqlens_k",
            fused["nsa_cu_seqlens_k"][: bs + 1],
            ref["nsa_cu_seqlens_k"][: bs + 1],
        )

        if precomputed.real_page_table is not None:
            rows, cols = precomputed.real_page_table.shape
            check_tensor_equal(
                idx,
                "real_page_table",
                fused["real_page_table"][:rows, :cols],
                ref["real_page_table"][:rows, :cols],
            )

        if precomputed.flashmla_metadata is not None:
            # DECODE mode uses bs + 1 for flashmla_num_splits
            check_tensor_equal(
                idx,
                "flashmla_num_splits",
                fused["flashmla_num_splits"][: bs + 1],
                ref["flashmla_num_splits"][: bs + 1],
            )
            check_tensor_equal(
                idx,
                "flashmla_metadata",
                fused["flashmla_metadata"],
                ref["flashmla_metadata"],
            )
