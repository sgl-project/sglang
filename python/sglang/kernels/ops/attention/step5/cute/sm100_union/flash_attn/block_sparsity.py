"""
Computes block-sparse attention masks for Flex Attention.

This utility generates block sparsity patterns based on common attention masking
strategies (e.g., causal, sliding window). The resulting tensors define which
blocks are fully computed, which are partially computed (requiring a mask), and
which are skipped entirely. This is a temporary solution intended to be replaced
by a more robust preprocessing kernel in the future.
"""

from typing import Tuple, Optional, Callable, List, NamedTuple
import torch


# placeholder
Config = type("Config", (), {})


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor]
    full_block_idx: Optional[torch.Tensor]

    def __new_from_mlir_values__(self, values):
        if len(values) == 2:
            values = (*values, None, None)
        return BlockSparseTensors(*values)


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor] = None
    full_block_idx: Optional[torch.Tensor] = None


def _expand_sparsity_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
) -> torch.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = tensor.shape != expected_shape
    if not needs_expand:
        return tensor
    can_expand = all(map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape))
    if not can_expand:
        raise ValueError(
            f"{tensor_name} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
        )
    return tensor.expand(*expected_shape).contiguous()


def _check_and_expand_block(
    name: str,
    cnt: Optional[torch.Tensor],
    idx: Optional[torch.Tensor],
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must have dtype torch.int32")
    if cnt.device != idx.device:
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must be on the same device")
    if not cnt.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    expanded_cnt = _expand_sparsity_tensor(cnt, expected_count_shape, f"{name}_block_cnt")
    expanded_idx = _expand_sparsity_tensor(idx, expected_index_shape, f"{name}_block_idx")
    return expanded_cnt, expanded_idx


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
    *,
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
) -> BlockSparseTensorsTorch:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
    )
    if full_cnt is not None and mask_cnt.device != full_cnt.device:
        raise ValueError("All block sparse tensors must be on the same device")

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsTorch) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def to_cute_block_sparse_tensors(tensors: BlockSparseTensorsTorch) -> Optional[BlockSparseTensors]:
    if not is_block_sparsity_enabled(tensors):
        return None

    mask_block_cnt_tensor = tensors.mask_block_cnt
    mask_block_idx_tensor = tensors.mask_block_idx
    full_block_cnt_tensor = tensors.full_block_cnt
    full_block_idx_tensor = tensors.full_block_idx

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
    )


def compute_block_sparsity(
    config: Config,
    mask_mod_flex: Optional[Callable],
    device: str,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    aux_tensors: Optional[List[torch.Tensor]] = None,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """
    Computes block sparsity tensors from a given masking function.

    This function serves as the main entry point for generating block-sparse masks.
    It dispatches to specialized handlers for variable-length and fixed-length
    sequences.

    Args:
        config: A configuration object containing model and tiling parameters.
        mask_mod_flex: The mask function for generic flex attention patterns.
        device: The device to create tensors on (e.g., 'cuda').
        cu_seqlens_q: Cumulative sequence lengths for Q (for varlen).
        cu_seqlens_k: Cumulative sequence lengths for K (for varlen).
        aux_tensors: A list of auxiliary tensors, e.g., for document masking.

    Returns:
        A tuple of four tensors:
        - `full_block_cnt`: (batch, nheads, n_blocks_q) - Count of full n blocks per m block.
        - `full_block_idx`: (batch, nheads, n_blocks_q, max_n_blocks) - Indices of full n blocks.
        - `mask_block_cnt`: (batch, nheads, n_blocks_q) - Count of partial n blocks per m block.
        - `mask_block_idx`: (batch, nheads, n_blocks_q, max_n_blocks) - Indices of partial n blocks.
        Returns (None, None, None, None) if masking is disabled.
    """
    if not config.use_mask_mod or mask_mod_flex is None:
        return None, None, None, None

    if cu_seqlens_q is not None:
        # Handle variable-length sequences
        return _compute_varlen_sparsity(config, mask_mod_flex, device, cu_seqlens_q, cu_seqlens_k)
    else:
        # Handle fixed-length sequences
        return _compute_sparsity(config, device, aux_tensors)


## ---------------------------------------------------------------------------
## Fixed-Length Sequence Kernels
## ---------------------------------------------------------------------------


def _compute_sparsity(
    config: Config, device: str, aux_tensors: Optional[List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes block sparsity for fixed-length sequences."""
    n_blocks_q = (config.seqlen_q + config.tile_m - 1) // config.tile_m
    n_blocks_k = (config.seqlen_k + config.tile_n - 1) // config.tile_n

    # Pre-allocate output tensors
    full_block_cnt = torch.zeros(
        (config.batch_size, config.nheads, n_blocks_q), device=device, dtype=torch.int32
    )
    mask_block_cnt = torch.zeros(
        (config.batch_size, config.nheads, n_blocks_q), device=device, dtype=torch.int32
    )
    full_block_idx = torch.zeros(
        (config.batch_size, config.nheads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )
    mask_block_idx = torch.zeros(
        (config.batch_size, config.nheads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )

    # --- Identity Mask ---
    # All blocks are fully computed.
    if config.mask_mod_name == "identity":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block_idx in range(n_blocks_q):
            full_block_cnt[:, :, q_block_idx] = n_blocks_k
            full_block_idx[:, :, q_block_idx, :n_blocks_k] = k_blocks

    # --- Identity Partial Mask ---
    # All blocks are partially computed (masked).
    elif config.mask_mod_name == "identity_partial":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block_idx in range(n_blocks_q):
            mask_block_cnt[:, :, q_block_idx] = n_blocks_k
            mask_block_idx[:, :, q_block_idx, :n_blocks_k] = k_blocks

    # --- Block Causal Mask ---
    elif config.mask_mod_name == "block_causal":
        k_blocks = torch.arange(n_blocks_k, device=device)
        for q_block_idx in range(n_blocks_q):
            causal_indices = k_blocks[k_blocks <= q_block_idx]
            num_causal_indices = len(causal_indices)
            if num_causal_indices > 0:
                full_block_cnt[:, :, q_block_idx] = num_causal_indices
                full_block_idx[:, :, q_block_idx, :num_causal_indices] = causal_indices

    # --- Causal and Sliding Window Masks ---
    elif config.mask_mod_name in ["causal", "sliding_window"]:
        q_block_indices = torch.arange(n_blocks_q, device=device)
        k_block_indices = torch.arange(n_blocks_k, device=device)

        q_starts = q_block_indices * config.tile_m
        q_ends = torch.minimum(
            (q_block_indices + 1) * config.tile_m, torch.tensor(config.seqlen_q, device=device)
        )
        k_starts = k_block_indices * config.tile_n
        k_ends = torch.minimum(
            (k_block_indices + 1) * config.tile_n, torch.tensor(config.seqlen_k, device=device)
        )

        # Expand dims for broadcasting: (n_blocks_q, 1) and (1, n_blocks_k)
        q_starts, q_ends = q_starts.unsqueeze(1), q_ends.unsqueeze(1)
        k_starts, k_ends = k_starts.unsqueeze(0), k_ends.unsqueeze(0)

        offset = config.seqlen_k - config.seqlen_q

        if config.mask_mod_name == "causal":
            is_full = (k_ends - 1) <= (q_starts + offset)
            # min(k_pos) <= max(q_pos) AND not is_full.
            is_partial = (k_starts <= (q_ends - 1 + offset)) & ~is_full

        else:  # sliding_window
            window_size = getattr(config, "window_size", 1024)
            is_full = (k_ends - 1 <= q_starts + offset) & (
                k_starts >= q_ends - 1 + offset - (window_size - 1)
            )
            # A block is EMPTY if no (q, k) pairs satisfy the constraint.
            is_empty = (k_starts > q_ends - 1 + offset) | (
                k_ends - 1 < q_starts + offset - (window_size - 1)
            )
            # A block is PARTIAL if it's not empty and not full.
            is_partial = ~is_empty & ~is_full

        # Populate indices based on the computed block classifications
        for q_block_idx in range(n_blocks_q):
            full_indices = k_block_indices[is_full[q_block_idx]]
            if len(full_indices) > 0:
                full_block_cnt[:, :, q_block_idx] = len(full_indices)
                full_block_idx[:, :, q_block_idx, : len(full_indices)] = full_indices

            partial_indices = k_block_indices[is_partial[q_block_idx]]
            if len(partial_indices) > 0:
                mask_block_cnt[:, :, q_block_idx] = len(partial_indices)
                mask_block_idx[:, :, q_block_idx, : len(partial_indices)] = partial_indices

    elif config.mask_mod_name == "document":
        raise NotImplementedError("Block sparsity for document masking not yet implemented")

    return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx


## ---------------------------------------------------------------------------
## Variable-Length Sequence Kernels
## ---------------------------------------------------------------------------


def _compute_varlen_sparsity(
    config: Config,
    mask_mod_flex: Callable,
    device: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes block sparsity for variable-length sequences."""
    assert cu_seqlens_k is not None, "cu_seqlens_k is required for varlen attention"
    assert cu_seqlens_q.shape[0] == config.batch_size + 1
    assert cu_seqlens_k.shape[0] == config.batch_size + 1

    # In varlen, each sequence can have a different number of Q blocks.
    # We pad up to the maximum number of Q blocks in the batch.
    max_m_blocks = 0
    for seq_idx in range(config.batch_size):
        seq_len_q = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]).item()
        n_blocks_q = (seq_len_q + config.tile_m - 1) // config.tile_m
        max_m_blocks = max(max_m_blocks, n_blocks_q)

    # The number of K blocks is determined by the total length of all sequences.
    total_k_len = cu_seqlens_k[-1].item()
    max_n_blocks = (total_k_len + config.tile_n - 1) // config.tile_n

    # Pre-allocate padded output tensors
    full_block_cnt = torch.zeros(
        (config.batch_size, config.nheads, max_m_blocks), device=device, dtype=torch.int32
    )
    mask_block_cnt = torch.zeros(
        (config.batch_size, config.nheads, max_m_blocks), device=device, dtype=torch.int32
    )
    full_block_idx = torch.zeros(
        (config.batch_size, config.nheads, max_m_blocks, max_n_blocks),
        device=device,
        dtype=torch.int32,
    )
    mask_block_idx = torch.zeros(
        (config.batch_size, config.nheads, max_m_blocks, max_n_blocks),
        device=device,
        dtype=torch.int32,
    )

    # Process each sequence in the batch individually
    for seq_idx in range(config.batch_size):
        seq_start_q = cu_seqlens_q[seq_idx].item()
        seq_end_q = cu_seqlens_q[seq_idx + 1].item()
        seq_len_q = seq_end_q - seq_start_q

        seq_start_k = cu_seqlens_k[seq_idx].item()
        seq_end_k = cu_seqlens_k[seq_idx + 1].item()
        seq_len_k = seq_end_k - seq_start_k

        n_blocks_q = (seq_len_q + config.tile_m - 1) // config.tile_m
        n_blocks_k = (seq_len_k + config.tile_n - 1) // config.tile_n

        first_n_block_global = seq_start_k // config.tile_n

        common_args = {
            "full_block_cnt": full_block_cnt,
            "full_block_idx": full_block_idx,
            "mask_block_cnt": mask_block_cnt,
            "mask_block_idx": mask_block_idx,
            "seq_idx": seq_idx,
            "n_blocks_q": n_blocks_q,
            "n_blocks_k": n_blocks_k,
            "seq_start_q": seq_start_q,
            "seq_end_q": seq_end_q,
            "seq_start_k": seq_start_k,
            "seq_end_k": seq_end_k,
            "first_n_block_global": first_n_block_global,
            "tile_m": config.tile_m,
            "tile_n": config.tile_n,
            "device": device,
        }

        if config.mask_mod_name == "causal":
            _compute_causal_varlen_blocks(**common_args)
        elif config.mask_mod_name == "sliding_window":
            window_size = getattr(config, "window_size", 1024)
            _compute_sliding_window_varlen_blocks(**common_args, window_size=window_size)
        elif config.mask_mod_name == "identity":
            _compute_identity_varlen_blocks(
                full_block_cnt,
                full_block_idx,
                seq_idx,
                n_blocks_q,
                n_blocks_k,
                first_n_block_global,
                device,
            )
        else:
            # Generic case relies on sampling the user-provided mask function
            _compute_generic_varlen_blocks(
                **common_args,
                mask_mod_flex=mask_mod_flex,
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
                num_heads=config.nheads,
                nheads_kv=config.nheads_kv,
            )

    return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx


def _classify_varlen_block(
    m_local: int,
    n_local: int,
    seq_start_q: int,
    seq_end_q: int,
    seq_start_k: int,
    seq_end_k: int,
    tile_m: int,
    tile_n: int,
    is_full_fn: Callable,
    is_partial_fn: Callable,
) -> Tuple[bool, bool]:
    """Helper to classify a varlen block as full, partial, or empty."""
    m_start_global = seq_start_q + m_local * tile_m
    m_end_global = min(seq_start_q + (m_local + 1) * tile_m, seq_end_q)
    n_start_global = seq_start_k + n_local * tile_n
    n_end_global = min(seq_start_k + (n_local + 1) * tile_n, seq_end_k)

    # Use sequence-local coordinates for the logical check
    m_start_local = m_start_global - seq_start_q
    m_end_local = m_end_global - seq_start_q
    n_start_local = n_start_global - seq_start_k
    n_end_local = n_end_global - seq_start_k

    is_full = is_full_fn(m_start_local, m_end_local, n_start_local, n_end_local)
    is_partial = (
        is_partial_fn(m_start_local, m_end_local, n_start_local, n_end_local) and not is_full
    )

    # Any block that touches the sequence boundary is partial because it requires masking.
    at_boundary = (m_end_global > seq_end_q) or (n_end_global > seq_end_k)

    return is_full and not at_boundary, is_partial or (is_full and at_boundary)


def _compute_causal_varlen_blocks(
    full_block_cnt,
    full_block_idx,
    mask_block_cnt,
    mask_block_idx,
    seq_idx,
    n_blocks_q,
    n_blocks_k,
    seq_start_q,
    seq_end_q,
    seq_start_k,
    seq_end_k,
    first_n_block_global,
    tile_m,
    tile_n,
    device,
    **kwargs,
):
    """Computes causal block sparsity for a single varlen sequence."""
    is_full_fn = lambda m_start, m_end, n_start, n_end: (m_start >= n_end - 1)
    is_partial_fn = lambda m_start, m_end, n_start, n_end: (m_end - 1 >= n_start)

    for m_local in range(n_blocks_q):
        full_blocks, partial_blocks = [], []
        for n_local in range(n_blocks_k):
            is_full, is_partial = _classify_varlen_block(
                m_local,
                n_local,
                seq_start_q,
                seq_end_q,
                seq_start_k,
                seq_end_k,
                tile_m,
                tile_n,
                is_full_fn,
                is_partial_fn,
            )
            n_block_global = first_n_block_global + n_local
            if is_full:
                full_blocks.append(n_block_global)
            elif is_partial:
                partial_blocks.append(n_block_global)

        if full_blocks:
            full_block_cnt[seq_idx, :, m_local] = len(full_blocks)
            full_block_idx[seq_idx, :, m_local, : len(full_blocks)] = torch.tensor(
                full_blocks, device=device
            )
        if partial_blocks:
            mask_block_cnt[seq_idx, :, m_local] = len(partial_blocks)
            mask_block_idx[seq_idx, :, m_local, : len(partial_blocks)] = torch.tensor(
                partial_blocks, device=device
            )


def _compute_sliding_window_varlen_blocks(
    full_block_cnt,
    full_block_idx,
    mask_block_cnt,
    mask_block_idx,
    seq_idx,
    n_blocks_q,
    n_blocks_k,
    seq_start_q,
    seq_end_q,
    seq_start_k,
    seq_end_k,
    first_n_block_global,
    tile_m,
    tile_n,
    window_size,
    device,
    **kwargs,
):
    """Computes sliding window block sparsity for a single varlen sequence."""
    is_full_fn = lambda m_start, m_end, n_start, n_end: (n_end - 1 <= m_start) and (
        n_start >= m_start - window_size + 1
    )
    is_partial_fn = lambda m_start, m_end, n_start, n_end: not (
        (n_start > m_end - 1) or (n_end - 1 < m_start - window_size + 1)
    )

    for m_local in range(n_blocks_q):
        full_blocks, partial_blocks = [], []
        for n_local in range(n_blocks_k):
            is_full, is_partial = _classify_varlen_block(
                m_local,
                n_local,
                seq_start_q,
                seq_end_q,
                seq_start_k,
                seq_end_k,
                tile_m,
                tile_n,
                is_full_fn,
                is_partial_fn,
            )
            n_block_global = first_n_block_global + n_local
            if is_full:
                full_blocks.append(n_block_global)
            elif is_partial:
                partial_blocks.append(n_block_global)

        if full_blocks:
            full_block_cnt[seq_idx, :, m_local] = len(full_blocks)
            full_block_idx[seq_idx, :, m_local, : len(full_blocks)] = torch.tensor(
                full_blocks, device=device
            )
        if partial_blocks:
            mask_block_cnt[seq_idx, :, m_local] = len(partial_blocks)
            mask_block_idx[seq_idx, :, m_local, : len(partial_blocks)] = torch.tensor(
                partial_blocks, device=device
            )


def _compute_identity_varlen_blocks(
    full_block_cnt,
    full_block_idx,
    seq_idx,
    n_blocks_q,
    n_blocks_k,
    first_n_block_global,
    device,
    **kwargs,
):
    """Computes identity (all-attend) block sparsity for a single varlen sequence."""
    n_blocks_global = torch.arange(
        first_n_block_global, first_n_block_global + n_blocks_k, device=device, dtype=torch.int32
    )
    for m_local in range(n_blocks_q):
        full_block_cnt[seq_idx, :, m_local] = n_blocks_k
        full_block_idx[seq_idx, :, m_local, :n_blocks_k] = n_blocks_global


def _compute_generic_varlen_blocks(
    full_block_cnt,
    full_block_idx,
    mask_block_cnt,
    mask_block_idx,
    mask_mod_flex,
    seq_idx,
    num_heads,
    n_blocks_q,
    n_blocks_k,
    seq_len_q,
    seq_len_k,
    first_n_block_global,
    tile_m,
    tile_n,
    nheads_kv,
    device,
    **kwargs,
):
    """Generic sampling-based block classification for a varlen sequence."""
    for h_q in range(num_heads):
        for m_local in range(n_blocks_q):
            m_start_local = m_local * tile_m
            m_end_local = min((m_local + 1) * tile_m, seq_len_q)

            full_blocks, partial_blocks = [], []
            for n_local in range(n_blocks_k):
                n_start_local = n_local * tile_n
                n_end_local = min((n_local + 1) * tile_n, seq_len_k)

                # Sample points within the block (corners and center) to classify it.
                # Coordinates are sequence-local, as required by mask_mod_flex.
                sample_positions = [
                    (m_start_local, n_start_local),
                    (m_start_local, n_end_local - 1),
                    (m_end_local - 1, n_start_local),
                    (m_end_local - 1, n_end_local - 1),
                    ((m_start_local + m_end_local) // 2, (n_start_local + n_end_local) // 2),
                ]

                unmasked_count = sum(
                    1
                    for q_pos, k_pos in sample_positions
                    if mask_mod_flex(seq_idx, h_q, q_pos, k_pos, seq_len_q, seq_len_k)
                )

                n_block_global = first_n_block_global + n_local
                if unmasked_count == len(sample_positions):  # All samples unmasked -> full
                    full_blocks.append(n_block_global)
                elif unmasked_count > 0:  # Some unmasked -> partial
                    partial_blocks.append(n_block_global)

            if full_blocks:
                full_block_cnt[seq_idx, h_q, m_local] = len(full_blocks)
                full_block_idx[seq_idx, h_q, m_local, : len(full_blocks)] = torch.tensor(
                    full_blocks, device=device
                )
            if partial_blocks:
                mask_block_cnt[seq_idx, h_q, m_local] = len(partial_blocks)
                mask_block_idx[seq_idx, h_q, m_local, : len(partial_blocks)] = torch.tensor(
                    partial_blocks, device=device
                )
