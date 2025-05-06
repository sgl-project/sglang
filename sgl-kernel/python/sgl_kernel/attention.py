from typing import Optional, Tuple

import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    r"""
    Perform decoding with Lightning Attention, integrating current and past key/value states.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor for the current decoding step. Shape is typically
        ``(batch_size, num_heads, 1, head_dim)``.
    k : torch.Tensor
        Key tensor for the current decoding step. Shape is typically
        ``(batch_size, num_heads, 1, head_dim)``.
    v : torch.Tensor
        Value tensor for the current decoding step. Shape is typically
        ``(batch_size, num_heads, 1, head_dim)``.
    past_kv : torch.Tensor
        Past key and value tensor, storing previous decoding steps.
        Shape is typically ``(batch_size, num_heads, seq_len, head_dim * 2)``,
        where the last dimension contains concatenated keys and values.
    slope : torch.Tensor or float
        Slope parameter(s) for positional bias or rotary embedding.
        Shape or type depends on the attention mechanism and implementation.
    output : torch.Tensor
        Output tensor to store the attention result for the current step.
        Should have shape ``(batch_size, num_heads, 1, head_dim)``.
    new_kv : torch.Tensor
        Tensor to store the updated key and value after incorporating
        the current step. Typically has the same shape as `past_kv`.

    Returns
    -------
    None

    Note
    ----
    This function is typically used in transformer decoding scenarios where
    Lightning Attention is employed for efficient and fast inference.
    It updates the key/value cache (`past_kv` and `new_kv`) as part of the
    autoregressive decoding process.
    """
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Merge two state representations, combining their value and score tensors.

    Parameters
    ----------
    v_a : torch.Tensor
        The value tensor from state A. Shape and dtype depend on the specific
        application (e.g., could be key/value cache, hidden states, etc.).
    s_a : torch.Tensor
        The score tensor from state A, representing associated scores, weights,
        or attention values.
    v_b : torch.Tensor
        The value tensor from state B, with the same shape and dtype as `v_a`.
    s_b : torch.Tensor
        The score tensor from state B, with the same shape and dtype as `s_a`.
    v_merged : Optional[torch.Tensor], default = None
        Optional pre-allocated output tensor to store the merged value result.
        If provided, the merged value is written to this tensor in-place.
        Otherwise, a new tensor is allocated and returned.
    s_merged : Optional[torch.Tensor], default = None
        Optional pre-allocated output tensor to store the merged score result.
        If provided, the merged score is written to this tensor in-place.
        Otherwise, a new tensor is allocated and returned.

    Returns
    -------
    v_merged : torch.Tensor
        The merged value tensor, representing the combination of `v_a` and `v_b`.
    s_merged : torch.Tensor
        The merged score tensor, representing the combination of `s_a` and `s_b`.

    Note
    ----
    The merging logic depends on the specific application. Typical strategies
    include weighted averaging, element-wise maximum, or concatenation of the
    input states. Pre-allocated output tensors (`v_merged` and `s_merged`) can
    be used for memory efficiency in high-performance scenarios.
    """
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Merge two state representations v2, combining their value and score tensors.

    Parameters
    ----------
    v_a : torch.Tensor
        The value tensor from state A. Shape and dtype depend on the specific
        application (e.g., could be key/value cache, hidden states, etc.).
    s_a : torch.Tensor
        The score tensor from state A, representing associated scores, weights,
        or attention values.
    v_b : torch.Tensor
        The value tensor from state B, with the same shape and dtype as `v_a`.
    s_b : torch.Tensor
        The score tensor from state B, with the same shape and dtype as `s_a`.
    v_merged : Optional[torch.Tensor], default = None
        Optional pre-allocated output tensor to store the merged value result.
        If provided, the merged value is written to this tensor in-place.
        Otherwise, a new tensor is allocated and returned.
    s_merged : Optional[torch.Tensor], default = None
        Optional pre-allocated output tensor to store the merged score result.
        If provided, the merged score is written to this tensor in-place.
        Otherwise, a new tensor is allocated and returned.

    Returns
    -------
    v_merged : torch.Tensor
        The merged value tensor, representing the combination of `v_a` and `v_b`.
    s_merged : torch.Tensor
        The merged score tensor, representing the combination of `s_a` and `s_b`.

    Note
    ----
    The merging logic depends on the specific application. Typical strategies
    include weighted averaging, element-wise maximum, or concatenation of the
    input states. Pre-allocated output tensors (`v_merged` and `s_merged`) can
    be used for memory efficiency in high-performance scenarios.
    """
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def cutlass_mla_decode(
    q_nope_and_q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    r"""
    Perform decoding using CUTLASS-accelerated MLA.

    Parameters
    ----------
    q_nope_and_q_pe : torch.Tensor
        Query tensor containing both non-positional encoded (nope) and positional encoded (pe)
        query representations for the current decoding step.
        Shape: typically ``(batch_size, num_heads, 1, head_dim * 2)``, where the last
        dimension contains concatenated nope and pe representations.
    kv_c_and_k_pe_cache : torch.Tensor
        Cached key/value tensors and key positional encoding cache from previous decoding steps.
        Shape: typically ``(batch_size, num_heads, max_seq_len, head_dim * 2)``, where the
        last dimension contains concatenated key/value and positional encodings.
    seq_lens : torch.Tensor
        Tensor containing the sequence lengths for each item in the batch,
        used to determine valid attention ranges.
        Shape: ``(batch_size,)``.
    page_table : torch.Tensor
        Tensor representing a mapping table for memory pages, used to manage
        dynamic memory access or efficient cache utilization during decoding.
        Shape and dtype depend on the implementation details (e.g., ``(batch_size, num_pages)``).
    workspace : torch.Tensor
        Temporary workspace tensor for intermediate computations and memory
        required during the decoding process. Should be allocated with sufficient size.

    Returns
    -------
    output : torch.Tensor
        Output tensor containing the decoded results for the current step.
        Shape: typically ``(batch_size, num_heads, 1, head_dim)``.

    Note
    ----
    This function leverages CUTLASS kernels for high-performance Multi-Query Lightning
    Attention decoding, supporting efficient handling of positional encodings and
    dynamic sequence lengths. The `workspace` tensor is used for temporary storage
    and must be properly sized for the batch and sequence configuration.
    """
    assert (
        q_nope_and_q_pe.ndim == 3
    ), f"q_nope_and_q_pe must be a 3D tensor, but got {q_nope_and_q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"
    B_q, H, D_q = q_nope_and_q_pe.shape
    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q == D_ckv and D_q == D_latent + D_rope, (
        f"D_q must be equal to D_ckv and D_q must be equal to D_latent + D_rope, "
        f"but got D_q = {D_q}, D_ckv = {D_ckv}, D_latent = {D_latent}, D_rope = {D_rope}"
    )
    assert H == 128, f"H must be 128, but got {H}"

    assert len(page_table.shape) == 2
    B_block_table, block_num = page_table.shape
    assert B_block_table == B_q
    assert block_num > 0, f"block num must be greater than 0, got {block_num}"
    assert block_num % (128 / PAGE_SIZE) == 0

    # TODO(kaixih@nvidia): support fp8
    assert q_nope_and_q_pe.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope_and_q_pe.dtype needs to be fp16 or bf16 but got {q_nope_and_q_pe.dtype}."
    assert kv_c_and_k_pe_cache.dtype == q_nope_and_q_pe.dtype, (
        f"kv_c_and_k_pe_cache.dtype needs to be the same as q_nope_and_q_pe.dtype, "
        f"but got {kv_c_and_k_pe_cache.dtype}."
    )
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
    assert (
        page_table.dtype == torch.int32
    ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

    out = torch.empty(
        (B_q, H, D_latent), device=q_nope_and_q_pe.device, dtype=q_nope_and_q_pe.dtype
    )

    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace
    )
    return out


def cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int = 0
) -> int:
    r"""
    Calculate the required workspace size (in bytes) for CUTLASS MLA decoding.

    Parameters
    ----------
    max_seq_len : int
        The maximum sequence length expected during decoding. This determines
        the upper bound for the memory allocation required by the workspace.
    num_batches : int
        The number of batch elements to be processed in parallel. Workspace size
        scales linearly with the batch size.
    sm_count : int, optional, default = 0
        The number of Streaming Multiprocessors (SMs) available or to be used on
        the GPU. If set to 0, the function may use the device's default or query
        the hardware for the actual count. This parameter allows for tuning the
        workspace size according to parallelism on the hardware.

    Returns
    -------
    workspace_size : int
        The required workspace size in bytes. This value should be used to
        allocate the `workspace` tensor for `cutlass_mla_decode` to ensure
        sufficient temporary memory is available for all computations.

    Note
    ----
    The workspace size depends on the model configuration and hardware parameters.
    Allocating insufficient workspace may result in runtime errors or degraded
    performance. Always use this function to determine the correct buffer size
    before calling CUTLASS MLA decoding functions.

    Assertions
    ----------
    Ensures that the input parameters are valid:
    - `max_seq_len` must be greater than 0.
    - `num_batches` must be greater than 0.
    """

    assert max_seq_len > 0, f"max_seq_len must be greater than 0, got {max_seq_len}"
    assert num_batches > 0, f"num_batches must be greater than 0, got {num_batches}"

    return torch.ops.sgl_kernel.cutlass_mla_get_workspace_size.default(
        max_seq_len, num_batches, sm_count
    )
