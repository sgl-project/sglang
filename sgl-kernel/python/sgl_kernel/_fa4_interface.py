# Adapted from https://github.com/Dao-AILab/flash-attention/blob/5d4c9537a1e0f1adcc3e4c3e11ae46fe94a18b11/flash_attn/cute/interface.py

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-10-14] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.2.1.


import copy
import gc
import logging
import math
import os
from functools import lru_cache
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn_origin.cute import utils
from flash_attn_origin.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    get_block_sparse_expected_shapes,
    normalize_block_sparse_tensors,
    to_cute_block_sparse_tensors,
)
from flash_attn_origin.cute.flash_fwd import FlashAttentionForwardSm90
from flash_attn_origin.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn_origin.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


@lru_cache(maxsize=None)
def _get_device_capability():
    """Cached device capability check."""
    return torch.cuda.get_device_capability()[0]


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert (
        t.shape == expected_shape
    ), f"{name} shape {t.shape} != expected {expected_shape}"
    assert (
        t.dtype == expected_dtype
    ), f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert (
        t.device == expected_device
    ), f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert (
            page_table.stride(-1) == 1
        ), "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        _validate_tensor(
            out,
            "out",
            (*q_batch_seqlen_shape, num_head, head_dim_v),
            out_torch_dtype,
            device,
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]
    compute_capability = (
        _get_device_capability() if _compute_capability is None else _compute_capability
    )

    assert compute_capability in [
        9,
        10,
        11,
    ], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"

    use_block_sparsity = block_sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
                window_size_right = None
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if (
            head_dim == head_dim_v == 128
            and not causal
            and not local
            and not use_block_sparsity
        ):
            n_block_size = 192

    if compute_capability in [10, 11]:
        if pack_gqa and (128 % qhead_per_kvhead != 0):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if compute_capability == 10:
        q_stage = 2 if seqlen_q_packgqa > m_block_size else 1
    else:
        q_stage = 1

    if num_splits < 1:
        m_block_size_effective = q_stage * m_block_size
        seqlen_k_loaded = (
            max_seqlen_k
            if not local
            else max(
                0,
                min(
                    max_seqlen_k,
                    window_size_right + window_size_left + 1 + m_block_size,
                ),
            )
        )
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (
            seqlen_q_packgqa + m_block_size_effective - 1
        ) // m_block_size_effective
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(
            num_splits, *lse_shape, dtype=torch.float32, device=device
        )

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    if mask_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[1] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        page_size not in [None, 128],  # paged KV non-TMA
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t)
            for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if block_sparse_tensors is not None:
            if seqlen_q is None:
                raise ValueError(
                    "Block sparsity requires fixed-length sequences (seqlen_q must be known)."
                )
            expected_count_shape, expected_index_shape = (
                get_block_sparse_expected_shapes(
                    batch_size,
                    num_head,
                    seqlen_q,
                    seqlen_k,
                    m_block_size,
                    n_block_size,
                    q_stage,
                )
            )
            compile_time_normalized = normalize_block_sparse_tensors(
                block_sparse_tensors,
                expected_count_shape=expected_count_shape,
                expected_index_shape=expected_index_shape,
            )
            sparse_tensors = to_cute_block_sparse_tensors(compile_time_normalized)

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [
                to_cute_tensor(buf, assumed_align=None, fully_dynamic=True)
                for buf in aux_tensors
            ]

        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif compute_capability in [10, 11]:
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                q_stage=q_stage,
                is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None
                and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=page_size not in [None, 128],
                is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x, 11.x"
            )
        # TODO: check @can_implement
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            sparse_tensors,
            cute_aux_tensors,
            options="--enable-tvm-ffi",
        )

    # Expand block sparse tensors to match actual head count (may be broadcast from 1)
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
            batch_size,
            num_head,
            seqlen_q,
            seqlen_k,
            m_block_size,
            n_block_size,
            q_stage,
        )
        normalized_block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
        )
    _flash_attn_fwd.compile_cache[compile_key](
        q,
        k,
        v,
        out if not is_split_kv else out_partial,
        lse_partial if is_split_kv else lse,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        window_size_left,
        window_size_right,
        learnable_sink,
        normalized_block_sparse_tensors,
        aux_tensors,
    )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "out_partial must be fp16, bf16, or fp32"
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert (
        out_partial.stride(-1) == 1
    ), "out_partial must be contiguous in the last dimension"
    assert (
        lse_partial.stride(-2) == 1
    ), "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = (
        8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    )
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        out_partial_tensor = to_cute_tensor(
            out_partial, leading_dim=4 if not is_varlen else 3
        )
        lse_partial_tensor = to_cute_tensor(
            lse_partial, assumed_align=4, leading_dim=lse_partial.ndim - 2
        )
        out_tensor = to_cute_tensor(out, leading_dim=3 if not is_varlen else 2)
        lse_tensor = (
            to_cute_tensor(lse, assumed_align=4, leading_dim=lse.ndim - 2)
            if lse is not None
            else None
        )

        optional_tensors = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
        ]
        (
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
        ) = optional_tensors
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def warmup_flash_attn(f):
    """
    Decorator for flash_attn_varlen_func:
    - On first call, run several warmup passes with different flag combinations:
        * return_softmax_lse in {False, True}
        * global noncausal (window_size=(None,None))
        * causal (window_size=(None,0))
        * local sliding window (window_size=(64,64))
        * optionally pack_gqa=True if qheads > kvheads and allowed
    - No score_mod / softcap (not supported for varlen yet)
    - Executes sequentially to minimize peak GPU mem
    - Does not modify user tensors (clones)
    """
    disable_warmup = os.getenv("SGLANG_DISABLE_FA4_WARMUP", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if disable_warmup:
        return f

    done = False

    def _clone_args(args, kwargs):
        """Clone tensor arguments to avoid sharing storage; deepcopy for others."""

        def maybe_clone(x):
            if isinstance(x, torch.Tensor):
                return x.detach().clone()  # detach to avoid autograd edges
            return copy.deepcopy(x)

        return tuple(maybe_clone(a) for a in args), {
            k: maybe_clone(v) for k, v in kwargs.items()
        }

    def _infer_heads(args, kwargs):
        """Infer q and kv head counts from arguments."""
        # Expect signature: (q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
        q = args[0] if len(args) > 0 else kwargs.get("q")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        try:
            qh = int(q.shape[-2])
            kvh = int(k.shape[-2])
            return qh, kvh
        except Exception:
            return None, None

    def _run_warmups(args, kwargs):
        """Run warmup calls sequentially and release memory after each."""
        base_args, base_kwargs = _clone_args(args, kwargs)

        qh, kvh = _infer_heads(base_args, base_kwargs)
        can_pack_gqa = (
            qh is not None and kvh is not None and qh % kvh == 0 and qh // kvh > 1
        )
        has_page_table = (
            "page_table" in base_kwargs and base_kwargs["page_table"] is not None
        )

        # Window presets covering global, causal, and local
        window_presets = [
            (None, None),  # global noncausal
            (None, 0),  # causal
            (64, 64),  # local sliding window
        ]

        lse_flags = [False, True]

        # Base combo list
        combos = []
        for ws in window_presets:
            for return_lse_flag in lse_flags:
                combos.append(dict(window_size=ws, return_softmax_lse=return_lse_flag))

        # Optionally add a pack_gqa=True variant (FA4 may disable it internally for some varlen shapes/SMs)
        if can_pack_gqa:
            for ws in window_presets:
                combos.append(
                    dict(window_size=ws, return_softmax_lse=False, pack_gqa=True)
                )

        # If page_table is present, warm one combo with it (page_table in compile key for SM100)
        if has_page_table:
            combos.append(dict(window_size=(None, None), return_softmax_lse=False))

        # Run sequentially
        for combo in combos:
            wa, wk = _clone_args(base_args, base_kwargs)
            # Keep user-provided softcap/score_mod OUT (varlen+score_mod unsupported)
            wk.pop("score_mod", None)
            if "softcap" in wk and wk["softcap"]:
                wk["softcap"] = 0.0
            # Apply combo
            wk.update(combo)
            with torch.cuda.stream(torch.cuda.current_stream()):
                try:
                    f(*wa, **wk)
                except Exception as e:
                    # Some combos can be invalid for specific head dims / arch. Ignore and continue.
                    logger.debug("Warmup combo skipped: %s", e)
            del wa, wk
            torch.cuda.empty_cache()
            gc.collect()

    def wrapper(*args, **kwargs):
        nonlocal done
        if not done:
            logger.info(
                "Running FA4 warmup (global/causal/local, LSE on/off, optional GQA pack)..."
            )
            _run_warmups(args, kwargs)
            done = True
        return f(*args, **kwargs)

    return wrapper


@warmup_flash_attn
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    return_softmax_lse: Optional[bool] = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_lse=return_softmax_lse,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
    )

    return (out, lse) if return_softmax_lse else out
