"""Sync-free FlashInfer planning for EAGLE target verify and the multi-step draft.

On replay, FlashInfer's ``BatchPrefillWithPagedKVCacheWrapper.plan`` reads the
qo/kv indptr and last-page lengths back to the host and sizes the packed custom
mask with ``.item()``, and ``FlashInferMultiStepDraftBackend`` copies its draft
``kv_indptr`` rows back with ``.cpu()``. Each read blocks the scheduler until the
GPU has drained the work queued before it, so the planning that follows runs while
the GPU idles. Every value those reads return is a function of ``seq_lens_cpu``,
which the overlap scheduler already holds; the helpers here compute them on the
host and hand them to the same planning calls, so the plan state is unchanged.

Enabled by ``SGLANG_ENABLE_SYNC_FREE_SPEC_PLAN``.
"""

from __future__ import annotations

import weakref
from typing import Optional, Union

import numpy as np
import torch


class _PlanResources:
    def __init__(self) -> None:
        # Recorded after each plan; the next plan of the same wrapper waits on it.
        self.event = torch.cuda.Event()
        # int32: [kv_lens (bs), mask bit offsets (bs + 1), mask byte offsets (bs + 1)].
        self.pinned = torch.empty(0, dtype=torch.int32)
        self.mask_offsets: Optional[torch.Tensor] = None


_RESOURCES: weakref.WeakKeyDictionary[object, _PlanResources] = (
    weakref.WeakKeyDictionary()
)


def _resources(wrapper) -> _PlanResources:
    resources = _RESOURCES.get(wrapper)
    if resources is None:
        resources = _RESOURCES[wrapper] = _PlanResources()
    return resources


def wait_for_previous_plan(wrapper) -> None:
    # FlashInfer's plan writes a per-wrapper pinned buffer and copies it to the device
    # with cudaMemcpyAsync without waiting; the blocking reads used to order the next
    # overwrite after that copy. Normally complete by the time the next plan starts.
    _resources(wrapper).event.synchronize()


def record_plan(wrapper) -> None:
    _resources(wrapper).event.record()


def _stage(resources: _PlanResources, values: np.ndarray) -> torch.Tensor:
    n = values.size
    if resources.pinned.numel() < n:
        resources.pinned = torch.empty(max(n, 1024), dtype=torch.int32).pin_memory()
    resources.pinned.numpy()[:n] = values
    return resources.pinned[:n]


def fast_verify_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: np.ndarray,
    max_q_len: int,
    max_kv_len: int,
    mask_offsets_host: np.ndarray,
    packed_mask_len: int,
) -> None:
    """``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE target-verify graph
    (fa2, cuda-graph mode, custom mask) without device-to-host reads.

    Follows FlashInfer 0.7.0's plan() step by step and leaves the same plan state; the
    host-known layout comes from ``eagle_verify_plan_host_inputs``. The mask's bit and
    byte segment offsets, which plan() derives on the device, are uploaded in one copy.
    The caller orders reuse of this wrapper's pinned buffers with
    ``wait_for_previous_plan`` / ``record_plan``.
    """
    from flashinfer.prefill import _nvfp4_kv_requires_disabled_split_kv
    from flashinfer.quantization.packbits import get_quantization_module
    from flashinfer.utils import canonicalize_torch_dtype

    assert self.is_cuda_graph_enabled, "fast_verify_plan is cuda-graph only"
    assert self._backend == "fa2", "fast_verify_plan supports the fa2 backend only"
    assert self._cached_module is not None, (
        "fast_verify_plan requires _cached_module from a prior real plan() (capture)"
    )
    assert custom_mask is not None, "fast_verify_plan expects the EAGLE verify mask"

    q_data_type = canonicalize_torch_dtype(q_data_type)
    kv_data_type = canonicalize_torch_dtype(
        kv_data_type if kv_data_type is not None else q_data_type
    )
    o_data_type = canonicalize_torch_dtype(
        o_data_type if o_data_type is not None else q_data_type
    )
    if head_dim_vo is None:
        head_dim_vo = head_dim_qk

    batch_size = len(qo_indptr) - 1
    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads

    resources = _resources(self)
    n = batch_size + 1
    staged = _stage(resources, np.concatenate([kv_lens_host, mask_offsets_host]))
    if resources.mask_offsets is None or resources.mask_offsets.numel() < 2 * n:
        resources.mask_offsets = torch.empty(
            2 * n, dtype=torch.int32, device=self.device
        )
    mask_offsets = resources.mask_offsets[: 2 * n]
    mask_offsets.copy_(staged[batch_size:], non_blocking=True)
    packed_custom_mask = torch.empty(
        packed_mask_len, dtype=torch.uint8, device=custom_mask.device
    )
    mask_indptr = mask_offsets[n:]
    get_quantization_module().segment_packbits(
        custom_mask.contiguous().view(-1),
        mask_offsets[:n],
        mask_indptr,
        "little",
        packed_custom_mask,
    )

    if prefix_len_ptr is not None and self._variant_owns_mask:
        raise ValueError(
            "prefix_len_ptr (multi-item scoring) is incompatible with variant_owns_mask"
        )
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    assert batch_size <= self._kv_lens_buffer.shape[0]
    self._kv_lens_buffer[:batch_size].copy_(
        staged[:batch_size], non_blocking=non_blocking
    )
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows
    elif total_num_rows > self._max_total_num_rows:
        raise ValueError(
            f"qo rows {total_num_rows} exceed the captured {self._max_total_num_rows}"
        )
    if batch_size != self._fixed_batch_size:
        raise ValueError(
            f"batch size {batch_size} differs from the captured {self._fixed_batch_size}"
        )
    if len(paged_kv_indices) > len(self._paged_kv_indices_buf):
        raise ValueError("paged_kv_indices exceeds the allocated buffer")

    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )
    self._custom_mask_buf[:packed_mask_len].copy_(
        packed_custom_mask,
        non_blocking=(packed_custom_mask.device == self.device) and non_blocking,
    )
    self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = kv_data_type
    self._cached_o_data_type = o_data_type
    self._block_tables = None

    self._plan_info = self._cached_module.plan(
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        torch.from_numpy(kv_lens_host),
        self._max_total_num_rows or total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        self.is_cuda_graph_enabled,
        head_dim_qk,
        head_dim_vo,
        causal,
        window_left,
        fixed_split_size if fixed_split_size is not None else -1,
        # SGLang never sets disable_split_kv; plan() forces it for NVFP4 KV.
        _nvfp4_kv_requires_disabled_split_kv(kv_data_type, self.device),
        0,  # num_colocated_ctas
        0,  # uniform_q_len
    )

    self._causal = causal
    self._pos_encoding_mode = "NONE"
    self._use_fp16_qk_reduction = False
    self._window_left = window_left
    self._logits_soft_cap = 0.0
    self._sm_scale = None
    self._rope_scale = None
    self._rope_theta = None
    self._seq_lens_kv = None
    self._seq_lens_q = None


def eagle_verify_plan_host_inputs(
    seq_lens_cpu: torch.Tensor, draft_token_num: int, bs: int
) -> dict:
    # Mirrors EagleVerifyInput.generate_attn_arg_prefill (page size 1): qo_indptr is a
    # stride of draft_token_num and kv lengths are seq_lens + draft_token_num; the
    # mask packs draft_token_num * kv_len bits per request into ceil(bits / 8) bytes.
    kv_lens = seq_lens_cpu[:bs].numpy().astype(np.int32) + np.int32(draft_token_num)
    kv_indptr = np.zeros(bs + 1, dtype=np.int32)
    np.cumsum(kv_lens, out=kv_indptr[1:])
    bits = draft_token_num * kv_lens.astype(np.int64)
    mask_offsets = np.zeros(2 * (bs + 1), dtype=np.int64)
    np.cumsum(bits, out=mask_offsets[1 : bs + 1])
    np.cumsum((bits + 7) // 8, out=mask_offsets[bs + 2 :])
    return dict(
        qo_indptr_host=torch.from_numpy(
            np.arange(0, (bs + 1) * draft_token_num, draft_token_num, dtype=np.int32)
        ),
        kv_indptr_host=torch.from_numpy(kv_indptr),
        kv_lens_host=kv_lens,
        max_q_len=draft_token_num,
        max_kv_len=int(kv_lens.max()),
        mask_offsets_host=mask_offsets.astype(np.int32),
        packed_mask_len=int(mask_offsets[-1]),
    )


def draft_kv_indptr_host(
    seq_lens_cpu: torch.Tensor,
    num_seqs: int,
    num_padding: int,
    topk: int,
    num_steps: int,
    window_cap: int,
) -> torch.Tensor:
    """Host copy of the ``kv_indptr`` rows ``generate_draft_decode_kv_indices`` writes.

    Row ``z`` of step ``i`` is ``sum(positions[:z]) + z * (i + 1)``, where a request's
    draft position is its committed length repeated ``topk`` times, capped at
    ``window_cap`` when a draft window is set, and zero on cuda-graph padding rows.
    Returns int32 of shape ``(num_steps, num_seqs * topk + 1)``.
    """
    raw = num_seqs - num_padding
    rows = num_seqs * topk
    positions = np.zeros(rows, dtype=np.int64)
    positions[: raw * topk] = np.repeat(seq_lens_cpu[:raw].numpy(), topk)
    if window_cap > 0:
        np.minimum(positions, window_cap, out=positions)
    out = np.zeros((num_steps, rows + 1), dtype=np.int64)
    out[:, 1:] = np.cumsum(positions) + np.outer(
        np.arange(1, num_steps + 1), np.arange(1, rows + 1)
    )
    return torch.from_numpy(out.astype(np.int32))
